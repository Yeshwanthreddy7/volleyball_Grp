"""
mamba_model.py – Mamba-based Sequence Classifier for Volleyball Analytics.

Implements the Mamba selective state-space model (SSM) in pure PyTorch
(no CUDA-specific kernels required) and wraps it in a sequence classifier
suitable for the 29-frame volleyball clip dataset produced by label_clips.py.

Architecture overview
---------------------
Input  : (batch, seq_len=29, input_dim=14)
          14 = ball_x, ball_y + p1_x, p1_y, …, p6_x, p6_y

Encoder: Linear projection → N × MambaBlock → LayerNorm

Head   : Mean-pool over time → Linear → 4-class softmax

References
----------
Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with
Selective State Spaces. arXiv:2312.00752.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Selective State Space Model (S6) – core Mamba building block
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    One Mamba block with selective scan.

    Parameters
    ----------
    d_model  : model (inner) dimension
    d_state  : SSM state dimension N  (paper default: 16)
    d_conv   : local depthwise-conv width (paper default: 4)
    dt_rank  : rank of the Δ projection  (default: ceil(d_model/16))
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # ── Input expansion ────────────────────────────────────────────────
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # ── Local context: depthwise conv along the time axis ─────────────
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
            bias=True,
        )

        # ── SSM parameter projections ─────────────────────────────────────
        # Maps inner activation → (dt_rank + 2*d_state)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # dt projection: rank → d_model, with bias initialised so softplus(dt) ≈ 1
        dt_init_std = self.dt_rank ** -0.5
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Bias: softplus^{-1}(U[0.001, 0.1])
        dt_bias = torch.empty(d_model).uniform_(0.001, 0.1)
        self.dt_proj.bias = nn.Parameter(torch.log(torch.expm1(dt_bias)))

        # A: fixed diagonal state-transition matrix (learnable in log space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, N)
        A = A.expand(d_model, -1)  # (D, N)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # ── Output projection ──────────────────────────────────────────────
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------------
    # Selective scan (sequential implementation; works without CUDA ext)
    # ------------------------------------------------------------------

    def _selective_scan(
        self,
        x: torch.Tensor,      # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,      # (D, N)
        B: torch.Tensor,      # (B, L, N)
        C: torch.Tensor,      # (B, L, N)
    ) -> torch.Tensor:
        """Compute the discretised SSM recurrence and return (B, L, D)."""
        B_sz, L, D = x.shape
        N = A.shape[1]

        # Discretise with zero-order hold (ZOH)
        # Ā  = exp(Δ ⊗ A)              shape: (B, L, D, N)
        # B̄x = Δ ⊗ B ⊗ x              shape: (B, L, D, N)
        delta_A = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, D, N)
        delta_Bx = (
            delta.unsqueeze(-1)               # (B, L, D, 1)
            * B.unsqueeze(2)                  # (B, L, 1, N)
            * x.unsqueeze(-1)                 # (B, L, D, 1)
        )  # (B, L, D, N)

        # Sequential scan
        h = x.new_zeros(B_sz, D, N)
        ys: list[torch.Tensor] = []
        for t in range(L):
            h = delta_A[:, t] * h + delta_Bx[:, t]   # (B, D, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, D)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_model)

        Returns
        -------
        out : (B, L, d_model)
        """
        B, L, _ = x.shape

        # Expand then gate
        xz = self.in_proj(x)                  # (B, L, 2*D)
        x_in, z = xz.chunk(2, dim=-1)         # each (B, L, D)

        # Depthwise conv along time axis (causal: trim right padding)
        x_conv = x_in.transpose(1, 2)          # (B, D, L)
        x_conv = self.conv1d(x_conv)[..., :L]  # causal crop
        x_conv = x_conv.transpose(1, 2)        # (B, L, D)
        x_conv = F.silu(x_conv)

        # SSM parameter projections
        bcdt = self.x_proj(x_conv)             # (B, L, dt_rank + 2*N)
        dt_rank = self.dt_rank
        delta_raw = bcdt[..., :dt_rank]        # (B, L, dt_rank)
        B_mat = bcdt[..., dt_rank: dt_rank + self.d_state]   # (B, L, N)
        C_mat = bcdt[..., dt_rank + self.d_state:]           # (B, L, N)

        # dt
        delta = F.softplus(self.dt_proj(delta_raw))          # (B, L, D)

        A = -torch.exp(self.A_log.float())     # (D, N)

        # SSM recurrence
        y = self._selective_scan(x_conv, delta, A, B_mat, C_mat)  # (B, L, D)

        # Skip connection with D
        y = y + x_conv * self.D               # (B, L, D)

        # Gate with SiLU of z-branch
        y = y * F.silu(z)                     # (B, L, D)

        return self.out_proj(y)               # (B, L, D)


# ---------------------------------------------------------------------------
# Mamba Block  (SSM + residual + layer-norm)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# Full Mamba Sequence Classifier
# ---------------------------------------------------------------------------

# Label index → string mapping  (must match label_clips.py)
LABEL_NAMES = [
    "Spacing Breakdown",
    "Delayed Support",
    "Coordinated Attack",
    "Coordinated Defense",
]
NUM_CLASSES = len(LABEL_NAMES)

# Feature columns expected in each CSV row (same order as label_clips.py)
FEATURE_COLS = [
    "ball_x", "ball_y",
    "p1_x", "p1_y",
    "p2_x", "p2_y",
    "p3_x", "p3_y",
    "p4_x", "p4_y",
    "p5_x", "p5_y",
    "p6_x", "p6_y",
]
INPUT_DIM = len(FEATURE_COLS)  # 14


class MambaClassifier(nn.Module):
    """
    Mamba-based sequence classifier for volleyball clip labeling.

    Parameters
    ----------
    input_dim   : number of input features per frame (default 14)
    d_model     : model dimension (default 64)
    n_layers    : number of stacked Mamba blocks (default 4)
    d_state     : SSM state dimension per block (default 16)
    d_conv      : local conv width per block (default 4)
    num_classes : number of output classes (default 4)
    dropout     : dropout rate on the classifier head (default 0.1)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Input embedding ───────────────────────────────────────────────
        self.embed = nn.Linear(input_dim, d_model)

        # ── Stack of Mamba blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state, d_conv=d_conv) for _ in range(n_layers)]
        )

        # ── Final layer-norm ──────────────────────────────────────────────
        self.norm = nn.LayerNorm(d_model)

        # ── Classifier head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        logits : (batch, num_classes)
        """
        h = self.embed(x)                   # (B, L, d_model)
        for block in self.blocks:
            h = block(h)                    # (B, L, d_model)
        h = self.norm(h)                    # (B, L, d_model)
        h = h.mean(dim=1)                   # (B, d_model) – global avg pool
        return self.head(h)                 # (B, num_classes)

    # ------------------------------------------------------------------
    # Convenience: predict a single clip tensor
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[str, torch.Tensor]:
        """
        Predict the label for one clip.

        Parameters
        ----------
        x : (seq_len, input_dim)  or  (1, seq_len, input_dim)

        Returns
        -------
        label      : predicted class string
        probs      : (num_classes,) probability tensor
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        logits = self(x)                    # (1, num_classes)
        probs = F.softmax(logits, dim=-1)[0]
        label = LABEL_NAMES[probs.argmax().item()]
        return label, probs
