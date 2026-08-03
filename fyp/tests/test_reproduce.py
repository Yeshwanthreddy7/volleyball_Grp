"""
Tests for the reproduction harness (reproduce.py).

The harness is what a panel runs to verify every claim, so its own contract
matters: the stage registry must be complete, the check mechanism must actually
record failures (a gate that always passes is worse than no gate), and
METRICS.md must state the label-provenance caveat rather than presenting
agreement-with-a-heuristic as accuracy.

No video, no weights, no GPU.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "fyp"))

_spec = importlib.util.spec_from_file_location(
    "reproduce", os.path.join(ROOT, "reproduce.py"))
reproduce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reproduce)


def test_every_stage_is_registered_and_callable():
    assert set(reproduce.STAGES) == {0, 1, 2, 3, 4, 5, 6}
    assert all(callable(f) for f in reproduce.STAGES.values())


def test_check_records_failures_and_passes_cleanly(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    assert ctx.failures == []

    assert reproduce.check(ctx, True, "a property that holds") is True
    assert ctx.failures == []

    assert reproduce.check(ctx, False, "a property that does not hold") is False
    assert ctx.failures == ["a property that does not hold"]


def test_sha_is_stable_and_none_for_missing_files(tmp_path):
    f = tmp_path / "artefact.bin"
    f.write_bytes(b"volleyball")
    a = reproduce._sha(str(f))
    assert a == reproduce._sha(str(f))          # deterministic
    assert len(a) == 12
    assert reproduce._sha(str(tmp_path / "nope.bin")) is None


def test_sha_changes_when_the_artefact_changes(tmp_path):
    """Provenance is worthless if two different weight files hash the same."""
    f = tmp_path / "w.pt"
    f.write_bytes(b"model-A")
    a = reproduce._sha(str(f))
    f.write_bytes(b"model-B")
    assert reproduce._sha(str(f)) != a


def test_ctx_creates_the_figure_directory(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path / "out"), quick=False)
    assert os.path.isdir(ctx.figdir)
    assert ctx.fig("x.png").endswith(os.path.join("figures", "x.png"))


def test_metrics_document_states_the_label_provenance_caveat(tmp_path):
    """The headline scores measure agreement with a rule engine. If METRICS.md
    ever stops saying so, the document misrepresents the work."""
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    ctx.results["provenance"] = {"generated_at": "now", "packages": {},
                                 "artefacts": {}}
    reproduce.write_metrics(ctx)

    text = open(os.path.join(str(tmp_path), "METRICS.md"), encoding="utf-8").read()
    assert "label_clips.py" in text
    assert "agreement with that" in text.lower() or "agreement with a" in text.lower()
    assert "not tactical correctness" in text.lower()


def test_metrics_document_surfaces_failures(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    ctx.failures.append("detector saw no players")
    reproduce.write_metrics(ctx)

    text = open(os.path.join(str(tmp_path), "METRICS.md"), encoding="utf-8").read()
    assert "CHECK(S) FAILED" in text
    assert "detector saw no players" in text


def test_metrics_document_reports_a_clean_run(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    reproduce.write_metrics(ctx)
    text = open(os.path.join(str(tmp_path), "METRICS.md"), encoding="utf-8").read()
    assert "All automated checks passed" in text


def test_video_stages_are_declared_so_skip_video_is_meaningful():
    src = open(os.path.join(ROOT, "reproduce.py"), encoding="utf-8").read()
    assert "video_stages = {2, 3}" in src


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
