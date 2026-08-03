"""Undefined-name gate: pyflakes over every fyp module (skips if pyflakes
absent). Complements test_import_graph.py - imports can be valid while a BARE
NAME inside a function is not (the INPUT_DIM incident, pipeline.py:444, found
only at full-run time on Kaggle). This makes that class of bug a test failure."""
import os
import subprocess
import sys

FYP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyflakes_undefined():
    try:
        import pyflakes  # noqa: F401
    except ImportError:
        return None                      # not installed -> skip
    files = [os.path.join(FYP_DIR, f) for f in sorted(os.listdir(FYP_DIR))
             if f.endswith(".py")]
    r = subprocess.run([sys.executable, "-m", "pyflakes"] + files,
                       capture_output=True, text=True)
    return [ln for ln in (r.stdout + r.stderr).splitlines()
            if "undefined name" in ln]


def test_no_undefined_names():
    hits = _pyflakes_undefined()
    if hits is None:
        import pytest
        pytest.skip("pyflakes not installed")
    assert not hits, "\n".join(["undefined names:"] + hits)


def _run_all():
    hits = _pyflakes_undefined()
    if hits is None:
        print("  SKIP  pyflakes not installed")
        return
    if hits:
        print("UNDEFINED NAMES:")
        for h in hits:
            print("  -", h)
        sys.exit(1)
    print("  PASS  no undefined names in any fyp module")


if __name__ == "__main__":
    _run_all()
