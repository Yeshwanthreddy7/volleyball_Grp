"""Import-graph regression test - catches stale cross-module imports WITHOUT
importing anything (pure AST), so it runs on machines without torch/cv2.

Motivation: after the July refactor, prepare_training_data.py still imported
_build_homography/_detect/_track from pipeline.py - names that no longer
existed. Local unit tests missed it because importing pipeline needs torch;
the bug only surfaced at full-run time (on Kaggle). This test parses every
module's top-level bindings and verifies every `from <local module> import X`
against them, torch-free.
"""
import ast
import os
import sys

FYP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _top_level_bindings(tree):
    """All names bound at module top level (incl. inside top-level if/try)."""
    names = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name != "*":
                        names.add(a.asname or a.name)
            elif isinstance(node, (ast.If, ast.Try)):
                visit(node.body)
                for h in getattr(node, "handlers", []):
                    visit(h.body)
                visit(getattr(node, "orelse", []))
                visit(getattr(node, "finalbody", []))
    visit(tree.body)
    return names


def _local_modules():
    mods = {}
    for f in os.listdir(FYP_DIR):
        if f.endswith(".py"):
            path = os.path.join(FYP_DIR, f)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                try:
                    mods[f[:-3]] = ast.parse(fh.read(), filename=f)
                except SyntaxError as e:
                    raise AssertionError("syntax error in {}: {}".format(f, e))
    return mods


def check_import_graph():
    mods = _local_modules()
    bindings = {m: _top_level_bindings(t) for m, t in mods.items()}
    problems = []
    for mod, tree in mods.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 \
                    and node.module in bindings:
                for a in node.names:
                    if a.name != "*" and a.name not in bindings[node.module]:
                        problems.append(
                            "{}.py imports '{}' from {}.py - not defined there"
                            .format(mod, a.name, node.module))
    return problems


def test_no_stale_cross_module_imports():
    problems = check_import_graph()
    assert not problems, "\n".join(["stale imports:"] + problems)


def _run_all():
    problems = check_import_graph()
    if problems:
        print("STALE IMPORTS FOUND:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("  PASS  import graph clean ({} local modules)".format(
        len(_local_modules())))


if __name__ == "__main__":
    _run_all()
