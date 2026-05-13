from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _line_of_import(tree: ast.Module, module_name: str) -> int | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name:
                    return node.lineno
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            return node.lineno
    return None


def test_torch_imports_after_app_launcher_start_for_isaac_entrypoints() -> None:
    for relative_path in (
        "apps/isaac/play.py",
        "apps/isaac/export_torchscript.py",
        "apps/isaac/train_v2.py",
        "apps/isaac/verify_ultimate_v5.py",
    ):
        tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
        torch_line = _line_of_import(tree, "torch")
        app_launcher_assignment = next(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "simulation_app" for target in node.targets)
        )

        assert torch_line is not None
        assert torch_line > app_launcher_assignment, relative_path
