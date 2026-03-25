from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .io_utils import ensure_dir


DEFAULT_SYNC_PATHS = [
    "apps/isaac",
    "src/dashgo_rl",
    "configs/training",
    "autopilot",
    "tools/diagnostics",
    "tests",
]

DEFAULT_OVERRIDE_REL_PATH = "configs/training/autoresearch_active_overrides.json"


def git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def git_output(*args: str, cwd: Path) -> str:
    return git(*args, cwd=cwd).stdout.strip()


def branch_exists(repo_root: Path, branch: str) -> bool:
    result = git("branch", "--list", branch, cwd=repo_root, check=False)
    return bool(result.stdout.strip())


def ensure_branch(repo_root: Path, branch: str, *, base_ref: str = "HEAD") -> None:
    if branch_exists(repo_root, branch):
        return
    git("branch", branch, base_ref, cwd=repo_root)


def worktree_initialized(worktree_root: Path) -> bool:
    return (worktree_root / ".git").exists()


def sync_path(source_root: Path, worktree_root: Path, relative_path: str) -> None:
    source = source_root / relative_path
    target = worktree_root / relative_path
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    else:
        shutil.copy2(source, target)


def sync_training_paths(source_root: Path, worktree_root: Path, relative_paths: list[str] | None = None) -> list[str]:
    synced: list[str] = []
    for item in relative_paths or DEFAULT_SYNC_PATHS:
        sync_path(source_root, worktree_root, item)
        synced.append(item)
    return synced


def ensure_worktree(
    *,
    repo_root: Path,
    worktree_root: Path,
    branch: str,
    sync_paths: list[str] | None = None,
) -> dict[str, str | bool | list[str]]:
    ensure_dir(worktree_root.parent)
    ensure_branch(repo_root, branch)
    created = False
    if not worktree_initialized(worktree_root):
        git("worktree", "add", "--force", str(worktree_root), branch, cwd=repo_root)
        created = True
    synced = sync_training_paths(repo_root, worktree_root, sync_paths)
    status = git_output("status", "--short", cwd=worktree_root)
    if status:
        git("add", *synced, cwd=worktree_root)
        git(
            "-c",
            "user.name=Codex",
            "-c",
            "user.email=codex@local",
            "commit",
            "-m",
            "experiment: baseline sync",
            cwd=worktree_root,
            check=False,
        )
    return {
        "created": created,
        "worktree_root": str(worktree_root),
        "branch": branch,
        "head": current_head(worktree_root),
        "synced_paths": synced,
    }


def current_head(worktree_root: Path) -> str:
    return git_output("rev-parse", "HEAD", cwd=worktree_root)


def diff_head_patch(worktree_root: Path) -> str:
    return git_output("show", "--stat", "--patch", "HEAD", cwd=worktree_root)


def restore_best_commit(worktree_root: Path, best_commit: str) -> None:
    git("reset", "--hard", best_commit, cwd=worktree_root)
    git("clean", "-fd", cwd=worktree_root)


def write_override_profile(worktree_root: Path, payload: dict) -> Path:
    target = worktree_root / DEFAULT_OVERRIDE_REL_PATH
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def commit_experiment_change(worktree_root: Path, *, message: str) -> str:
    git("add", *DEFAULT_SYNC_PATHS, cwd=worktree_root)
    git(
        "-c",
        "user.name=Codex",
        "-c",
        "user.email=codex@local",
        "commit",
        "-m",
        message,
        cwd=worktree_root,
    )
    return current_head(worktree_root)


def working_tree_dirty(worktree_root: Path) -> bool:
    return bool(git_output("status", "--short", cwd=worktree_root))
