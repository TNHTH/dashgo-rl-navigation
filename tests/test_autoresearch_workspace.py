from __future__ import annotations

import subprocess
from pathlib import Path

from autopilot import autoresearch_workspace as workspace


def git_init_repo(repo_root: Path) -> None:
    (repo_root / "README.md").write_text("dashgo test repo\n", encoding="utf-8")
    (repo_root / "apps" / "isaac").mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "dashgo_rl").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "training").mkdir(parents=True, exist_ok=True)
    (repo_root / "autopilot").mkdir(parents=True, exist_ok=True)
    (repo_root / "tools" / "diagnostics").mkdir(parents=True, exist_ok=True)
    (repo_root / "tests").mkdir(parents=True, exist_ok=True)
    (repo_root / "apps" / "isaac" / "train_v2.py").write_text("print('train')\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Tester", "-c", "user.email=tester@example.com", "commit", "-m", "init"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def test_ensure_worktree_creates_branch_and_commits_baseline(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_init_repo(repo_root)
    (repo_root / "configs" / "training" / "train_cfg_v2.yaml").write_text("runner:\n  max_iterations: 10\n", encoding="utf-8")

    worktree_root = tmp_path / "worktree"
    payload = workspace.ensure_worktree(
        repo_root=repo_root,
        worktree_root=worktree_root,
        branch="autotrain/autoresearch",
        sync_paths=workspace.DEFAULT_SYNC_PATHS,
    )

    assert payload["branch"] == "autotrain/autoresearch"
    assert worktree_root.exists()
    assert workspace.current_head(worktree_root)


def test_override_profile_commit_and_restore(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    git_init_repo(repo_root)
    worktree_root = tmp_path / "worktree"
    workspace.ensure_worktree(
        repo_root=repo_root,
        worktree_root=worktree_root,
        branch="autotrain/autoresearch",
        sync_paths=workspace.DEFAULT_SYNC_PATHS,
    )
    baseline = workspace.current_head(worktree_root)
    profile_path = workspace.write_override_profile(
        worktree_root,
        {"idea_id": "reward.orbit_weight.up_4_0", "env": {"DASHGO_ORBIT_WEIGHT": "4.0"}, "config": {}},
    )
    assert profile_path.exists()
    commit = workspace.commit_experiment_change(worktree_root, message="experiment: test profile")
    assert commit != baseline
    workspace.restore_best_commit(worktree_root, baseline)
    assert workspace.current_head(worktree_root) == baseline
