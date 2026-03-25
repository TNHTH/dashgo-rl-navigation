from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in os.sys.path:
        os.sys.path.insert(0, candidate_str)

from autopilot.io_utils import ensure_dir, read_json, write_json
from dashgo_rl.project_paths import ROS1_PACKAGE_ROOT, ROS2_PACKAGE_ROOT, ROS2_WS_ROOT

EXPECTED_OBS_TERM_ORDER = [
    "lidar_history",
    "waypoint_vector_history",
    "goal_vector_history",
    "lin_vel_x_history",
    "yaw_rate_history",
    "last_action_history",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo 模型归档、staging、promote 与 rollback 工具")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage-only", action="store_true", help="只归档候选模型，不覆盖 ROS 包内模型")
    mode.add_argument("--promote", action="store_true", help="将候选模型部署到 ROS1/ROS2 包目录")
    mode.add_argument("--rollback", metavar="DEPLOYMENT_ID", help="回滚到指定 deployment 的 before 快照")
    parser.add_argument("--source-model", type=Path, help="候选模型路径")
    parser.add_argument("--source-manifest", type=Path, help="候选 manifest 路径")
    parser.add_argument("--label", default="", help="deployment 标签，仅用于元数据")
    parser.add_argument("--note", default="", help="deployment 备注")
    parser.add_argument("--dry-run", action="store_true", help="只写 deployment 元数据，不改线上模型")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="项目根目录")
    return parser


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}.manifest.json")


def deployment_root_for(project_root: Path, deployment_id: str) -> Path:
    return project_root / ".artifacts" / "autopilot" / "deployments" / deployment_id


def new_deployment_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def default_target_model_paths() -> list[Path]:
    targets = [
        ROS2_PACKAGE_ROOT / "models" / "policy_torchscript.pt",
        ROS1_PACKAGE_ROOT / "models" / "policy_torchscript.pt",
    ]
    ros2_install_model = ROS2_WS_ROOT / "install" / "dashgo_rl_ros2" / "share" / "dashgo_rl_ros2" / "models" / "policy_torchscript.pt"
    if ros2_install_model.parent.exists():
        targets.append(ros2_install_model)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in targets:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path, default=None)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest 非法或不存在: {path}")
    return payload


def validate_manifest(source_model: Path, manifest: dict[str, Any], *, allow_trace: bool) -> list[str]:
    errors: list[str] = []
    if manifest.get("action_semantics") != "bounded_tanh_gaussian":
        errors.append("manifest.action_semantics 必须为 bounded_tanh_gaussian")
    if int(manifest.get("obs_dim", -1)) != 246:
        errors.append("manifest.obs_dim 必须为 246")
    if manifest.get("obs_term_order") != EXPECTED_OBS_TERM_ORDER:
        errors.append("manifest.obs_term_order 与当前部署合同不一致")
    export_mode = str(manifest.get("export_mode") or "")
    if export_mode != "script" and not allow_trace:
        errors.append("manifest.export_mode 必须为 script，trace 只允许 staging")

    torchscript_entries = manifest.get("torchscript") or []
    if not isinstance(torchscript_entries, list) or not torchscript_entries:
        errors.append("manifest.torchscript 不能为空")
    else:
        source_sha = file_sha256(source_model)
        matched = False
        for entry in torchscript_entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("sha256") == source_sha:
                matched = True
                break
        if not matched:
            errors.append("source model sha256 未在 manifest.torchscript 中找到匹配项")
    return errors


def snapshot_targets(snapshot_root: Path, target_models: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    ensure_dir(snapshot_root)
    for index, model_path in enumerate(target_models):
        manifest_path = manifest_path_for_model(model_path)
        entry: dict[str, Any] = {
            "target_model_path": str(model_path),
            "target_manifest_path": str(manifest_path),
            "model_exists": model_path.exists(),
            "manifest_exists": manifest_path.exists(),
        }
        if model_path.exists():
            model_snapshot = snapshot_root / f"{index:02d}_policy_torchscript.pt"
            shutil.copy2(model_path, model_snapshot)
            entry["model_snapshot"] = str(model_snapshot)
            entry["model_sha256"] = file_sha256(model_path)
        if manifest_path.exists():
            manifest_snapshot = snapshot_root / f"{index:02d}_policy_torchscript.manifest.json"
            shutil.copy2(manifest_path, manifest_snapshot)
            entry["manifest_snapshot"] = str(manifest_snapshot)
            entry["manifest_sha256"] = file_sha256(manifest_path)
        entries.append(entry)
    return entries


def copy_candidate(source_model: Path, source_manifest: Path, candidate_root: Path) -> tuple[Path, Path]:
    ensure_dir(candidate_root)
    candidate_model = candidate_root / "policy_torchscript.pt"
    candidate_manifest = candidate_root / "policy_torchscript.manifest.json"
    shutil.copy2(source_model, candidate_model)
    shutil.copy2(source_manifest, candidate_manifest)
    return candidate_model, candidate_manifest


def deployed_manifest_payload(source_manifest: dict[str, Any], target_models: list[Path], candidate_sha256: str) -> dict[str, Any]:
    payload = dict(source_manifest)
    payload["deployed_at"] = datetime.now().astimezone().isoformat()
    payload["torchscript"] = [
        {"path": str(model_path), "sha256": candidate_sha256}
        for model_path in target_models
    ]
    return payload


def restore_snapshot_entries(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        model_path = Path(entry["target_model_path"])
        manifest_path = Path(entry["target_manifest_path"])
        model_exists = bool(entry.get("model_exists"))
        manifest_exists = bool(entry.get("manifest_exists"))
        if model_exists:
            snapshot = Path(entry["model_snapshot"])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot, model_path)
        elif model_path.exists():
            model_path.unlink()
        if manifest_exists:
            snapshot = Path(entry["manifest_snapshot"])
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot, manifest_path)
        elif manifest_path.exists():
            manifest_path.unlink()


def stage_or_promote(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_model is None or args.source_manifest is None:
        raise ValueError("stage/promote 模式必须同时提供 --source-model 与 --source-manifest")

    source_model = args.source_model.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    if not source_model.exists():
        raise FileNotFoundError(f"source model 不存在: {source_model}")
    if not source_manifest.exists():
        raise FileNotFoundError(f"source manifest 不存在: {source_manifest}")

    manifest = load_manifest(source_manifest)
    validation_errors = validate_manifest(source_model, manifest, allow_trace=args.stage_only)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    deployment_id = new_deployment_id()
    root = deployment_root_for(args.project_root.resolve(), deployment_id)
    before_root = root / "before"
    candidate_root = root / "candidate"
    after_root = root / "after"
    ensure_dir(root)

    target_models = default_target_model_paths()
    before_entries = snapshot_targets(before_root, target_models) if not args.dry_run else []
    candidate_model = candidate_root / "policy_torchscript.pt"
    candidate_manifest = candidate_root / "policy_torchscript.manifest.json"
    if not args.dry_run:
        candidate_model, candidate_manifest = copy_candidate(source_model, source_manifest, candidate_root)

    candidate_sha = file_sha256(source_model)
    payload: dict[str, Any] = {
        "deployment_id": deployment_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "label": args.label,
        "note": args.note,
        "mode": "stage_only" if args.stage_only else "promote",
        "status": "planned" if args.dry_run else ("staged" if args.stage_only else "promoted"),
        "dry_run": bool(args.dry_run),
        "source_model": str(source_model),
        "source_manifest": str(source_manifest),
        "source_model_sha256": candidate_sha,
        "source_manifest_sha256": file_sha256(source_manifest),
        "validation": {
            "obs_dim": manifest.get("obs_dim"),
            "obs_term_order": manifest.get("obs_term_order"),
            "action_semantics": manifest.get("action_semantics"),
            "export_mode": manifest.get("export_mode"),
        },
        "targets": [str(path) for path in target_models],
        "before": before_entries,
        "candidate": {
            "model_path": str(candidate_model),
            "manifest_path": str(candidate_manifest),
            "sha256": candidate_sha,
        },
    }

    after_entries: list[dict[str, Any]] = []
    if args.promote and not args.dry_run:
        deployed_manifest = deployed_manifest_payload(manifest, target_models, candidate_sha)
        deployed_manifest_text = json.dumps(deployed_manifest, ensure_ascii=False, indent=2) + "\n"
        for target_model in target_models:
            target_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_model, target_model)
            manifest_path = manifest_path_for_model(target_model)
            manifest_path.write_text(deployed_manifest_text, encoding="utf-8")
        after_entries = snapshot_targets(after_root, target_models)
    elif not args.dry_run:
        after_entries = snapshot_targets(after_root, target_models)

    payload["after"] = after_entries
    write_json(root / "deployment.json", payload)
    return payload


def rollback_deployment(args: argparse.Namespace) -> dict[str, Any]:
    source_id = str(args.rollback).strip()
    source_root = deployment_root_for(args.project_root.resolve(), source_id)
    if not source_root.exists():
        raise FileNotFoundError(f"deployment 不存在: {source_id}")
    source_payload = load_manifest(source_root / "deployment.json")
    before_entries = source_payload.get("before") or []
    if not before_entries:
        raise ValueError(f"deployment {source_id} 没有可回滚的 before 快照")

    deployment_id = new_deployment_id()
    root = deployment_root_for(args.project_root.resolve(), deployment_id)
    ensure_dir(root)
    target_models = default_target_model_paths()
    current_before = snapshot_targets(root / "before", target_models) if not args.dry_run else []
    if not args.dry_run:
        restore_snapshot_entries(before_entries)
    after_entries = snapshot_targets(root / "after", target_models) if not args.dry_run else []

    payload = {
        "deployment_id": deployment_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "rollback",
        "status": "planned" if args.dry_run else "rolled_back",
        "dry_run": bool(args.dry_run),
        "rollback_of": source_id,
        "label": args.label,
        "note": args.note,
        "targets": [str(path) for path in target_models],
        "before": current_before,
        "candidate": {"restore_from_deployment": source_id},
        "after": after_entries,
    }
    write_json(root / "deployment.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()

    if args.rollback:
        payload = rollback_deployment(args)
    else:
        payload = stage_or_promote(args)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
