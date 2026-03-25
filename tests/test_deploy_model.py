from __future__ import annotations

import json
from pathlib import Path

from tools.diagnostics import deploy_model


def write_model(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return deploy_model.file_sha256(path)


def write_manifest(path: Path, sha256: str, *, export_mode: str = "script") -> None:
    payload = {
        "obs_dim": 246,
        "obs_term_order": deploy_model.EXPECTED_OBS_TERM_ORDER,
        "action_semantics": "bounded_tanh_gaussian",
        "export_mode": export_mode,
        "torchscript": [{"path": str(path.with_suffix('.pt')), "sha256": sha256}],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_validate_manifest_rejects_trace_for_promote(tmp_path: Path) -> None:
    model_path = tmp_path / "policy_torchscript.pt"
    manifest_path = tmp_path / "policy_torchscript.manifest.json"
    sha = write_model(model_path, b"trace-model")
    write_manifest(manifest_path, sha, export_mode="trace")
    manifest = deploy_model.load_manifest(manifest_path)

    errors = deploy_model.validate_manifest(model_path, manifest, allow_trace=False)

    assert any("export_mode" in item for item in errors)
    assert deploy_model.validate_manifest(model_path, manifest, allow_trace=True) == []


def test_stage_only_dry_run_writes_deployment_metadata(monkeypatch, tmp_path: Path) -> None:
    target_model = tmp_path / "targets" / "ros2" / "policy_torchscript.pt"
    target_manifest = deploy_model.manifest_path_for_model(target_model)
    old_sha = write_model(target_model, b"old-model")
    write_manifest(target_manifest, old_sha)

    source_model = tmp_path / "candidate" / "policy_torchscript.pt"
    source_manifest = tmp_path / "candidate" / "policy_torchscript.manifest.json"
    source_sha = write_model(source_model, b"new-model")
    write_manifest(source_manifest, source_sha)

    monkeypatch.setattr(deploy_model, "default_target_model_paths", lambda: [target_model])

    args = deploy_model.build_parser().parse_args(
        [
            "--stage-only",
            "--source-model",
            str(source_model),
            "--source-manifest",
            str(source_manifest),
            "--project-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    payload = deploy_model.stage_or_promote(args)

    deployment_root = deploy_model.deployment_root_for(tmp_path, payload["deployment_id"])
    deployment_json = deployment_root / "deployment.json"
    deploy_model.write_json(deployment_json, payload)

    assert payload["status"] == "planned"
    assert json.loads(deployment_json.read_text(encoding="utf-8"))["dry_run"] is True
    assert target_model.read_bytes() == b"old-model"


def test_promote_then_rollback_restores_previous_model(monkeypatch, tmp_path: Path) -> None:
    target_model = tmp_path / "targets" / "ros2" / "policy_torchscript.pt"
    target_manifest = deploy_model.manifest_path_for_model(target_model)
    old_sha = write_model(target_model, b"old-model")
    write_manifest(target_manifest, old_sha)

    source_model = tmp_path / "candidate" / "policy_torchscript.pt"
    source_manifest = tmp_path / "candidate" / "policy_torchscript.manifest.json"
    new_sha = write_model(source_model, b"new-model")
    write_manifest(source_manifest, new_sha)

    monkeypatch.setattr(deploy_model, "default_target_model_paths", lambda: [target_model])

    promote_args = deploy_model.build_parser().parse_args(
        [
            "--promote",
            "--source-model",
            str(source_model),
            "--source-manifest",
            str(source_manifest),
            "--project-root",
            str(tmp_path),
        ]
    )
    promoted = deploy_model.stage_or_promote(promote_args)
    deployment_root = deploy_model.deployment_root_for(tmp_path, promoted["deployment_id"])
    deploy_model.write_json(deployment_root / "deployment.json", promoted)

    assert target_model.read_bytes() == b"new-model"

    rollback_args = deploy_model.build_parser().parse_args(
        [
            "--rollback",
            promoted["deployment_id"],
            "--project-root",
            str(tmp_path),
        ]
    )
    rolled_back = deploy_model.rollback_deployment(rollback_args)

    assert rolled_back["status"] == "rolled_back"
    assert target_model.read_bytes() == b"old-model"
