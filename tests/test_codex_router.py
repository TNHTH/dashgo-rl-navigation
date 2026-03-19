from __future__ import annotations

import json
from pathlib import Path

from autopilot.codex_router import resolve_codex_route
from autopilot.codex_escalator import enqueue_codex_job


def write_models_cache(path: Path, models: list[dict]) -> Path:
    path.write_text(json.dumps({"models": models}, ensure_ascii=False), encoding="utf-8")
    return path


def test_resolve_diagnose_route_prefers_gpt53(tmp_path) -> None:
    cache_path = write_models_cache(
        tmp_path / "models_cache.json",
        [
            {
                "slug": "gpt-5.3-codex",
                "supported_reasoning_levels": [{"effort": "high"}],
            },
            {
                "slug": "gpt-5.4-mini",
                "supported_reasoning_levels": [{"effort": "high"}],
            },
        ],
    )

    route = resolve_codex_route("debug_job", cache_path=cache_path)

    assert route.requested_model == "gpt-5.3-codex"
    assert route.effective_model == "gpt-5.3-codex"
    assert route.effective_reasoning_effort == "high"
    assert route.resolution_mode == "exact"


def test_resolve_diagnose_route_falls_back_to_gpt54mini(tmp_path) -> None:
    cache_path = write_models_cache(
        tmp_path / "models_cache.json",
        [
            {
                "slug": "gpt-5.4-mini",
                "supported_reasoning_levels": [{"effort": "high"}],
            }
        ],
    )

    route = resolve_codex_route("debug_job", cache_path=cache_path)

    assert route.requested_model == "gpt-5.3-codex"
    assert route.effective_model == "gpt-5.4-mini"
    assert route.effective_reasoning_effort == "high"
    assert route.resolution_mode == "fallback"


def test_enqueue_codex_job_launch_disabled_exposes_route(monkeypatch, tmp_path) -> None:
    cache_path = write_models_cache(
        tmp_path / "models_cache.json",
        [
            {
                "slug": "gpt-5.2",
                "supported_reasoning_levels": [{"effort": "medium"}],
            }
        ],
    )

    monkeypatch.setattr("autopilot.codex_router.MODEL_CACHE_PATH", cache_path)

    payload = enqueue_codex_job(
        project_root=tmp_path,
        job_type="monitor_job",
        prompt="summary",
        allowed_paths=[],
        inputs={},
        launch=False,
    )

    assert payload["status"] == "queued_only"
    assert payload["effective_model"] == "gpt-5.2"
    assert payload["effective_reasoning_effort"] == "medium"
