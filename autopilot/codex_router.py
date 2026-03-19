from __future__ import annotations

import json
from pathlib import Path

from .types import CodexRouteDecision


MODEL_CACHE_PATH = Path.home() / ".codex" / "models_cache.json"

TIER_CONFIG = {
    "monitor": {
        "profile": "monitor",
        "requested_model": "gpt-5.2",
        "requested_reasoning_effort": "medium",
        "fallbacks": (),
    },
    "diagnose": {
        "profile": "diagnose",
        "requested_model": "gpt-5.3-codex",
        "requested_reasoning_effort": "high",
        "fallbacks": (("gpt-5.4-mini", "high", "gpt-5.3-codex_unavailable"),),
    },
    "authoring": {
        "profile": "authoring",
        "requested_model": "gpt-5.4",
        "requested_reasoning_effort": "xhigh",
        "fallbacks": (),
    },
}

JOB_TYPE_TO_TIER = {
    "debug_job": "diagnose",
    "review_job": "diagnose",
    "retro_job": "diagnose",
    "monitor_job": "monitor",
    "patch_job": "authoring",
    "research_job": "authoring",
    "planning_job": "authoring",
}


def route_tier_for_job(job_type: str) -> str:
    return JOB_TYPE_TO_TIER.get(job_type, "authoring")


def load_model_catalog(cache_path: Path | None = None) -> tuple[dict[str, set[str]], str]:
    target = cache_path or MODEL_CACHE_PATH
    if not target.exists():
        return {}, str(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    catalog: dict[str, set[str]] = {}
    for item in payload.get("models", []):
        slug = item.get("slug")
        if not slug:
            continue
        efforts = {
            level.get("effort")
            for level in item.get("supported_reasoning_levels", [])
            if isinstance(level, dict) and level.get("effort")
        }
        catalog[str(slug)] = efforts
    return catalog, str(target)


def model_supports(
    catalog: dict[str, set[str]],
    *,
    model: str,
    reasoning_effort: str,
) -> bool:
    supported = catalog.get(model)
    if supported is None:
        return False
    return reasoning_effort in supported


def resolve_codex_route(job_type: str, cache_path: Path | None = None) -> CodexRouteDecision:
    tier = route_tier_for_job(job_type)
    config = TIER_CONFIG[tier]
    requested_model = str(config["requested_model"])
    requested_effort = str(config["requested_reasoning_effort"])
    catalog, catalog_source = load_model_catalog(cache_path)

    if not catalog or model_supports(catalog, model=requested_model, reasoning_effort=requested_effort):
        return CodexRouteDecision(
            job_type=job_type,
            route_tier=tier,
            requested_profile=str(config["profile"]),
            requested_model=requested_model,
            effective_model=requested_model,
            requested_reasoning_effort=requested_effort,
            effective_reasoning_effort=requested_effort,
            resolution_mode="exact",
            fallback_reason="",
            catalog_source=catalog_source,
        )

    for fallback_model, fallback_effort, fallback_reason in config.get("fallbacks", ()):
        if model_supports(catalog, model=fallback_model, reasoning_effort=fallback_effort):
            return CodexRouteDecision(
                job_type=job_type,
                route_tier=tier,
                requested_profile=str(config["profile"]),
                requested_model=requested_model,
                effective_model=str(fallback_model),
                requested_reasoning_effort=requested_effort,
                effective_reasoning_effort=str(fallback_effort),
                resolution_mode="fallback",
                fallback_reason=str(fallback_reason),
                catalog_source=catalog_source,
            )

    raise RuntimeError(
        f"未找到可用的 Codex 路由模型: tier={tier} requested={requested_model}@{requested_effort}"
    )
