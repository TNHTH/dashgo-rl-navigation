from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(
    path: str | Path,
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=target.parent,
        prefix=f".{target.name}.tmp.",
    ) as handle:
        handle.write(serialized)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, target)
    return target
