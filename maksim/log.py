from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .config import settings


_lock = threading.Lock()


def event(kind: str, **fields: Any) -> None:
    """Append a JSONL event to the action log."""
    record = {"ts": time.time(), "kind": kind, **fields}
    with _lock:
        with settings.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def latest(n: int = 50) -> list[dict]:
    p: Path = settings.log_path
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
