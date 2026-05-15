"""Persistent conversation memory.

Every user turn and assistant reply is appended to
``~/.maksim/conversation.jsonl`` so Maksim can answer questions like
"что я говорил пять минут назад" via the ``recall_conversation`` tool
even across restarts. Audio and tool plumbing are NOT stored — only the
final spoken text.

This file is **NEVER committed** — kept entirely under MAKSIM_HOME
which lives outside the repo (and .jsonl is in .gitignore as a belt
& suspenders).
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .config import settings


def _path() -> Path:
    return settings.history_path  # ~/.maksim/conversation.jsonl


# Session ID — set once per process start. Lets us group/filter by session.
SESSION_ID = uuid.uuid4().hex[:8]


def append_turn(role: str, text: str) -> None:
    """Append one turn (user/assistant) to the conversation log."""
    if not text or not text.strip():
        return
    entry = {
        "ts": time.time(),
        "session": SESSION_ID,
        "role": role,
        "text": text.strip(),
    }
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _iter_entries() -> Iterator[dict]:
    p = _path()
    if not p.exists():
        return
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def recall(
    since_minutes_ago: float | None = None,
    last_n_turns: int | None = None,
    query: str | None = None,
    limit: int = 40,
) -> list[dict]:
    """Return a flat list of turns matching the filters, newest last.

    - ``since_minutes_ago``: keep only turns within the last N minutes.
    - ``last_n_turns``: return the most recent N turns (overrides time filter
      if both given — applied after).
    - ``query``: case-insensitive substring search across the text field.
    - ``limit``: hard cap on entries returned (default 40).
    """
    now = time.time()
    cutoff_ts = (now - since_minutes_ago * 60) if since_minutes_ago else None
    q = query.lower().strip() if query else None

    matches: list[dict] = []
    for e in _iter_entries():
        ts = e.get("ts", 0)
        if cutoff_ts is not None and ts < cutoff_ts:
            continue
        if q and q not in e.get("text", "").lower():
            continue
        matches.append(e)

    if last_n_turns is not None:
        matches = matches[-last_n_turns:]
    if len(matches) > limit:
        matches = matches[-limit:]

    return [
        {
            "time": _format_ts(e["ts"]),
            "role": e.get("role", "?"),
            "text": e.get("text", ""),
            "session": e.get("session", "?"),
        }
        for e in matches
    ]
