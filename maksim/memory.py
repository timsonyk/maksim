"""Two layers of persistent memory.

1. **Conversation log** (``~/.maksim/conversation.jsonl``) — every spoken
   user turn and assistant reply, append-only. Used by the
   ``recall_conversation`` tool for "что я говорил пять минут назад".

2. **Long-term facts** (``~/.maksim/long_term.json``) — structured
   key→value pairs Maksim explicitly remembers (e.g. "user_name" =
   "Тимофей", "main_project" = "Maksim"). Loaded into the system
   prompt on every Brain init, written via ``remember`` / ``forget``
   tools.

Short-term memory (last ~12 turns of the current session) lives in
``Brain.history`` in RAM — no module needed.

All these files live under MAKSIM_HOME, which is OUTSIDE the repo. They
are also explicitly listed in ``.gitignore`` (belt & suspenders).
"""
from __future__ import annotations

import json
import os
import tempfile
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


# ───────────────────────────────────────────────────────────────────
# Long-term memory — structured facts that survive across sessions.
# Loaded into the system prompt at Brain init so Maksim always knows
# them without needing a tool call.
# ───────────────────────────────────────────────────────────────────

def _facts_path() -> Path:
    return settings.home / "long_term.json"


def _load_facts() -> dict:
    p = _facts_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_facts(facts: dict) -> None:
    p = _facts_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write — tempfile in same dir + rename.
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".long_term.", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def set_fact(key: str, value: str, category: str = "general") -> None:
    facts = _load_facts()
    facts[key] = {
        "value": value,
        "category": category,
        "ts": time.time(),
    }
    _save_facts(facts)


def forget_fact(key: str) -> bool:
    facts = _load_facts()
    if key in facts:
        del facts[key]
        _save_facts(facts)
        return True
    return False


def all_facts() -> dict:
    return _load_facts()


def facts_for_system_prompt() -> str:
    """Return a compact textual summary of all long-term facts, suitable
    for splicing into the system prompt. Empty string if no facts."""
    facts = _load_facts()
    if not facts:
        return ""
    by_cat: dict[str, list[str]] = {}
    for k, entry in facts.items():
        cat = entry.get("category", "general")
        by_cat.setdefault(cat, []).append(f"  - {k}: {entry.get('value')}")
    lines = ["", "Известные факты о пользователе (из долгосрочной памяти):"]
    for cat in sorted(by_cat):
        lines.append(f"[{cat}]")
        lines.extend(by_cat[cat])
    return "\n".join(lines)
