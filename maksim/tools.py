"""System control tools exposed to the LLM.

Each tool is a small Python function with a JSON-Schema description. All
calls are logged to ~/.maksim/log.jsonl. Optional destructive-confirm
policy can require a verbal "да" before running rm/sudo/etc."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from . import log, memory
from .config import settings


# ─── Destructive command heuristics ──────────────────────────────────
_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+(-[rRfFi]*\s+)?(/|~|\$HOME|\*)",  # rm against home/root/glob
    r"\bsudo\b",
    r"\bdd\s+if=",
    r"\bmkfs",
    r":\(\)\s*\{",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bdiskutil\s+erase",
    r">\s*/dev/",
]

def _looks_destructive(command: str) -> bool:
    return any(re.search(p, command) for p in _DESTRUCTIVE_PATTERNS)


# ─── Implementations ─────────────────────────────────────────────────
async def _run(cmd: list[str], input_data: str | None = None,
               cwd: str | None = None, timeout: float = 30.0) -> dict:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if input_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_data.encode() if input_data else None),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        return {"ok": False, "error": f"timeout after {timeout}s"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace")[:8000],
        "stderr": stderr.decode("utf-8", errors="replace")[:2000],
    }


async def run_shell(command: str) -> dict:
    log.event("tool.run_shell", command=command)
    if settings.confirm_policy == "destructive" and _looks_destructive(command):
        return {"ok": False, "needs_confirmation": True,
                "reason": "destructive operation detected", "command": command}
    return await _run(["/bin/zsh", "-c", command], timeout=120)


async def run_applescript(script: str) -> dict:
    log.event("tool.run_applescript", script=script[:500])
    return await _run(["/usr/bin/osascript", "-e", script], timeout=30)


async def open_app(name: str) -> dict:
    log.event("tool.open_app", name=name)
    return await _run(["/usr/bin/open", "-a", name], timeout=10)


async def open_url(url: str) -> dict:
    log.event("tool.open_url", url=url)
    return await _run(["/usr/bin/open", url], timeout=10)


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


async def read_file(path: str, max_bytes: int = 65536) -> dict:
    p = _resolve(path)
    log.event("tool.read_file", path=str(p))
    try:
        data = p.read_bytes()[:max_bytes]
        try:
            text = data.decode("utf-8")
            return {"ok": True, "text": text, "size": p.stat().st_size}
        except UnicodeDecodeError:
            return {"ok": True, "binary": True, "size": p.stat().st_size}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def write_file(path: str, content: str, append: bool = False) -> dict:
    p = _resolve(path)
    log.event("tool.write_file", path=str(p), append=append, bytes=len(content))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(p)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def list_dir(path: str = "~", show_hidden: bool = False) -> dict:
    p = _resolve(path)
    log.event("tool.list_dir", path=str(p))
    if not p.exists() or not p.is_dir():
        return {"ok": False, "error": f"not a directory: {p}"}
    entries = []
    for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        if not show_hidden and child.name.startswith("."):
            continue
        entries.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "size": child.stat().st_size if child.is_file() else None,
        })
    return {"ok": True, "entries": entries[:300]}


async def search_files(query: str, root: str = "~", max_results: int = 50) -> dict:
    """mdfind (Spotlight) lookup — fast and indexes everything."""
    log.event("tool.search_files", query=query, root=root)
    root_p = str(_resolve(root))
    res = await _run(["/usr/bin/mdfind", "-onlyin", root_p, query], timeout=15)
    if res["ok"]:
        lines = [l for l in res["stdout"].splitlines() if l.strip()][:max_results]
        return {"ok": True, "results": lines}
    return res


async def get_clipboard() -> dict:
    log.event("tool.get_clipboard")
    res = await _run(["/usr/bin/pbpaste"], timeout=5)
    return {"ok": res["ok"], "text": res.get("stdout", "")}


async def set_clipboard(text: str) -> dict:
    log.event("tool.set_clipboard", bytes=len(text))
    return await _run(["/usr/bin/pbcopy"], input_data=text, timeout=5)


async def notify(title: str, message: str = "") -> dict:
    """Send a macOS notification."""
    log.event("tool.notify", title=title, message=message)
    safe_title = title.replace('"', '\\"')
    safe_msg = message.replace('"', '\\"')
    script = f'display notification "{safe_msg}" with title "{safe_title}"'
    return await _run(["/usr/bin/osascript", "-e", script], timeout=5)


async def look_at_screen(scale: float = 0.5) -> dict:
    """Capture the current display and return it as base64-encoded JPEG so the
    LLM can see what the user is looking at. Returns a special dict with an
    `_image` field which the LLM module turns into a vision content block."""
    log.event("tool.look_at_screen", scale=scale)
    scale = max(0.2, min(1.0, float(scale)))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
    try:
        # -x: silent (no shutter sound), -t jpg: jpeg, -C: include cursor
        res = await _run(["/usr/sbin/screencapture", "-x", "-t", "jpg", "-C", path], timeout=10)
        if not res["ok"]:
            return {"ok": False, "error": "screencapture failed",
                    "stderr": res.get("stderr", "")}
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {"ok": False, "error": "screencapture produced no file — "
                    "Screen Recording permission likely missing"}
        # Downscale via sips to keep payload under Claude's 5MB image limit.
        if scale < 1.0:
            await _run(["/usr/bin/sips", "-Z", str(int(2880 * scale)), path], timeout=10)
        data = Path(path).read_bytes()
        if len(data) > 4_500_000:
            # Aggressive downscale fallback
            await _run(["/usr/bin/sips", "-Z", "1280", path], timeout=10)
            data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "ok": True,
            "_image": b64,
            "_media_type": "image/jpeg",
            "size_bytes": len(data),
        }
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


async def get_active_app() -> dict:
    log.event("tool.get_active_app")
    res = await _run([
        "/usr/bin/osascript", "-e",
        'tell application "System Events" to get name of first application process whose frontmost is true',
    ], timeout=5)
    return {"ok": res["ok"], "name": res.get("stdout", "").strip()}


# ─── Mouse / keyboard automation ─────────────────────────────────────
# All implemented via AppleScript / System Events. Requires Accessibility
# permission to be granted to Maksim.app once (System Settings → Privacy
# & Security → Accessibility).
async def click_at(x: int, y: int, button: str = "left", double: bool = False) -> dict:
    """Move mouse to (x, y) and click. Coordinates in absolute screen pixels
    from top-left. Look at the screen first with `look_at_screen` to find
    the right coordinates."""
    log.event("tool.click_at", x=x, y=y, button=button, double=double)
    action = "double click" if double else ("right click" if button == "right" else "click")
    script = f'tell application "System Events" to {action} at {{{int(x)}, {int(y)}}}'
    return await _run(["/usr/bin/osascript", "-e", script], timeout=5)


async def type_text(text: str) -> dict:
    """Type text into the currently focused field (as if from keyboard)."""
    log.event("tool.type_text", bytes=len(text))
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "System Events" to keystroke "{safe}"'
    return await _run(["/usr/bin/osascript", "-e", script], timeout=10)


_KEY_CODES = {
    "return": 36, "enter": 36, "tab": 48, "space": 49, "escape": 53, "esc": 53,
    "delete": 51, "backspace": 51, "forward_delete": 117,
    "up": 126, "down": 125, "left": 123, "right": 124,
    "home": 115, "end": 119, "page_up": 116, "page_down": 121,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
    "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}


async def press_key(key: str, modifiers: list[str] | None = None) -> dict:
    """Press a special key, optionally with modifiers.
    key: 'return', 'tab', 'escape', 'space', 'delete', 'up/down/left/right', 'f1'..'f12'
    modifiers: any of 'cmd', 'shift', 'option', 'control'."""
    log.event("tool.press_key", key=key, modifiers=modifiers)
    code = _KEY_CODES.get(key.lower())
    if code is None:
        return {"ok": False, "error": f"unknown key {key!r}; supported: {sorted(_KEY_CODES)}"}
    mods = modifiers or []
    mod_map = {"cmd": "command down", "command": "command down",
               "shift": "shift down", "option": "option down", "alt": "option down",
               "control": "control down", "ctrl": "control down"}
    using = ", ".join(mod_map[m.lower()] for m in mods if m.lower() in mod_map)
    if using:
        script = f'tell application "System Events" to key code {code} using {{{using}}}'
    else:
        script = f'tell application "System Events" to key code {code}'
    return await _run(["/usr/bin/osascript", "-e", script], timeout=5)


async def recall_conversation(
    since_minutes_ago: float | None = None,
    last_n_turns: int | None = None,
    query: str | None = None,
    limit: int = 40,
) -> dict:
    """Look up past user/assistant turns from the persistent conversation log.
    Use when the user asks 'что я говорил...', 'о чём мы говорили...',
    'напомни что обсуждали' — even when referring to a previous session."""
    log.event("tool.recall_conversation",
              since_minutes_ago=since_minutes_ago,
              last_n_turns=last_n_turns,
              query=query, limit=limit)
    entries = memory.recall(
        since_minutes_ago=since_minutes_ago,
        last_n_turns=last_n_turns,
        query=query,
        limit=limit,
    )
    return {"ok": True, "count": len(entries), "turns": entries}


async def scroll(dx: int = 0, dy: int = 0) -> dict:
    """Scroll by (dx, dy) units. Positive dy scrolls down."""
    log.event("tool.scroll", dx=dx, dy=dy)
    # System Events doesn't have native scroll; we approximate via AppleScript
    # arrow-key spam for vertical. For horizontal we rely on shift+wheel sim.
    if dx == 0 and dy == 0:
        return {"ok": True}
    if dy != 0:
        key = "down" if dy > 0 else "up"
        steps = min(abs(dy), 30)
        for _ in range(steps):
            await _run(["/usr/bin/osascript", "-e",
                        f"tell application \"System Events\" to key code {_KEY_CODES[key]}"], timeout=2)
    return {"ok": True}


# ─── Tool registry for Anthropic ─────────────────────────────────────
TOOLS_SCHEMA = [
    {
        "name": "run_shell",
        "description": (
            "Execute an arbitrary zsh command on the user's Mac. Use this for "
            "any system operation that doesn't have a more specific tool. "
            "Has full access to the filesystem, network, and installed CLIs "
            "(brew, git, python, node, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "run_applescript",
        "description": (
            "Run AppleScript via osascript. Use for controlling macOS apps, "
            "windows, dialogs, system preferences, music/video, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"script": {"type": "string"}},
            "required": ["script"],
        },
    },
    {
        "name": "open_app",
        "description": "Launch a Mac application by name (e.g. 'Safari', 'Telegram', 'Visual Studio Code').",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "open_url",
        "description": "Open a URL in the default browser.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file. Use absolute paths or ~.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "default": 65536},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or append UTF-8 text to a file. Creates parent dirs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List directory entries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "~"},
                "show_hidden": {"type": "boolean", "default": False},
            },
            "required": [],
        },
    },
    {
        "name": "search_files",
        "description": "Fast Spotlight (mdfind) search for files by name or content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "root": {"type": "string", "default": "~"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_clipboard",
        "description": "Read the current macOS clipboard text.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_clipboard",
        "description": "Replace the macOS clipboard with the given text.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "notify",
        "description": "Show a native macOS notification.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "message": {"type": "string", "default": ""},
            },
            "required": ["title"],
        },
    },
    {
        "name": "get_active_app",
        "description": "Get the name of the frontmost (focused) Mac application.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "look_at_screen",
        "description": (
            "Capture a screenshot of the user's display so you can SEE what's on it. "
            "Use this when the user asks 'посмотри на экран', 'что у меня открыто', "
            "'что я делаю', 'помоги с тем что вижу', or any request that needs visual "
            "context. The screenshot is returned to you as an image — describe what "
            "you see and answer the user's question based on it. To click on something "
            "you see, call `click_at` with pixel coordinates derived from the screenshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Downscale factor (0.2–1.0). 0.5 keeps detail while halving size.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "click_at",
        "description": (
            "Click at absolute screen coordinates (pixels from top-left of the main display). "
            "Use `look_at_screen` first to find the button/link, then estimate (x, y) and call this. "
            "Set `double=true` for double-click, `button='right'` for right-click."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right"], "default": "left"},
                "double": {"type": "boolean", "default": False},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "type_text",
        "description": "Type literal text into the currently focused input field, as if the user typed it.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": (
            "Press a special key, optionally with modifiers. "
            "Keys: return, tab, escape, space, delete, up/down/left/right, home, end, "
            "page_up, page_down, f1..f12. "
            "Modifiers: cmd, shift, option, control. Examples: press_key('return'), "
            "press_key('tab', ['cmd']) — Cmd+Tab."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "modifiers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key"],
        },
    },
    {
        "name": "recall_conversation",
        "description": (
            "Look up past user/assistant turns from Maksim's persistent conversation "
            "log. Use this when the user asks things like 'что я говорил пять минут "
            "назад', 'о чём мы говорили час назад', 'напомни что обсуждали вчера', "
            "or 'найди в истории про X'. Works ACROSS sessions — past restarts. "
            "Filter by time (since_minutes_ago), recent turn count (last_n_turns), "
            "or a keyword (query). Returns a list of {time, role, text, session}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "since_minutes_ago": {
                    "type": "number",
                    "description": "Return turns from the last N minutes (e.g. 5 for 'пять минут назад').",
                },
                "last_n_turns": {
                    "type": "integer",
                    "description": "Return the most recent N turns regardless of time.",
                },
                "query": {
                    "type": "string",
                    "description": "Case-insensitive substring to search for in turn text.",
                },
                "limit": {
                    "type": "integer",
                    "default": 40,
                    "description": "Hard cap on returned entries (default 40).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the focused window. Positive dy scrolls down by dy lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dx": {"type": "integer", "default": 0},
                "dy": {"type": "integer", "default": 0},
            },
            "required": [],
        },
    },
]


TOOL_FUNCTIONS = {
    "run_shell": run_shell,
    "run_applescript": run_applescript,
    "open_app": open_app,
    "open_url": open_url,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "search_files": search_files,
    "get_clipboard": get_clipboard,
    "set_clipboard": set_clipboard,
    "notify": notify,
    "get_active_app": get_active_app,
    "look_at_screen": look_at_screen,
    "click_at": click_at,
    "type_text": type_text,
    "press_key": press_key,
    "scroll": scroll,
    "recall_conversation": recall_conversation,
}


async def dispatch(name: str, arguments: dict) -> Any:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    t0 = time.monotonic()
    try:
        result = await fn(**arguments)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    # Don't dump base64 image payloads into the log file.
    log_ok = result.get("ok", True) if isinstance(result, dict) else True
    log.event("tool.result", name=name, ms=int((time.monotonic() - t0) * 1000), ok=log_ok)
    return result
