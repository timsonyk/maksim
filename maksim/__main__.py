"""Entry point: `python -m maksim` or `uv run maksim`."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from . import log
from .orchestrator import Maksim, console


def _missing_keys() -> list[str]:
    from .config import settings
    missing = []
    if settings.tts_provider == "elevenlabs" and not settings.elevenlabs_api_key:
        missing.append("ELEVENLABS_API_KEY")
    if not settings.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if settings.stt_provider == "groq" and not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    return missing


def main() -> None:
    missing = _missing_keys()
    if missing:
        # Graceful daemon idle. KeepAlive restarts on non-zero exit; with a
        # 60-second sleep + ThrottleInterval=10s, we re-check ~every 70 s.
        msg = (f"⏸  Не хватает ключей: {', '.join(missing)}. "
               "Положи их в .env.local. Перезапуск через 60 с.")
        console.print(f"[yellow]{msg}[/yellow]")
        log.event("startup.missing_keys", keys=missing)
        time.sleep(60)
        sys.exit(1)

    j = Maksim()
    try:
        asyncio.run(j.run())
    except KeyboardInterrupt:
        console.print("\n[cyan]Bye.[/cyan]")


if __name__ == "__main__":
    main()
