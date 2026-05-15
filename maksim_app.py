"""PyInstaller entry point + rumps menu-bar wrapper.

Runs Maksim in a background thread and shows a status emoji in the macOS
menu bar (top-right). Status meanings:

  🎙  loading  — modules initializing
  👂  idle     — listening for wake word
  ⚡  wake     — wake fired, capturing utterance
  💭  thinking — LLM is generating a response
  🗣️   talking  — TTS is speaking
  💬  session  — continuous session, waiting for follow-up
  ⚠️   error    — exception in the background loop

Click the icon for a menu with the current status and a Quit item.
"""
from __future__ import annotations

import multiprocessing

# CRITICAL: must be first import inside a frozen PyInstaller bundle, otherwise
# multiprocessing.resource_tracker / spawn forks recursively re-execute the
# whole .app — fork-bomb that runs N copies of Maksim simultaneously.
multiprocessing.freeze_support()

import asyncio
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import rumps

from maksim.__main__ import _missing_keys
from maksim.orchestrator import Maksim
from maksim.config import settings


_ICONS = {
    "loading":  "🎙",
    "idle":     "👂",
    "wake":     "⚡",
    "thinking": "💭",
    "talking":  "🗣️",
    "session":  "💬",
    "error":    "⚠️",
}

_LABELS = {
    "loading":  "Загрузка модулей…",
    "idle":     "Слушаю wake-word",
    "wake":     "Wake! Слушаю команду…",
    "thinking": "Думаю…",
    "talking":  "Отвечаю…",
    "session":  "Продолжаю диалог (скажи «конец связи»)",
    "error":    "Ошибка — см. ~/.maksim/log.jsonl",
}


class MaksimStatusBarApp(rumps.App):
    def __init__(self) -> None:
        super().__init__(_ICONS["loading"], quit_button=None)
        self._status_item = rumps.MenuItem(_LABELS["loading"])
        self.menu = [
            self._status_item,
            None,  # separator
            rumps.MenuItem("Информация…", callback=self._show_info),
            rumps.MenuItem("Открыть папку логов", callback=self._open_logs),
            rumps.MenuItem("Редактировать конфиг", callback=self._edit_config),
            None,  # separator
            rumps.MenuItem("Перезапуск", callback=self._restart),
            rumps.MenuItem("Выход", callback=self._quit),
        ]

        # Status updates arrive from the Maksim asyncio thread; AppKit objects
        # (NSMenuItem, status-bar title) MUST only be mutated from the main
        # thread or macOS kills the process with EXC_BREAKPOINT in
        # NSWMWindowCoordinator. Marshal via a thread-safe queue drained on
        # the main thread by rumps.Timer.
        self._status_q: queue.Queue[str] = queue.Queue()
        self._drain_timer = rumps.Timer(self._drain_status, 0.15)
        self._drain_timer.start()

        self._current_status = "loading"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._maksim_thread = threading.Thread(target=self._run_maksim, daemon=True)
        self._maksim_thread.start()

    def _set_status(self, status: str) -> None:
        # Called from the Maksim thread — never touch AppKit here.
        try:
            self._status_q.put_nowait(status)
        except queue.Full:
            pass  # drop, next status will arrive shortly

    def _drain_status(self, _sender) -> None:
        # Runs on the main thread courtesy of rumps.Timer.
        latest = None
        try:
            while True:
                latest = self._status_q.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            return
        self._current_status = latest
        self.title = _ICONS.get(latest, "🎙")
        self._status_item.title = _LABELS.get(latest, latest)

    # ── Menu actions ────────────────────────────────────────────────
    def _show_info(self, _sender) -> None:
        """Show a status / config summary in an alert dialog."""
        try:
            from maksim import memory
            facts = memory.all_facts()
            conv_path = settings.history_path
            conv_lines = 0
            if conv_path.exists():
                with conv_path.open() as f:
                    conv_lines = sum(1 for _ in f)
            info = (
                f"Статус: {_LABELS.get(self._current_status, self._current_status)}\n"
                f"\n"
                f"Wake-word: {settings.wake_word} (threshold {settings.wake_threshold})\n"
                f"STT: {settings.stt_provider}  •  local model: {settings.local_stt_model}\n"
                f"TTS: {settings.tts_provider}"
                + (f" ({settings.silero_speaker})" if settings.tts_provider == "silero" else "")
                + "\n"
                f"LLM: {settings.llm_model}\n"
                f"\n"
                f"Долгосрочных фактов: {len(facts)}\n"
                f"Реплик в логе: {conv_lines}\n"
                f"\n"
                f"Конфиг:  ~/.maksim/.env\n"
                f"Логи:    ~/.maksim/log.jsonl\n"
            )
        except Exception as e:
            info = f"Ошибка при получении инфо: {e}"
        rumps.alert(title="Maksim", message=info, ok="OK")

    def _open_logs(self, _sender) -> None:
        subprocess.Popen(["/usr/bin/open", str(settings.home)])

    def _edit_config(self, _sender) -> None:
        env_path = settings.home / ".env"
        if not env_path.exists():
            env_path.write_text("# Maksim config — see project .env.example for keys\n")
        subprocess.Popen(["/usr/bin/open", "-e", str(env_path)])

    def _restart(self, _sender) -> None:
        # Relaunch ourselves via 'open' so macOS spawns a fresh process from
        # the .app bundle. Then exit current process.
        bundle_root = Path(sys.argv[0]).resolve()
        # When frozen, sys.argv[0] is <Maksim.app>/Contents/MacOS/maksim → walk up.
        app_root = bundle_root
        for _ in range(3):
            if app_root.suffix == ".app":
                break
            app_root = app_root.parent
        if app_root.suffix == ".app":
            subprocess.Popen(["/usr/bin/open", "-n", str(app_root)])
        else:
            subprocess.Popen([sys.executable, *sys.argv])
        self._quit(None)

    def _run_maksim(self) -> None:
        missing = _missing_keys()
        if missing:
            self._set_status("error")
            return  # error label set via the queue from _set_status
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            j = Maksim(status_cb=self._set_status)
            self._loop.run_until_complete(j.run())
        except Exception as e:
            traceback.print_exc()
            self._set_status("error")
            # NOTE: do NOT touch self._status_item.title here — wrong thread.

    def _quit(self, _sender) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        rumps.quit_application()


def main() -> None:
    MaksimStatusBarApp().run()


if __name__ == "__main__":
    main()
