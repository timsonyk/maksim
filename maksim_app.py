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
import queue
import threading
import traceback

import rumps

from maksim.__main__ import _missing_keys
from maksim.orchestrator import Maksim


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
        self.menu = [self._status_item, None, rumps.MenuItem("Выход", callback=self._quit)]

        # Status updates arrive from the Maksim asyncio thread; AppKit objects
        # (NSMenuItem, status-bar title) MUST only be mutated from the main
        # thread or macOS kills the process with EXC_BREAKPOINT in
        # NSWMWindowCoordinator. Marshal via a thread-safe queue drained on
        # the main thread by rumps.Timer.
        self._status_q: queue.Queue[str] = queue.Queue()
        self._drain_timer = rumps.Timer(self._drain_status, 0.15)
        self._drain_timer.start()

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
        self.title = _ICONS.get(latest, "🎙")
        self._status_item.title = _LABELS.get(latest, latest)

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
