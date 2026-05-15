"""Main Maksim event loop: wake → VAD → STT → LLM → TTS."""
from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator, Callable, Optional

from rich.console import Console

from . import log
from .audio import make_audio
from .config import settings
from .llm import make_brain
from .stt import make_stt
from .tts import make_tts
from .vad import make_vad
from .wake import make_wake


console = Console()

# Wake word for the STT-based fallback. Strict — only matches "максим" as a
# standalone word (not "максимум", "максимально", "максимка"). Lowercased.
_STT_WAKE_WORD = "максим"
_WORD_RE = re.compile(r"[\wа-яёА-ЯЁ]+", re.UNICODE)


def _has_stt_wake(text: str) -> bool:
    """True if 'максим' appears as a standalone word within the first 3 tokens."""
    if not text:
        return False
    words = _WORD_RE.findall(text.lower())
    return _STT_WAKE_WORD in words[:3]


def _strip_wake_prefix(text: str) -> str:
    """Return everything after the first 'максим' token, or '' if it was alone."""
    return re.sub(rf"^.*?\b{_STT_WAKE_WORD}\b[\s,.!?]*", "", text, count=1,
                  flags=re.IGNORECASE).strip()


# Phrases that end a continuous session. Matched on a normalized
# (lowercased, no trailing punctuation) transcript.
_END_PHRASES = (
    "конец связи",
    "конец сессии",
    "конец разговора",
    "отбой",
)
# Max time to wait for a follow-up utterance before auto-closing the session.
_FOLLOWUP_TIMEOUT_S = 30.0

# Whisper hallucinations on near-silent input — common for `small`/`base` models.
# These appear as fully-formed Russian phrases from background noise alone.
# Used in the continuous-session loop to drop spurious wake-ups.
_HALLUCINATIONS = frozenset({
    "продолжение следует",
    "продолжение следует...",
    "спасибо за просмотр",
    "спасибо за внимание",
    "конец фильма",
    "и.и.и.",
    "и, и, и",
    "тереть",
    "пу-пум...",
})

# Regex patterns that match a whole family of subtitle/credits hallucinations
# Whisper generates on silence (random editor/translator names).
_HALLUCINATION_PATTERNS = [
    re.compile(r"\bсубтитр", re.IGNORECASE),
    re.compile(r"\bкорректор", re.IGNORECASE),
    re.compile(r"\bредактор\b", re.IGNORECASE),
    re.compile(r"\bперевёл", re.IGNORECASE),
    re.compile(r"dimatorzok", re.IGNORECASE),
    re.compile(r"довжик", re.IGNORECASE),
]


def _is_end_phrase(text: str) -> bool:
    if not text:
        return False
    lc = re.sub(r"[.,!?\s]+$", "", text.lower().strip())
    return any(p in lc for p in _END_PHRASES)


def _looks_like_hallucination(text: str) -> bool:
    """Heuristic: drop very short transcripts and known whisper-junk phrases."""
    if not text:
        return True
    normalized = re.sub(r"[.,!?]+$", "", text.lower().strip())
    if normalized in _HALLUCINATIONS:
        return True
    for pat in _HALLUCINATION_PATTERNS:
        if pat.search(normalized):
            return True
    # In continuous mode, real commands almost always have ≥2 words.
    if len(_WORD_RE.findall(normalized)) < 2:
        return True
    return False


class Maksim:
    def __init__(self, status_cb: Optional[Callable[[str], None]] = None):
        # status_cb is called with one of: "loading", "idle", "wake", "thinking",
        # "talking", "session", "error". Used by the rumps menu-bar wrapper to
        # update the status-bar icon. Safe to leave as None for CLI usage.
        self._status_cb = status_cb
        self._set_status("loading")
        console.print("[cyan]Загружаю модули…[/cyan]")
        self.audio = make_audio()
        self.wake = make_wake()
        # Second wake detector with a lower threshold, used ONLY during TTS
        # playback for barge-in. Trade-off: more sensitive, more false positives
        # — but only matters while Maksim is actively speaking.
        from .wake import WakeWordDetector
        self.bargein_wake = WakeWordDetector(
            settings.wake_word,
            threshold=max(0.4, settings.wake_threshold - 0.3),
        )
        self.vad = make_vad()
        self.stt = make_stt()
        self.brain = make_brain()
        self.tts = make_tts()

    def _set_status(self, status: str) -> None:
        if self._status_cb is None:
            return
        try:
            self._status_cb(status)
        except Exception as e:
            log.event("status_cb.error", error=str(e))

    async def _drain_playback(self) -> None:
        # Wait until queued audio finished playing.
        while self.audio.is_playing():
            await asyncio.sleep(0.05)

    async def _stt_wake_loop(self) -> str:
        """Continuously VAD-segment audio, transcribe, return when wake word
        appears in transcript. Returns the FULL transcript so the caller can
        extract any command that followed the wake word."""
        while True:
            pcm = await self.vad.capture_utterance(self.audio)
            if len(pcm) < 4000:
                continue
            try:
                text = await self.stt.transcribe(pcm, 16000)
            except Exception as e:
                log.event("stt_wake.fail", error=str(e))
                await asyncio.sleep(0.1)
                continue
            text = (text or "").strip()
            log.event("stt_wake.heard", text=text)
            if _has_stt_wake(text):
                return text

    async def _one_turn(self) -> None:
        self._set_status("idle")
        label = self.wake.keyword.rsplit("/", 1)[-1].removesuffix(".onnx")
        console.print(f"[dim]Жду '{label}' (onnx) или произнесённое 'максим' (STT)…[/dim]")

        onnx_task = asyncio.create_task(self.wake.wait_for_trigger(self.audio))
        stt_task = asyncio.create_task(self._stt_wake_loop())

        done, pending = await asyncio.wait(
            [onnx_task, stt_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        fired = done.pop()
        stt_wake_transcript: Optional[str] = fired.result() if fired is stt_task else None
        wake_source = "stt" if stt_wake_transcript else "onnx"

        t_wake = time.monotonic()
        log.event("wake.fire", source=wake_source,
                  transcript=stt_wake_transcript or "")
        console.print(f"[green]● Wake[/green] [dim]({wake_source})[/dim]")
        self._set_status("wake")
        self.audio.play_tone(880, 0.06, amplitude=0.18)

        # If STT-wake captured a command in the same utterance, use it directly;
        # otherwise capture the next utterance for the command.
        initial_command: Optional[str] = None
        if stt_wake_transcript:
            initial_command = _strip_wake_prefix(stt_wake_transcript)
            if initial_command:
                log.event("stt_wake.has_command", command=initial_command)

        if initial_command:
            if _is_end_phrase(initial_command):
                log.event("wake.end_phrase_echo", text=initial_command)
                console.print("[yellow]Конец связи (после wake) — игнорирую.[/yellow]")
                return
            text = initial_command
            t_stt = time.monotonic()
            console.print(f"[bold blue]👤 {text}[/bold blue]")
            await self._run_brain(text, t_wake, t_stt)
        else:
            ok = await self._capture_and_handle(t_wake)
            if not ok:
                return

        # Continuous session — keep handling follow-up commands without re-waking
        # until the user says an end phrase, stays silent too long, OR background
        # noise hallucinates too many times in a row.
        consecutive_hallucinations = 0
        MAX_HALLUCINATIONS = 4  # exit continuous after this many garbage transcripts
        while True:
            console.print(
                "[dim]Слушаю продолжение… "
                "(скажи 'конец связи' чтобы выйти)[/dim]"
            )
            await asyncio.sleep(0.3)  # grace so mic doesn't pick up TTS tail
            try:
                pcm = await asyncio.wait_for(
                    self.vad.capture_utterance(self.audio),
                    timeout=_FOLLOWUP_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                log.event("followup.timeout")
                console.print("[dim]Время ожидания истекло, возвращаюсь к ожиданию wake.[/dim]")
                break

            if len(pcm) < 4000:
                continue

            try:
                text = await self.stt.transcribe(pcm, 16000)
            except Exception as e:
                log.event("stt.failed", error=str(e), phase="followup")
                console.print(f"[red]STT failed: {e}[/red]")
                continue

            text = (text or "").strip()
            log.event("followup.heard", text=text)
            if _is_end_phrase(text):
                log.event("followup.end_phrase", text=text)
                console.print("[yellow]Конец связи.[/yellow]")
                self.audio.play_tone(440, 0.12, amplitude=0.18)
                # Wait for the tone + any residual mic audio to clear so the
                # next wake-detector doesn't fire on the just-spoken phrase.
                await asyncio.sleep(0.8)
                break

            # If user re-spoke the wake word in the follow-up (alone or with
            # a command), treat as a fresh wake event.
            if _has_stt_wake(text):
                consecutive_hallucinations = 0
                stripped = _strip_wake_prefix(text)
                console.print(f"[green]● Re-wake[/green] [dim]({text!r})[/dim]")
                self.audio.play_tone(880, 0.06, amplitude=0.18)
                if stripped:
                    t_followup = time.monotonic()
                    console.print(f"[bold blue]👤 {stripped}[/bold blue]")
                    await self._run_brain(stripped, t_followup, t_followup)
                else:
                    await self._capture_and_handle(time.monotonic())
                continue

            if _looks_like_hallucination(text):
                consecutive_hallucinations += 1
                log.event("followup.dropped_hallucination",
                          text=text, streak=consecutive_hallucinations)
                console.print(
                    f"[dim]Пропускаю '{text}' (шум, {consecutive_hallucinations}/{MAX_HALLUCINATIONS})[/dim]"
                )
                if consecutive_hallucinations >= MAX_HALLUCINATIONS:
                    log.event("followup.exit_too_much_noise")
                    console.print("[dim]Слишком много шума, выхожу из диалога.[/dim]")
                    break
                continue

            consecutive_hallucinations = 0
            console.print(f"[bold blue]👤 {text}[/bold blue]")
            t_followup = time.monotonic()
            await self._run_brain(text, t_followup, t_followup)

    async def _capture_and_handle(self, t_wake: float) -> bool:
        """Capture one utterance, transcribe, and run the brain. Returns True
        if a command was actually handled, False if it was empty/too short or
        was an end-phrase echo (so we should NOT enter continuous mode)."""
        pcm = await self.vad.capture_utterance(self.audio)
        t_vad = time.monotonic()
        if len(pcm) < 4000:
            console.print("[yellow]Слишком коротко, игнорирую.[/yellow]")
            return False
        log.event("vad.captured", bytes=len(pcm), ms_since_wake=int((t_vad - t_wake) * 1000))

        try:
            text = await self.stt.transcribe(pcm, 16000)
        except Exception as e:
            log.event("stt.failed", error=str(e))
            console.print(f"[red]STT failed: {e}[/red]")
            return False
        t_stt = time.monotonic()

        text = (text or "").strip()
        if len(text) < 2:
            log.event("stt.empty")
            return False

        # If the wake-triggered utterance is an end-phrase, treat it as a
        # session close, not a command. This catches the case where a phantom
        # onnx wake fires right after the user said "Конец связи" — without
        # this guard, the LLM gets called with "конец связи" and replies.
        if _is_end_phrase(text):
            log.event("wake.end_phrase_echo", text=text)
            console.print("[yellow]Конец связи (после wake) — игнорирую.[/yellow]")
            return False

        log.event("stt.text", text=text, ms_stt=int((t_stt - t_vad) * 1000))
        console.print(f"[bold blue]👤 {text}[/bold blue]")

        await self._run_brain(text, t_wake, t_stt)
        return True

    async def _run_brain(self, text: str, t_wake: float, t_stt: float) -> None:
        """LLM streams sentences; TTS plays them as they arrive. A parallel
        barge-in monitor listens for the wake word during playback — if the
        user says 'Максим' over the assistant's voice, we cancel TTS and
        immediately treat it as a new turn."""
        self._set_status("thinking")
        first_audio_logged = [False]
        talking_started = [False]
        cancel_event = asyncio.Event()

        async def sentence_iter() -> AsyncIterator[str]:
            async for sentence in self.brain.stream_turn(text):
                if cancel_event.is_set():
                    return
                if not first_audio_logged[0]:
                    log.event("llm.first_sentence",
                              ms_from_stt=int((time.monotonic() - t_stt) * 1000),
                              text=sentence)
                    first_audio_logged[0] = True
                if not talking_started[0]:
                    self._set_status("talking")
                    talking_started[0] = True
                console.print(f"[green]🤖 {sentence}[/green]")
                yield sentence

        tts_task = asyncio.create_task(self.tts.stream(sentence_iter(), self.audio, cancel_event))
        # Barge-in: separate wake detector with lower threshold (set in __init__)
        # so the user can interrupt over their assistant's voice. Using the same
        # detector as the main loop would clash on model state.
        bargein_task = asyncio.create_task(self.bargein_wake.wait_for_trigger(self.audio))

        bargein_fired = False
        try:
            done, _ = await asyncio.wait(
                [tts_task, bargein_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if bargein_task in done and not bargein_task.cancelled():
                bargein_fired = True
                log.event("tts.bargein")
                console.print("[yellow]⏸  Перебили![/yellow]")
                cancel_event.set()
                self.audio.stop_playback() if hasattr(self.audio, "stop_playback") else None
            # Always cancel the loser
            for t in (tts_task, bargein_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            # Surface any TTS exception that finished first.
            if tts_task.done() and not tts_task.cancelled():
                exc = tts_task.exception()
                if exc:
                    log.event("tts.failed", error=str(exc))
                    console.print(f"[red]TTS failed: {exc}[/red]")
        finally:
            if not bargein_fired:
                await self._drain_playback()

        log.event("turn.done", total_ms=int((time.monotonic() - t_wake) * 1000),
                  bargein=bargein_fired)

        if bargein_fired:
            # User interrupted — treat exactly like a fresh wake event:
            # play the wake tone, give a brief grace period for any audio echo
            # to subside, then capture the next utterance.
            self._set_status("wake")
            self.audio.play_tone(880, 0.06, amplitude=0.18)
            await asyncio.sleep(0.3)
            await self._capture_and_handle(time.monotonic())
            return

        self._set_status("session")

    async def run(self) -> None:
        self.audio.start()
        log.event("system.start", model=settings.llm_model,
                  voice=settings.elevenlabs_voice_id, wake=settings.wake_word)
        console.print("[bold green]Максим запущен.[/bold green]")
        try:
            while True:
                try:
                    await self._one_turn()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.event("turn.error", error=str(e))
                    console.print(f"[red]Ошибка цикла: {e}[/red]")
                    await asyncio.sleep(0.5)
        finally:
            self.audio.stop()
            log.event("system.stop")
