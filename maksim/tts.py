"""ElevenLabs Flash v2.5 streaming over WebSocket.

We open one WS per LLM turn and feed sentences incrementally as they
become available — the model produces audio for partial input, so the
first audio chunk arrives in ~150 ms.

Output format: raw PCM 22050 Hz 16-bit mono (no client-side decoding)."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator

import websockets

from . import log
from .config import settings


def _ws_url(voice_id: str, model_id: str, output_format: str) -> str:
    return (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        f"?model_id={model_id}"
        f"&output_format={output_format}"
        f"&auto_mode=true"
    )


_PCM_RATES = {
    "pcm_16000": 16000,
    "pcm_22050": 22050,
    "pcm_24000": 24000,
    "pcm_44100": 44100,
}


class TtsStreamer:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "pcm_22050",
        stability: float = 0.5,
        similarity: float = 0.8,
        speed: float = 1.0,
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.sample_rate = _PCM_RATES.get(output_format, 22050)
        self.voice_settings = {
            "stability": stability,
            "similarity_boost": similarity,
            "speed": speed,
        }

    async def stream(
        self,
        sentences: AsyncIterator[str],
        audio_io,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        url = _ws_url(self.voice_id, self.model_id, self.output_format)
        headers = {"xi-api-key": self.api_key}
        log.event("tts.open", voice=self.voice_id, model=self.model_id)
        try:
            async with websockets.connect(
                url, additional_headers=headers, ping_interval=20, max_size=None
            ) as ws:
                # BOS
                await ws.send(json.dumps({
                    "text": " ",
                    "voice_settings": self.voice_settings,
                }))

                async def sender():
                    async for sentence in sentences:
                        if not sentence.strip():
                            continue
                        if cancel_event and cancel_event.is_set():
                            break
                        await ws.send(json.dumps({"text": sentence + " "}))
                    # EOS — flush remaining audio and close server side
                    await ws.send(json.dumps({"text": ""}))

                async def receiver():
                    async for msg in ws:
                        if cancel_event and cancel_event.is_set():
                            break
                        try:
                            data = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        if data.get("audio"):
                            pcm = base64.b64decode(data["audio"])
                            audio_io.play_pcm(pcm, self.sample_rate)
                        if data.get("isFinal"):
                            break
                        err = data.get("error")
                        if err:
                            log.event("tts.error", error=str(err))

                await asyncio.gather(sender(), receiver())
        except Exception as e:
            log.event("tts.failed", error=str(e))
            raise


class SayTTS:
    """macOS built-in `say` synthesizer. Plays each sentence via the system's
    audio output (not via AudioIO.play_pcm), so we sleep briefly at the end
    so room echo decays before the mic re-listens.

    Works offline, no API key, no geo block. Russian voice: Milena (default).
    """

    def __init__(self, voice: str = "Milena", rate: int = 200):
        self.voice = voice
        self.rate = rate  # words per minute

    async def stream(
        self,
        sentences: AsyncIterator[str],
        audio_io,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        log.event("tts.open", voice=self.voice, model="macos-say")
        for_count = 0
        try:
            async for sentence in sentences:
                if not sentence.strip():
                    continue
                if cancel_event and cancel_event.is_set():
                    break
                proc = await asyncio.create_subprocess_exec(
                    "/usr/bin/say",
                    "-v", self.voice,
                    "-r", str(self.rate),
                    sentence,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    await proc.wait()
                except asyncio.CancelledError:
                    proc.terminate()
                    raise
                for_count += 1
            # Brief decay so the mic doesn't pick up the speaker's tail when
            # the continuous-session loop starts listening again.
            await asyncio.sleep(0.2)
        except Exception as e:
            log.event("tts.failed", error=str(e))
            raise


class SileroTTS:
    """Silero TTS v4 — Russian, fully local, runs on CPU/MPS.

    Speakers: aidar (male, default), eugene (male), baya/kseniya/xenia (female).
    Latency: ~200-500ms/sentence on M2. Output: 24kHz int16 PCM → AudioIO.play_pcm.
    Model auto-downloaded by torch.hub on first init (~30MB), cached in
    ~/.cache/torch/hub.
    """

    SAMPLE_RATE = 24000

    def __init__(self, speaker: str = "aidar"):
        import torch
        self._torch = torch
        self.speaker = speaker
        self.device = torch.device("cpu")  # MPS sometimes glitches Silero
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        self.model.to(self.device)

    def _synthesize(self, text: str) -> bytes:
        # Generate 16-bit PCM bytes at 24kHz.
        import numpy as np
        audio = self.model.apply_tts(
            text=text, speaker=self.speaker, sample_rate=self.SAMPLE_RATE,
            put_accent=True, put_yo=True,
        )
        arr = audio.cpu().numpy().astype("float32")
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767).astype("int16").tobytes()

    async def stream(
        self,
        sentences: AsyncIterator[str],
        audio_io,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        log.event("tts.open", voice=self.speaker, model="silero_v4_ru")
        loop = asyncio.get_running_loop()
        try:
            async for sentence in sentences:
                if not sentence.strip():
                    continue
                if cancel_event and cancel_event.is_set():
                    break
                try:
                    pcm = await loop.run_in_executor(None, self._synthesize, sentence)
                except Exception as e:
                    log.event("tts.synth_failed", error=str(e), text=sentence[:200])
                    continue
                if cancel_event and cancel_event.is_set():
                    break
                audio_io.play_pcm(pcm, self.SAMPLE_RATE)
        except Exception as e:
            log.event("tts.failed", error=str(e))
            raise


def make_tts():
    provider = (settings.tts_provider or "silero").lower()
    if provider == "silero":
        return SileroTTS(speaker=settings.silero_speaker)
    if provider == "say":
        return SayTTS(voice=settings.say_voice, rate=settings.say_rate)
    if not settings.elevenlabs_api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for tts_provider=elevenlabs")
    return TtsStreamer(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,
        model_id=settings.elevenlabs_model_id,
        output_format=settings.elevenlabs_output_format,
    )
