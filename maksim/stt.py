"""Speech-to-text. Default = hybrid (Groq cloud, local fallback)."""
from __future__ import annotations

import asyncio
import io
import wave
from typing import Optional

import numpy as np

from .config import settings


def _pcm_to_wav_bytes(pcm_int16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16)
    return buf.getvalue()


class GroqSTT:
    def __init__(self, api_key: str, model: str, language: str = "ru"):
        from groq import AsyncGroq

        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.language = language

    async def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        wav = _pcm_to_wav_bytes(pcm_int16, sample_rate)
        resp = await self.client.audio.transcriptions.create(
            file=("speech.wav", wav, "audio/wav"),
            model=self.model,
            language=self.language,
            response_format="text",
            temperature=0.0,
        )
        return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))


class LocalSTT:
    """faster-whisper on CPU (Apple Silicon arm64 with int8)."""

    def __init__(self, model_size: str = "large-v3", compute_type: str = "int8",
                 language: str = "ru"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        self.language = language

    def _sync_transcribe(self, pcm_int16: bytes) -> str:
        arr = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            arr,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()

    async def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        # faster-whisper expects 16 kHz mono; AudioIO is already 16 kHz mono.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_transcribe, pcm_int16)


class HybridSTT:
    """Groq first; on any failure, fall back to local."""

    def __init__(self, groq: GroqSTT, local_factory):
        self.groq = groq
        self._local_factory = local_factory
        self._local: Optional[LocalSTT] = None

    async def transcribe(self, pcm_int16: bytes, sample_rate: int = 16000) -> str:
        try:
            return await asyncio.wait_for(
                self.groq.transcribe(pcm_int16, sample_rate),
                timeout=4.0,
            )
        except Exception:
            if self._local is None:
                self._local = self._local_factory()
            return await self._local.transcribe(pcm_int16, sample_rate)


def make_stt():
    provider = settings.stt_provider
    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for stt_provider=groq")
        return GroqSTT(settings.groq_api_key, settings.groq_stt_model, settings.stt_language)

    if provider == "local":
        return LocalSTT(settings.local_stt_model, settings.local_stt_compute,
                        settings.stt_language)

    # hybrid
    if not settings.groq_api_key:
        return LocalSTT(settings.local_stt_model, settings.local_stt_compute,
                        settings.stt_language)
    groq = GroqSTT(settings.groq_api_key, settings.groq_stt_model, settings.stt_language)

    def _local_factory():
        return LocalSTT(settings.local_stt_model, settings.local_stt_compute,
                        settings.stt_language)

    return HybridSTT(groq, _local_factory)
