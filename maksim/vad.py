"""Silero-VAD-based utterance endpointer.

After wake-word fires, keep recording until the user goes silent for
`silence_ms`. Returns the captured utterance as int16 PCM @ 16 kHz."""
from __future__ import annotations

import asyncio
import time
from typing import List

import numpy as np

from .config import settings


class VADEndpointer:
    FRAME = 512  # Silero v5 requires 512 samples at 16 kHz (~32 ms)

    def __init__(
        self,
        silence_ms: int = 600,
        min_speech_ms: int = 200,
        max_utterance_s: int = 15,
        threshold: float = 0.5,
    ):
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=True)
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_s = max_utterance_s
        self.threshold = threshold

    def _prob(self, pcm_int16: np.ndarray) -> float:
        import torch
        x = torch.from_numpy(pcm_int16.astype(np.float32) / 32768.0)
        return float(self.model(x, 16000).item())

    async def capture_utterance(self, audio) -> bytes:
        loop = asyncio.get_running_loop()
        try:
            self.model.reset_states()
        except Exception:
            pass

        collected: List[np.ndarray] = []
        speech_ms = 0.0
        silence_ms = 0.0
        started = False
        t0 = time.monotonic()
        pos = audio.total_samples
        carry = np.zeros(0, dtype=np.int16)

        while True:
            if time.monotonic() - t0 > self.max_utterance_s:
                break

            await asyncio.sleep(0.02)
            new, pos = audio.since(pos)
            if new.size == 0:
                continue
            buf = np.concatenate([carry, new]) if carry.size else new
            usable = (len(buf) // self.FRAME) * self.FRAME
            if usable == 0:
                carry = buf
                continue
            carry = buf[usable:]
            frames = buf[:usable].reshape(-1, self.FRAME)

            for frame in frames:
                p = await loop.run_in_executor(None, self._prob, frame)
                frame_ms = 32.0
                if p >= self.threshold:
                    speech_ms += frame_ms
                    silence_ms = 0
                    if speech_ms >= self.min_speech_ms:
                        started = True
                    collected.append(frame)
                else:
                    if started:
                        silence_ms += frame_ms
                        collected.append(frame)
                        if silence_ms >= self.silence_ms:
                            return np.concatenate(collected).tobytes()
                    else:
                        collected.append(frame)
                        if len(collected) > 8:
                            collected = collected[-8:]

        if collected:
            return np.concatenate(collected).tobytes()
        return b""


def make_vad() -> VADEndpointer:
    return VADEndpointer(
        silence_ms=settings.vad_silence_ms,
        min_speech_ms=settings.vad_min_speech_ms,
        max_utterance_s=settings.vad_max_utterance_s,
    )
