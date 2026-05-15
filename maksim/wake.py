"""Wake-word detection using openWakeWord pre-trained models."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from .config import settings


class WakeWordDetector:
    """Detects the wake word in a continuous 16 kHz int16 mono stream.
    openWakeWord recommends 1280-sample (80 ms) chunks at 16 kHz.

    `keyword` may be either a built-in model name (e.g. ``hey_jarvis``) which
    will be auto-downloaded, or a path to a local ``.onnx`` file (absolute, or
    relative to the project root).
    """

    CHUNK = 1280

    def __init__(self, keyword: str = "hey_jarvis", threshold: float = 0.6):
        from openwakeword.model import Model
        from openwakeword.utils import download_models

        is_local_path = keyword.endswith(".onnx") or "/" in keyword
        if is_local_path:
            p = Path(keyword)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / p
            if not p.exists():
                raise FileNotFoundError(f"Wake model not found: {p}")
            model_ref = str(p)
        else:
            try:
                download_models(model_names=[keyword])
            except Exception:
                pass
            model_ref = keyword

        self.keyword = keyword
        self.threshold = threshold
        self.model = Model(wakeword_models=[model_ref], inference_framework="onnx")
        # Score that crossed the threshold on the last wait_for_trigger() — exposed
        # so the orchestrator can include it in the wake.fire log event for tuning.
        self.last_score: float = 0.0

    def _predict_max(self, pcm_int16: np.ndarray) -> float:
        scores = self.model.predict(pcm_int16)
        return max(scores.values()) if scores else 0.0

    def reset(self) -> None:
        self.model.reset()

    async def wait_for_trigger(self, audio) -> None:
        """Block (async) until the keyword fires."""
        self.reset()
        loop = asyncio.get_running_loop()
        pos = audio.total_samples
        carry = np.zeros(0, dtype=np.int16)
        while True:
            await asyncio.sleep(0.04)
            new, pos = audio.since(pos)
            if new.size == 0:
                continue
            buf = np.concatenate([carry, new]) if carry.size else new
            usable = (len(buf) // self.CHUNK) * self.CHUNK
            if usable == 0:
                carry = buf
                continue
            carry = buf[usable:]
            chunks = buf[:usable].reshape(-1, self.CHUNK)
            for chunk in chunks:
                score = await loop.run_in_executor(None, self._predict_max, chunk)
                if score >= self.threshold:
                    self.last_score = float(score)
                    return


def make_wake() -> WakeWordDetector:
    return WakeWordDetector(settings.wake_word, settings.wake_threshold)
