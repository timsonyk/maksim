"""Audio I/O — continuous mic capture into a numpy ring buffer with a
monotonic sample counter so consumers (wake-word, VAD) can track their own
read positions safely even when the ring overwrites old samples.

All mic samples are 16 kHz mono int16. Playback supports any sample rate
and is dispatched on a worker thread."""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from .config import settings


class AudioIO:
    def __init__(
        self,
        sample_rate: int = 16000,
        input_device: Optional[str] = None,
        output_device: Optional[str] = None,
        ring_seconds: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.input_device = input_device or None
        self.output_device = output_device or None

        self._ring_size = int(sample_rate * ring_seconds)
        self._ring = np.zeros(self._ring_size, dtype=np.int16)
        self._write_idx = 0          # ring index where next sample lands
        self._total_in = 0           # monotonic global counter
        self._ring_lock = threading.Lock()
        self._new_samples_event = threading.Event()
        self._in_stream: Optional[sd.InputStream] = None

        self._play_queue: queue.Queue = queue.Queue()
        self._play_thread: Optional[threading.Thread] = None
        self._play_stop = threading.Event()
        self._playback_active = threading.Event()

    # ── Mic ───────────────────────────────────────────────────────────
    def _in_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata
        n = len(mono)
        with self._ring_lock:
            i = self._write_idx
            end = i + n
            if end <= self._ring_size:
                self._ring[i:end] = mono
            else:
                cut = self._ring_size - i
                self._ring[i:] = mono[:cut]
                self._ring[: end - self._ring_size] = mono[cut:]
            self._write_idx = end % self._ring_size
            self._total_in += n
        self._new_samples_event.set()

    def start(self) -> None:
        if self._in_stream is not None:
            return
        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(self.sample_rate * 0.04),  # 40 ms
            device=self.input_device,
            callback=self._in_callback,
        )
        self._in_stream.start()
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()

    def stop(self) -> None:
        if self._in_stream is not None:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None
        self._play_stop.set()
        self._play_queue.put(None)

    # ── Reading the ring ──────────────────────────────────────────────
    @property
    def total_samples(self) -> int:
        with self._ring_lock:
            return self._total_in

    def since(self, pos: int) -> tuple[np.ndarray, int]:
        """Return samples written after position `pos` and the new position.
        If we've lost samples (consumer fell behind by >ring), we return only
        what's still in the ring and the new position is advanced past the
        gap so the consumer naturally resyncs."""
        with self._ring_lock:
            new_pos = self._total_in
            available = new_pos - pos
            if available <= 0:
                return np.zeros(0, dtype=np.int16), pos
            # Cap to ring capacity; we may have overwritten old samples.
            if available > self._ring_size:
                available = self._ring_size
            # Build the linear view from the circular ring.
            start = (self._write_idx - available) % self._ring_size
            if start + available <= self._ring_size:
                out = self._ring[start : start + available].copy()
            else:
                first = self._ring_size - start
                out = np.concatenate([
                    self._ring[start:],
                    self._ring[: available - first],
                ])
        return out, new_pos

    def read_recent(self, n_samples: int) -> np.ndarray:
        """Return the most recent up-to-n_samples samples."""
        with self._ring_lock:
            avail = min(n_samples, self._total_in, self._ring_size)
            start = (self._write_idx - avail) % self._ring_size
            if start + avail <= self._ring_size:
                out = self._ring[start : start + avail].copy()
            else:
                first = self._ring_size - start
                out = np.concatenate([
                    self._ring[start:],
                    self._ring[: avail - first],
                ])
        return out

    def wait_for_samples(self, timeout: float = 0.5) -> None:
        self._new_samples_event.wait(timeout)
        self._new_samples_event.clear()

    # ── Playback ──────────────────────────────────────────────────────
    def play_pcm(self, pcm_int16: bytes, sample_rate: int) -> None:
        if not pcm_int16:
            return
        self._play_queue.put((pcm_int16, sample_rate))

    def play_tone(self, freq: float = 880.0, duration: float = 0.08,
                  amplitude: float = 0.25) -> None:
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)
        n_fade = int(sr * 0.01)
        if len(wave) > 2 * n_fade:
            ramp = np.linspace(0, 1, n_fade, dtype=np.float32)
            wave[:n_fade] = (wave[:n_fade].astype(np.float32) * ramp).astype(np.int16)
            wave[-n_fade:] = (wave[-n_fade:].astype(np.float32) * ramp[::-1]).astype(np.int16)
        self.play_pcm(wave.tobytes(), sr)

    def is_playing(self) -> bool:
        return self._playback_active.is_set() or not self._play_queue.empty()

    def stop_playback(self) -> None:
        """Drain any queued audio without playing it — used for barge-in."""
        try:
            while True:
                self._play_queue.get_nowait()
        except queue.Empty:
            pass
        self._playback_active.clear()

    def _play_worker(self) -> None:
        current_sr = None
        out: Optional[sd.OutputStream] = None
        try:
            while not self._play_stop.is_set():
                item = self._play_queue.get()
                if item is None:
                    break
                pcm, sr = item
                self._playback_active.set()
                if out is None or current_sr != sr:
                    if out is not None:
                        out.stop()
                        out.close()
                    out = sd.OutputStream(
                        samplerate=sr,
                        channels=1,
                        dtype="int16",
                        device=self.output_device,
                    )
                    out.start()
                    current_sr = sr
                arr = np.frombuffer(pcm, dtype=np.int16)
                step = sr // 10  # 100 ms slices
                for i in range(0, len(arr), step):
                    out.write(arr[i : i + step])
                if self._play_queue.empty():
                    self._playback_active.clear()
        finally:
            if out is not None:
                try:
                    out.stop()
                    out.close()
                except Exception:
                    pass


def make_audio() -> AudioIO:
    return AudioIO(
        sample_rate=settings.sample_rate,
        input_device=settings.input_device or None,
        output_device=settings.output_device or None,
    )
