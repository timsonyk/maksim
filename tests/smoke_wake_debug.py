"""Live wake-word score viewer — see what the model thinks every 80 ms."""
import asyncio
import sys
import time
import numpy as np

from maksim.audio import make_audio


async def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 999.0
    deadline = time.monotonic() + duration
    from openwakeword.model import Model
    from openwakeword.utils import download_models
    try:
        download_models(model_names=["hey_jarvis"])
    except Exception:
        pass
    m = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    a = make_audio()
    a.start()
    print("Скажи 'Hey Maksim' — увидишь scores в реальном времени.")
    print("Score > 0.5 = вероятный wake. Ctrl-C чтобы выйти.\n")

    pos = a.total_samples
    carry = np.zeros(0, dtype=np.int16)
    CHUNK = 1280
    loop = asyncio.get_running_loop()
    last_high = 0.0

    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.04)
            new, pos = a.since(pos)
            if new.size == 0:
                continue
            buf = np.concatenate([carry, new]) if carry.size else new
            usable = (len(buf) // CHUNK) * CHUNK
            if usable == 0:
                carry = buf; continue
            carry = buf[usable:]
            chunks = buf[:usable].reshape(-1, CHUNK)
            for chunk in chunks:
                peak = int(np.abs(chunk).max())
                scores = await loop.run_in_executor(None, m.predict, chunk)
                score = max(scores.values()) if scores else 0.0
                bar = "█" * int(score * 50)
                marker = " 🔔" if score > 0.5 else ""
                if score > 0.05 or peak > 2000:
                    print(f"score={score:.3f}  peak={peak:5d}  {bar:<50}{marker}")
                if score > last_high:
                    last_high = score
    except KeyboardInterrupt:
        pass
    print(f"\nMax score: {last_high:.3f}")
    a.stop()


if __name__ == "__main__":
    asyncio.run(main())
