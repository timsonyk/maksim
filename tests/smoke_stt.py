"""Wake → VAD → STT → print transcription."""
import asyncio
import time

from maksim.audio import make_audio
from maksim.stt import make_stt
from maksim.vad import make_vad
from maksim.wake import make_wake


async def main():
    a = make_audio()
    a.start()
    w = make_wake()
    v = make_vad()
    s = make_stt()
    print("Say 'Hey Maksim', then a phrase in Russian.")
    await w.wait_for_trigger(a)
    a.play_tone(880, 0.06)
    pcm = await v.capture_utterance(a)
    t0 = time.monotonic()
    text = await s.transcribe(pcm, 16000)
    print(f"[{int((time.monotonic() - t0) * 1000)} ms] → {text!r}")
    a.stop()


if __name__ == "__main__":
    asyncio.run(main())
