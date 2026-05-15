"""Wake → VAD capture → save utterance as utterance.wav."""
import asyncio
import wave

from maksim.audio import make_audio
from maksim.vad import make_vad
from maksim.wake import make_wake


async def main():
    a = make_audio()
    a.start()
    w = make_wake()
    v = make_vad()
    print("Say 'Hey Maksim', then a phrase.")
    await w.wait_for_trigger(a)
    a.play_tone(880, 0.06)
    print("Recording until silence…")
    pcm = await v.capture_utterance(a)
    print(f"Captured {len(pcm)} bytes ({len(pcm)/2/16000:.2f} s).")
    with wave.open("utterance.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(pcm)
    print("Saved utterance.wav")
    a.stop()


if __name__ == "__main__":
    asyncio.run(main())
