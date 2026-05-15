"""Synthesize a Russian phrase via ElevenLabs and play it."""
import asyncio
import time

from maksim.audio import make_audio
from maksim.tts import make_tts


async def main():
    a = make_audio()
    a.start()
    tts = make_tts()

    phrases = [
        "Привет, Тимофей.",
        "Я Максим, твой голосовой ассистент.",
        "Готов помочь.",
    ]

    async def gen():
        for p in phrases:
            yield p

    t0 = time.monotonic()
    await tts.stream(gen(), a)
    while a.is_playing():
        await asyncio.sleep(0.1)
    print(f"Total: {int((time.monotonic() - t0) * 1000)} ms")
    a.stop()


if __name__ == "__main__":
    asyncio.run(main())
