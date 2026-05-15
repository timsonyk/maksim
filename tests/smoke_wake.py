"""Print when the wake word fires. Ctrl-C to stop."""
import asyncio

from maksim.audio import make_audio
from maksim.wake import make_wake


async def main():
    a = make_audio()
    a.start()
    w = make_wake()
    print(f"Listening for '{w.keyword}' (threshold={w.threshold}). Speak the wake word.")
    try:
        while True:
            await w.wait_for_trigger(a)
            a.play_tone(880, 0.08)
            print("🔔 WAKE")
    except KeyboardInterrupt:
        pass
    finally:
        a.stop()


if __name__ == "__main__":
    asyncio.run(main())
