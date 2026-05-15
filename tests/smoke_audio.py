"""Verify mic + speakers work. Plays a tone, records 3 s, plays it back."""
import time

import numpy as np

from maksim.audio import make_audio


def main():
    a = make_audio()
    a.start()
    print("Beep…")
    a.play_tone(880, 0.2)
    time.sleep(0.4)
    print("Recording 3 s — say something.")
    time.sleep(3.0)
    pcm = a.read_recent(16000 * 3)
    print(f"Captured {len(pcm)} samples, peak={np.abs(pcm).max()}")
    print("Playback…")
    a.play_pcm(pcm.tobytes(), 16000)
    time.sleep(3.5)
    a.stop()


if __name__ == "__main__":
    main()
