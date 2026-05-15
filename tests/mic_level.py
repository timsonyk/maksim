"""Show real-time RMS of the mic. If you see numbers > a few hundred when
you talk, the mic is delivering audio. If it stays at 0 — TCC blocked it."""
import time
import numpy as np
import sounddevice as sd

sr = 16000
print(f"Mic devices visible to sounddevice:")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"  [{i}] {d['name']} (in={d['max_input_channels']}, sr={d['default_samplerate']})")
print(f"\nDefault input: {sd.default.device}\n")

print("Listening for 8 seconds — TALK NOW. (peak should be > 1000 when you talk)")
peaks = []
def cb(indata, frames, t, status):
    mono = indata[:, 0] if indata.ndim > 1 else indata
    peak = int(np.abs(mono).max())
    rms = int(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
    peaks.append(peak)
    bar = "█" * min(60, peak // 200)
    print(f"\rpeak={peak:5d}  rms={rms:5d}  {bar:<60}", end="", flush=True)

with sd.InputStream(samplerate=sr, channels=1, dtype="int16", callback=cb,
                    blocksize=int(sr * 0.1)):
    time.sleep(8)

print()
if max(peaks) < 100:
    print("❌ Mic is silent. macOS likely denied mic access to this Python binary.")
    print("   Open: System Settings ▸ Privacy & Security ▸ Microphone")
    print("   Grant access to Terminal (or whatever ran this).")
else:
    print(f"✅ Mic OK. Max peak seen: {max(peaks)}")
