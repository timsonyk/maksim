"""Generates scripts/train_wake_word.ipynb — a Colab notebook that trains a custom
openWakeWord .onnx model for a Russian wake word (default: "максим") using Silero
TTS for synthetic sample generation. Piper has no Russian voices, hence Silero.

To regenerate the notebook after editing this file:
    uv run python scripts/build_train_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "train_wake_word.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells: list[dict] = []

# ── Intro ──────────────────────────────────────────────────────────
cells.append(md(
"""# Train a Russian wake word — Colab notebook

Trains a custom openWakeWord `.onnx` model for a Russian wake word. Default is **максим**,
but you can change it in §2 (Parameters) to anything (`джарвис`, `алиса`, `ник`, etc).

Synthetic positive samples are generated with **Silero TTS** (Piper has no Russian voices).
Negative adversarial samples cover Russian words/phrases that sound similar to the target,
plus generic Russian sentences for variance.

**Requirements:** Colab with GPU runtime (T4 free tier is enough).

**Time:** ~2-3 hours end-to-end (downloads + training). Sample generation itself: ~10-15 min.

**Output:** `my_custom_model/<MODEL_NAME>.onnx` — downloaded automatically at the end.
Put the `.onnx` file in `<project>/models/<MODEL_NAME>.onnx` and set in `.env.local`:
```
WAKE_WORD=models/<MODEL_NAME>.onnx
```

---
**To retrain for a different word:**
1. Edit §2 — change `TARGET_PHRASE`, `MODEL_NAME`, and `ADVERSARIAL_TEXTS`
2. Run §0 (Cleanup) to wipe previous samples/features
3. Run all cells from §1 onwards
"""))

# ── §0. Cleanup ─────────────────────────────────────────────────────
cells.append(md(
"""## 0. Cleanup (run only when retraining a different word)

Skip on first run. Wipes synthetic samples, augmented features, and previous .onnx so the
next run trains from scratch."""))

cells.append(code(
r"""# Uncomment the line below to wipe previous training artifacts:
# !rm -rf my_custom_model
# Also wipe Silero/torchhub cache only if you suspect a bad download:
# !rm -rf /root/.cache/torch/hub
print("Cleanup skipped (uncomment lines above to actually clean).")
"""))

# ── §1. Setup ───────────────────────────────────────────────────────
cells.append(md("## 1. Environment setup"))

cells.append(code(
r"""# Clone openWakeWord and install training dependencies.
# We skip piper-sample-generator entirely — it has no Russian voices.

!git clone https://github.com/dscripka/openwakeword
!pip install -e ./openwakeword

# Dependencies pinned to versions known to work with the openwakeword training script.
!pip install mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0
!pip install speechbrain==0.5.14
# torch-audiomentations 0.11.2 needed: 0.11.0 calls torchaudio.set_audio_backend() which was removed in torchaudio 2.1+
!pip install audiomentations==0.33.0 torch-audiomentations==0.11.2
!pip install acoustics==0.2.6
# tensorflow-cpu / tensorflow_probability / onnx_tf intentionally NOT installed — they were only needed
# for the optional .onnx → .tflite conversion which we do NOT use. We consume .onnx directly.
!pip install pronouncing==0.2.0
!pip install datasets==2.14.6
!pip install deep-phonemizer==0.0.19
!pip install webrtcvad
!pip install omegaconf  # Silero TTS dep
!pip install onnxscript  # required by torch.onnx.export in torch 2.x for final ONNX model export

import os, sys
os.makedirs("./openwakeword/openwakeword/resources/models", exist_ok=True)
!wget -q https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx
!wget -q https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite
!wget -q https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx
!wget -q https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite

# torchaudio 2.9+ removed torchaudio.info / torchaudio.load (migrated to TorchCodec).
# torch_audiomentations still uses them in utils/io.py. Patch the package file DIRECTLY —
# sitecustomize.py is unreliable in Colab dist-packages.
_io_path = "/usr/local/lib/python3.12/dist-packages/torch_audiomentations/utils/io.py"
with open(_io_path) as _f:
    _code_str = _f.read()
_shim = '''
# === SHIM (added) for torchaudio 2.9+ which removed .info / .load ===
import soundfile as _sf_shim
from dataclasses import dataclass as _dc_shim
import torch as _torch_shim

@_dc_shim
class _MetaShim:
    num_frames: int
    sample_rate: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"

def _torchaudio_info_shim(path):
    i = _sf_shim.info(str(path))
    return _MetaShim(num_frames=i.frames, sample_rate=i.samplerate, num_channels=i.channels)

def _torchaudio_load_shim(path, **kw):
    data, sr = _sf_shim.read(str(path), dtype="float32", always_2d=True)
    return _torch_shim.from_numpy(data.T).contiguous(), sr
# === END SHIM ===
'''
if "_torchaudio_info_shim" not in _code_str:
    _code_str = _shim + "\n" + _code_str
    _code_str = _code_str.replace("torchaudio.info(", "_torchaudio_info_shim(")
    _code_str = _code_str.replace("torchaudio.load(", "_torchaudio_load_shim(")
    with open(_io_path, "w") as _f:
        _f.write(_code_str)
    print("✓ Patched torch_audiomentations")

print("Setup done.")
"""))

cells.append(code(
r"""import os, sys, uuid, random
from pathlib import Path
import numpy as np
import torch
import scipy.io.wavfile
import scipy.signal
import yaml
import datasets
from tqdm import tqdm
"""))

# ── §2. Parameters ──────────────────────────────────────────────────
cells.append(md(
"""## 2. Parameters — edit these to train a different wake word

- `TARGET_PHRASE` — the exact word/phrase to detect (lowercase Russian recommended)
- `MODEL_NAME` — short ASCII slug for the .onnx filename (no spaces/Cyrillic)
- `ADVERSARIAL_TEXTS` — Russian words/phrases the model should NOT trigger on. **Critical** —
  include words that share phonemes with the target (e.g. for "максим" → "максимум", "максимально")
- `N_SAMPLES_TRAIN / N_SAMPLES_VAL` — synthetic clip counts (1000/200 is a baseline; bump to
  2000/400 for higher quality at 2× generation cost)
- `TRAINING_STEPS` — 10000 fast (~30 min), 50000 slow (~2-3h) and better quality
"""))

cells.append(code(
r"""# ════════════════ EDIT BELOW TO CHANGE WAKE WORD ════════════════
TARGET_PHRASE = "максим"
MODEL_NAME = "maksim"

# Words/phrases that should NOT trigger the wake word.
# Include phonetically-similar Russian words (sharing prefix/syllables with TARGET_PHRASE)
# plus generic Russian phrases for negative variance.
ADVERSARIAL_TEXTS = [
    # Phonetically similar to "максим" — REPLACE these when changing target word
    "максимум", "максимально", "максима", "максимка", "максимально удобно",
    "максимизация", "максимизировать", "до максимума", "на максимуме",
    "максим горький", "максим перепелица", "максимилиан", "максимиллиан",
    "максимально быстро", "максимально точно", "максимально просто",
    "максимально эффективно", "максимально комфортно", "максимально безопасно",
    "максимум возможностей", "максимум удовольствия", "максимум информации",
    "наибольший максимум", "локальный максимум", "глобальный максимум",
    "достичь максимума", "выйти на максимум", "находится в максимуме",
    "максимально допустимый", "максимально возможный", "максимально близкий",
    "максимально широкий", "максимально полный", "максимально подробный",
    "максима жизни", "жизненная максима", "философская максима",
    "максик",  # diminutive
    # Generic Russian phrases — keep as-is for any wake word
    "привет как дела", "что нового сегодня", "открой пожалуйста окно",
    "включи музыку", "выключи свет", "позвони маме", "напомни мне завтра",
    "какая погода сегодня", "сколько сейчас времени", "поставь будильник",
    "найди ближайшее кафе", "запиши заметку", "прочитай новости",
    "переведи на английский", "посчитай два плюс два", "расскажи анекдот",
    "что такое квантовая физика", "как приготовить борщ", "где находится Эверест",
    "кто написал войну и мир", "когда родился Пушкин", "столица Японии",
    "переведи деньги", "купи билет", "забронируй столик", "вызови такси",
    "отправь сообщение", "прочитай почту", "открой документ", "сохрани файл",
    "запусти приложение", "закрой программу", "перезагрузи компьютер",
    "выключи компьютер", "увеличь громкость", "уменьши яркость",
    "поставь на паузу", "переключи песню", "повтори последнее",
]

# Synthetic sample counts
N_SAMPLES_TRAIN = 1000
N_SAMPLES_VAL = 200

# Training steps — 10000 fast (~30 min), 50000 better quality (~2-3 hours)
TRAINING_STEPS = 10000
# ════════════════════════════════════════════════════════════════════

# Auto-generate positive text variants from TARGET_PHRASE (different casing/punctuation
# slightly changes Silero's prosody, increasing acoustic diversity)
_t = TARGET_PHRASE
_cap = _t[0].upper() + _t[1:] if _t else _t
POSITIVE_TEXTS = [
    _t, _cap, f"{_cap}.", f"{_cap}!", f"{_cap}?",
    f"{_t},", f"{_t}...", f"{_cap}, ",
]

# Output dir layout (matches openwakeword/train.py expectations)
OUT_ROOT = Path("./my_custom_model")
MODEL_DIR = OUT_ROOT / MODEL_NAME
POS_TRAIN = MODEL_DIR / "positive_train"
POS_TEST  = MODEL_DIR / "positive_test"
NEG_TRAIN = MODEL_DIR / "negative_train"
NEG_TEST  = MODEL_DIR / "negative_test"
for d in (POS_TRAIN, POS_TEST, NEG_TRAIN, NEG_TEST):
    d.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  Target: {TARGET_PHRASE!r}  |  Model: {MODEL_NAME}")
print(f"Positive variants: {POSITIVE_TEXTS}")
print(f"Adversarials: {len(ADVERSARIAL_TEXTS)}")
"""))

# ── §3. Silero TTS ──────────────────────────────────────────────────
cells.append(md(
"""## 3. Load Silero TTS (Russian)

Silero v4 provides 5 named Russian voices (`aidar`, `baya`, `kseniya`, `xenia`, `eugene`)
plus `random_voice` — sampled randomly each call, great for diversity."""))

cells.append(code(
r"""silero_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language="ru",
    speaker="v4_ru",
    trust_repo=True,
)
silero_model.to(device)

SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene", "random_voice"]
SILERO_SR = 24000  # then we resample to 16000

def silero_to_16k(text, speaker):
    # Generate one utterance, return int16 mono 16kHz PCM.
    audio = silero_model.apply_tts(
        text=text, speaker=speaker, sample_rate=SILERO_SR,
        put_accent=True, put_yo=True,
    )
    wav = audio.cpu().numpy().astype(np.float32)
    n_out = int(round(len(wav) * 16000 / SILERO_SR))
    wav16 = scipy.signal.resample(wav, n_out).astype(np.float32)
    wav16 = np.clip(wav16, -1.0, 1.0)
    return (wav16 * 32767).astype(np.int16)

# smoke-test
_test = silero_to_16k(POSITIVE_TEXTS[0], "aidar")
print("Silero OK, sample length:", len(_test) / 16000, "sec")
"""))

# ── §4. Generate positive ────────────────────────────────────────────
cells.append(md("""## 4. Generate positive samples"""))

cells.append(code(
r"""def fill_dir(out_dir, texts, n, label):
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= int(0.95 * n):
        print(f"[{label}] already has {len(existing)} files (target {n}), skipping")
        return
    needed = n - len(existing)
    pbar = tqdm(total=needed, desc=label)
    for _ in range(needed):
        text = random.choice(texts)
        speaker = random.choice(SPEAKERS)
        try:
            wav = silero_to_16k(text, speaker)
        except Exception:
            continue  # very short text + random_voice sometimes errors; retry
        if len(wav) < 4000:
            continue  # < 0.25s, too short
        target_len = 24000  # 1.5 seconds; openwakeword expects ~1-2s clips
        if len(wav) < target_len:
            wav = np.concatenate([wav, np.zeros(target_len - len(wav), dtype=np.int16)])
        else:
            wav = wav[:target_len]
        scipy.io.wavfile.write(out_dir / f"{uuid.uuid4().hex}.wav", 16000, wav)
        pbar.update(1)
    pbar.close()

fill_dir(POS_TRAIN, POSITIVE_TEXTS, N_SAMPLES_TRAIN, "positive_train")
fill_dir(POS_TEST,  POSITIVE_TEXTS, N_SAMPLES_VAL,   "positive_test")
"""))

# ── §5. Generate negative ────────────────────────────────────────────
cells.append(md("""## 5. Generate negative (adversarial) samples"""))

cells.append(code(
r"""fill_dir(NEG_TRAIN, ADVERSARIAL_TEXTS, N_SAMPLES_TRAIN, "negative_train")
fill_dir(NEG_TEST,  ADVERSARIAL_TEXTS, N_SAMPLES_VAL,   "negative_test")
"""))

# ── §6. Background data ─────────────────────────────────────────────
cells.append(md("""## 6. Download background data

MIT impulse responses (room reverb) + AudioSet (general noise) + FMA (music).
AudioSet shard may fail to download — that's OK, FMA alone suffices as background."""))

cells.append(code(
r"""# MIT Room Impulse Responses
os.makedirs("./mit_rirs", exist_ok=True)
rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
for row in tqdm(rir_dataset, desc="MIT RIRs"):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(os.path.join("./mit_rirs", name), 16000, (row['audio']['array']*32767).astype(np.int16))
"""))

cells.append(code(
r"""# AudioSet (~10GB extracted, 1 shard is enough). HF sometimes 403s — wrap in try/except.
os.makedirs("./audioset_16k", exist_ok=True)
try:
    if not os.path.exists("audioset"):
        os.mkdir("audioset")
    fname = "bal_train09.tar"
    link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
    !wget --tries=2 -O audioset/{fname} {link}
    !cd audioset && tar -xf bal_train09.tar
    flac_files = list(Path("audioset/audio").glob("**/*.flac"))
    if flac_files:
        audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in flac_files]})
        audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
        for row in tqdm(audioset_dataset, desc="AudioSet→16k"):
            name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
            scipy.io.wavfile.write(os.path.join("./audioset_16k", name), 16000, (row['audio']['array']*32767).astype(np.int16))
    else:
        print("⚠ AudioSet download produced no FLACs — proceeding with FMA only")
except Exception as e:
    print(f"⚠ AudioSet step failed ({e}) — proceeding with FMA only")
"""))

cells.append(code(
r"""# Free Music Archive (music background) — ~1 hour
os.makedirs("./fma", exist_ok=True)
fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))
n_hours = 1
for i in tqdm(range(n_hours * 3600 // 30), desc="FMA"):
    row = next(fma_dataset)
    name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
    scipy.io.wavfile.write(os.path.join("./fma", name), 16000, (row['audio']['array']*32767).astype(np.int16))
"""))

cells.append(code(
r"""# Pre-computed openwakeword features (negative training set + validation FP)
!wget -q https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
!wget -q https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
!ls -lh *.npy
"""))

# ── §7. Config YAML ─────────────────────────────────────────────────
cells.append(md("""## 7. Build training config YAML"""))

cells.append(code(
r"""# Only include audioset_16k as background if it has files (HF sometimes 403s the AudioSet tar).
_backgrounds = []
if any(Path("./audioset_16k").glob("*.wav")):
    _backgrounds.append("./audioset_16k")
_backgrounds.append("./fma")

config = {
    "model_name": MODEL_NAME,
    "target_phrase": [TARGET_PHRASE],
    "custom_negative_phrases": [],  # negatives pre-generated, so empty
    "n_samples": N_SAMPLES_TRAIN,
    "n_samples_val": N_SAMPLES_VAL,
    "tts_batch_size": 50,
    "augmentation_batch_size": 16,
    "augmentation_rounds": 1,
    "piper_sample_generator_path": "./piper-sample-generator",  # imported by train.py
    "output_dir": str(OUT_ROOT),
    "rir_paths": ["./mit_rirs"],
    "background_paths": _backgrounds,
    "background_paths_duplication_rate": [1] * len(_backgrounds),
    "false_positive_validation_data_path": "validation_set_features.npy",
    "feature_data_files": {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"},
    "batch_n_per_class": {
        "ACAV100M_sample": 1024,
        "adversarial_negative": 50,
        "positive": 50,
    },
    "model_type": "dnn",
    "layer_size": 32,
    "steps": TRAINING_STEPS,
    "max_negative_weight": 1500,
    "target_false_positives_per_hour": 0.2,
    "target_accuracy": 0.7,
    "target_recall": 0.5,
}
with open("my_model.yaml", "w") as f:
    yaml.dump(config, f, allow_unicode=True)

# train.py imports piper-sample-generator even when --generate_clips is off.
# Create a stub so the import doesn't crash (we never actually call generate_samples).
os.makedirs("piper-sample-generator", exist_ok=True)
with open("piper-sample-generator/generate_samples.py", "w") as f:
    f.write("def generate_samples(*args, **kwargs):\n    raise RuntimeError('stub — Silero pre-gen')\n")
print(f"Config written. Backgrounds: {_backgrounds}")
"""))

# ── §8. Augment + train ─────────────────────────────────────────────
cells.append(md(
"""## 8. Augment + train

`--generate_clips` is skipped — our directories are already populated by Silero.

If you previously ran augment and it errored mid-way, delete stale features first:
```python
!rm -f my_custom_model/{MODEL_NAME}/*.npy
```
(uncomment and run before augment if needed)."""))

cells.append(code(
r"""# Augment (mixes RIR + background into the pre-generated clips, computes features)
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips
"""))

cells.append(code(
r"""# Train
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model
"""))

# ── §9. Download model ──────────────────────────────────────────────
cells.append(md(
"""## 9. Download the trained model

After training, `my_custom_model/<MODEL_NAME>.onnx` is downloaded automatically.
Put it in your project at `<project>/models/<MODEL_NAME>.onnx` and set in `.env.local`:
```
WAKE_WORD=models/<MODEL_NAME>.onnx
WAKE_THRESHOLD=0.5
```"""))

cells.append(code(
r"""from google.colab import files
# torch.onnx.export in torch 2.x writes external-data format: <name>.onnx + <name>.onnx.data
# Both files MUST be placed together — onnxruntime auto-loads .data alongside .onnx.
onnx_path = OUT_ROOT / f"{MODEL_NAME}.onnx"
data_path = OUT_ROOT / f"{MODEL_NAME}.onnx.data"
if onnx_path.exists():
    print(f"Downloading {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
    files.download(str(onnx_path))
    if data_path.exists():
        print(f"Downloading {data_path} ({data_path.stat().st_size / 1024:.1f} KB)")
        files.download(str(data_path))
    else:
        print("Note: no .onnx.data file (weights inlined in .onnx)")
else:
    print("ERROR: .onnx not found — check training logs above")
    !ls -la {OUT_ROOT}
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Wrote {NB_PATH}  ({NB_PATH.stat().st_size} bytes)")
