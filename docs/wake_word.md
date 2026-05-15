# Тренировка кастомного русского wake-слова

Готовых русских моделей openWakeWord нет, а Picovoice Console требует company email.
Поэтому модель тренируем сами в Colab.

## Ноутбук

`scripts/train_wake_word.ipynb` — универсальный пайплайн openWakeWord для любого русского слова:

- **TTS:** Silero (русский) вместо Piper — у Piper нет ru-голосов
- **Positive samples:** 1000 целевого слова разными голосами / интонациями
- **Negative samples:** 1000 adversarial фраз (фонетически похожие + случайные команды)
- **Background:** MIT RIRs + AudioSet (опционально) + FMA
- **Features:** pre-computed ACAV100M 2000ч
- **Фиксы внутри:**
  - `torch-audiomentations==0.11.2` (избавляет от `torchaudio.set_audio_backend`)
  - in-place патч `torch_audiomentations/utils/io.py` (shim для `torchaudio.info`/`load`,
    которые удалены в torchaudio 2.9+)
  - убраны `tensorflow-cpu`/`onnx_tf` (не нужны, мы используем `.onnx` напрямую)
  - AudioSet шар в try/except — graceful skip если HF вернёт 403

## Первая тренировка (на "максим")

1. Открой `scripts/train_wake_word.ipynb` в Colab: File → Upload notebook
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all** (пропусти §0 Cleanup — это для перетренировки)
4. ~2-3 часа ожидания
5. В конце автоматом скачается `maksim.onnx`
6. Положи в `<project>/models/maksim.onnx`
7. В `.env.local`:
   ```
   WAKE_WORD=models/maksim.onnx
   WAKE_THRESHOLD=0.5
   ```
8. `uv run maksim` → говори "Максим"

## Тренировка другого слова

В Colab после первой тренировки:

1. **Run §0 Cleanup** — раскомментируй `!rm -rf my_custom_model`, запусти
2. **Отредактируй §2 Parameters:**
   - `TARGET_PHRASE` — новое слово (например `"джарвис"`)
   - `MODEL_NAME` — slug на ASCII (`"maksim"`)
   - `ADVERSARIAL_TEXTS` — **критично** замени первую часть списка на слова которые
     фонетически похожи на новое слово. Generic-часть (после комментария "Generic Russian
     phrases") можно оставить как есть
3. **Runtime → Run all** (или с §2 вниз вручную)
4. ~2-3 часа → скачается новый `<MODEL_NAME>.onnx`
5. `WAKE_WORD=models/<MODEL_NAME>.onnx` в `.env.local`

## Tuning

- Ложно срабатывает на похожие слова → подними `WAKE_THRESHOLD` до 0.6-0.7
- Плохо ловит при разной громкости/интонации → перетренируй с
  `TRAINING_STEPS=50000` и `N_SAMPLES_TRAIN=2000` (~3 часа на T4)

## Regenerate the notebook

Notebook собирается из Python-генератора:

```
uv run python scripts/build_train_notebook.py
```

Правки делай в `scripts/build_train_notebook.py`, не в .ipynb напрямую — иначе при
следующей пересборке потеряются.
