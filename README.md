# Maksim

Голосовой ассистент для macOS на русском языке. Always-on (живёт в menu bar), полный системный доступ через инструменты Claude, кастомное русское wake-word, screen vision, кликает по экрану.

```
mic ─► [openWakeWord русское имя]                 ─┐
       [STT-канал (Whisper) ищет слово в речи]    ─┴► wake
                                                       │
                                                       ▼
                                            [Silero VAD endpoint]
                                                       │
                                                       ▼
                                  [Groq Whisper-turbo / faster-whisper fallback]
                                                       │
                                                       ▼
                          [Claude Sonnet 4.6 + tools: shell, AppleScript, screen, click...]
                                                       │
                                                       ▼
                                  [Silero TTS (local) / ElevenLabs / macOS say]
                                                       │
                                                       ▼
                                                  speakers
```

После ответа Maksim **продолжает слушать** — можно давать команды без повторного wake. Скажи *"конец связи"* → выйдет обратно в ожидание wake. Можно **перебить** его словом "Максим" пока он говорит.

---

## Установка

Требуется: macOS arm64 (Apple Silicon), Python 3.11+, [uv](https://github.com/astral-sh/uv).

```bash
# 1. Зависимости
brew install portaudio uv
git clone https://github.com/<твой-логин>/maksim.git
cd maksim
uv sync

# 2. Конфиг
cp .env.example .env.local
open -e .env.local   # вставь свои ключи (см. ниже)
```

### Какие ключи нужны

| Ключ | Откуда | Зачем |
|---|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com | LLM-мозг (обязательно) |
| `GROQ_API_KEY` | https://console.groq.com (free tier) | быстрая cloud STT (опционально — без него работает local) |
| `ELEVENLABS_API_KEY` | https://elevenlabs.io | только если `TTS_PROVIDER=elevenlabs` (по умолчанию Silero local) |

## Запуск

### Foreground (для отладки)

```bash
uv run maksim
```

Скажи **"Hey Jarvis"** (или своё слово) → услышишь короткий бип → говори фразу по-русски.

### Как .app в menu bar

```bash
./scripts/build-app.sh
```

Соберёт `Maksim.app` и положит в `/Applications/`. Запусти через Spotlight (Cmd+Space → "Maksim"). В правом верхнем углу появится иконка статуса (🎙 / 👂 / ⚡ / 💭 / 🗣 / 💬).

При первом запуске macOS попросит **Microphone**, **Screen Recording** и **Accessibility** — разреши.

Чтобы запускался при логине: System Settings → General → Login Items → **+** → выбрать Maksim.app.

## Wake-word

По умолчанию стоит встроенное `hey_jarvis` (английское "Хей Джарвис"). Чтобы Maksim откликался на своё имя на русском, тренируем кастомную модель openWakeWord:

1. Открой `scripts/train_wake_word.ipynb` в [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. В §2 укажи `TARGET_PHRASE`, `MODEL_NAME` и `ADVERSARIAL_TEXTS` (фразы что НЕ должны триггерить)
4. Runtime → Run all → ~2-3 часа
5. Скачается `<MODEL_NAME>.onnx` и `<MODEL_NAME>.onnx.data` — положи оба в `models/`
6. В `.env.local`:
   ```
   WAKE_WORD=models/<MODEL_NAME>.onnx
   WAKE_THRESHOLD=0.7    # 0.5 — чувствительный, 0.85 — строгий
   ```

Подробнее: `docs/wake_word.md`.

Параллельно с onnx работает **STT-канал**: faster-whisper транскрибирует речь и реагирует, если в первых 3 словах есть твоё имя. Это страховка от пропуска wake — даже если onnx промахнётся, STT поймает.

## Smoke-тесты

```bash
uv run python -m tests.smoke_audio   # запись/воспроизведение
uv run python -m tests.smoke_wake    # детекция wake word
uv run python -m tests.smoke_vad     # wake + захват утерренса в WAV
uv run python -m tests.smoke_stt     # wake + транскрипция
uv run python -m tests.smoke_tts     # синтез русской фразы
uv run python -m tests.smoke_llm     # текстовый чат с инструментами (без аудио)
```

## Кастомизация

| Хочешь | Куда |
|---|---|
| Сменить голос TTS | `TTS_PROVIDER` + `SILERO_SPEAKER` / `ELEVENLABS_VOICE_ID` / `SAY_VOICE` |
| Своё имя для wake | Тренируй модель (см. выше) → `WAKE_WORD=models/...onnx` |
| Уменьшить ложные срабатывания | `WAKE_THRESHOLD=0.85` |
| Быстрее реагирует на паузу | `VAD_SILENCE_MS=400` |
| Дешевле LLM | `LLM_MODEL=claude-haiku-4-5-20251001` |
| Только local STT (без интернета) | `STT_PROVIDER=local` + `LOCAL_STT_MODEL=small` |
| Голосовое "да" перед `rm`/`sudo` | `CONFIRM_POLICY=destructive` |
| Своя персона | `PERSONA="..."` |

## Инструменты, доступные Maksim

| Tool | Что делает |
|---|---|
| `run_shell` | zsh-команда (полный доступ) |
| `run_applescript` | osascript для управления Mac |
| `open_app` / `open_url` | запустить приложение, открыть URL |
| `read_file` / `write_file` | файловая система |
| `list_dir` / `search_files` | навигация (Spotlight `mdfind`) |
| `get_clipboard` / `set_clipboard` | буфер обмена |
| `notify` | macOS-уведомление |
| `get_active_app` | какое приложение в фокусе |
| `look_at_screen` | сделать скриншот, Maksim его видит |
| `click_at` | кликнуть по координатам (по виденному) |
| `type_text` / `press_key` / `scroll` | клавиатура и скролл |

Все вызовы пишутся в `~/.maksim/log.jsonl` с timestamp и параметрами.

## Безопасность

- API-ключи в `.env.local` — `gitignored`, никогда не попадают в коммит.
- `CONFIRM_POLICY=destructive` включает голосовое подтверждение перед `rm -rf`, `sudo`, `dd if=`, `mkfs`, `shutdown`, `reboot`, `diskutil erase`, `> /dev/`.
- Лог `~/.maksim/log.jsonl` хранит каждый wake, каждую команду и каждый tool-call. Можно ревьюить что Maksim делал.
- Бандл подписан Apple Development cert — macOS TCC помнит permissions между ребилдами.

## Latency

| Стадия | Цель | Примечание |
|---|---|---|
| Wake detect | <100 мс | openWakeWord на CPU |
| VAD silence trigger | 600 мс | `VAD_SILENCE_MS` |
| STT (Groq) | 150-300 мс | для фразы <5 с |
| LLM first token | 200-400 мс | Claude Sonnet streaming |
| TTS first audio | 200-500 мс | Silero на M2 |
| **End-to-end** | **~1.5-3 с** | от конца фразы до начала ответа |

Замеры в `~/.maksim/log.jsonl` — `wake.fire / vad.captured / stt.text / llm.first_sentence / turn.done`.

## Troubleshooting

**Microphone permission** → System Settings → Privacy & Security → Microphone → Maksim ✓

**Wake не срабатывает** → опусти `WAKE_THRESHOLD=0.5`, или перетренируй модель с `TRAINING_STEPS=50000`.

**Wake срабатывает на шум** → подними `WAKE_THRESHOLD=0.85`.

**Whisper галлюцинирует** ("Продолжение следует...") → используй `LOCAL_STT_MODEL=medium` (но медленнее) или `STT_PROVIDER=groq` (только облако).

**ElevenLabs выдаёт 302 redirect** → они гео-блокируют ряд стран. Либо VPN, либо переключись на `TTS_PROVIDER=silero`.

**.app просит permissions заново после каждого ребилда** → нужна стабильная code-signing identity. Spec уже настроен на Apple Development cert (`scripts/maksim.spec`, `codesign_identity=...`); подставь свою, если нет.

## Лицензия

MIT. Делайте что хотите, авторство сохраняйте.
