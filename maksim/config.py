from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load .env from (in order, first wins because override=False):
#   1. ~/.maksim/.env          — used when running as a packaged macOS .app
#   2. <project_root>/.env.local — dev override checked into .gitignore
#   3. <project_root>/.env       — dev defaults
_user_home_env = Path.home() / ".maksim" / ".env"
_root = Path(__file__).resolve().parent.parent
load_dotenv(_user_home_env, override=False)
load_dotenv(_root / ".env.local", override=False)
load_dotenv(_root / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # API keys
    elevenlabs_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""

    # TTS
    tts_provider: Literal["silero", "say", "elevenlabs"] = "silero"
    silero_speaker: str = "aidar"  # male: aidar, eugene; female: baya, kseniya, xenia
    say_voice: str = "Milena"
    say_rate: int = 200
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel multilingual
    elevenlabs_model_id: str = "eleven_flash_v2_5"
    elevenlabs_output_format: str = "pcm_22050"

    # STT
    stt_provider: Literal["groq", "local", "hybrid"] = "hybrid"
    stt_language: str = "ru"
    groq_stt_model: str = "whisper-large-v3-turbo"
    local_stt_model: str = "large-v3"
    local_stt_compute: str = "int8"

    # LLM
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.4

    # Wake
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.6

    # VAD
    vad_silence_ms: int = 600
    vad_min_speech_ms: int = 200
    vad_max_utterance_s: int = 15

    # Audio
    sample_rate: int = 16000
    input_device: str = ""
    output_device: str = ""

    # Behaviour
    persona: str = (
        "Ты Максим — личный голосовой ассистент Тимофея. "
        "Отвечай кратко, в одну-две фразы, на русском. "
        "Используй инструменты для управления компьютером без лишних подтверждений. "
        "Не объясняй что собираешься делать — просто делай и коротко сообщи результат. "
        "Если действие необратимое, скажи об этом одной фразой и сразу делай."
    )
    confirm_policy: Literal["none", "destructive"] = "none"
    maksim_home: str = "~/.maksim"

    @property
    def home(self) -> Path:
        p = Path(self.maksim_home).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_path(self) -> Path:
        return self.home / "log.jsonl"

    @property
    def history_path(self) -> Path:
        return self.home / "conversation.jsonl"


settings = Settings()
