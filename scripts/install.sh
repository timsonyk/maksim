#!/bin/zsh
set -e
cd "$(dirname "$0")/.."

echo "→ Installing portaudio (required by sounddevice)…"
if ! brew list portaudio &>/dev/null; then
    brew install portaudio
else
    echo "  already installed"
fi

echo "→ Resolving Python deps with uv (this is what makes .venv)…"
uv sync

echo "→ Creating ~/.maksim runtime dir…"
mkdir -p "$HOME/.maksim"

if [ ! -f .env.local ]; then
    echo "→ Creating .env.local from template — fill in API keys."
    cp .env.example .env.local
fi

echo ""
echo "✅ Install done."
echo ""
echo "Next:"
echo "  1. Put API keys into .env.local:    open -e .env.local"
echo "  2. Run once in foreground:          ./scripts/start.sh"
echo "  3. Install autostart at login:      ./scripts/install-daemon.sh"
echo ""
echo "First run will:"
echo "  - download the openWakeWord 'hey_jarvis' model (~10 MB)"
echo "  - download Silero VAD ONNX (~2 MB)"
echo "  - if STT_PROVIDER includes local: download faster-whisper large-v3 (~3 GB)"
