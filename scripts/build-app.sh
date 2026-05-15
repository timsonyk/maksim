#!/bin/bash
# Build Maksim.app and install to /Applications/.
#
# Usage: ./scripts/build-app.sh
#
# Notes:
#   - PyInstaller auto-signs with Apple Development cert (configured in maksim.spec)
#     so TCC remembers mic/Accessibility/Screen Recording permissions between
#     rebuilds (stable Team ID BH5W66Q8G5).
#   - Stops any running Maksim first so we don't overwrite a live bundle.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "→ Stopping any running Maksim…"
pkill -9 -f Maksim.app 2>/dev/null || true
sleep 1

# openWakeWord ships its melspectrogram / embedding / VAD models as a runtime
# download into <site-packages>/openwakeword/resources/models. Force the
# download now so PyInstaller's collect_all("openwakeword") picks them up.
# Passing a sentinel name in model_names downloads only feature + VAD models
# (skipping ~50 MB of official wake-word models we don't use).
echo "→ Ensuring openwakeword feature models are present…"
uv run python -c "from openwakeword.utils import download_models; download_models(model_names=['__features_only__'])"

echo "→ Building bundle…"
uv run pyinstaller scripts/maksim.spec --noconfirm --clean

echo "→ Verifying signature…"
codesign -dv dist/Maksim.app 2>&1 | grep -E "Identifier|TeamIdentifier"

echo "→ Installing to /Applications/Maksim.app…"
rm -rf /Applications/Maksim.app
cp -R dist/Maksim.app /Applications/

echo "✓ Done. Launch with:  open /Applications/Maksim.app"
