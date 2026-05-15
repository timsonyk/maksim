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

echo "→ Building bundle…"
uv run pyinstaller scripts/maksim.spec --noconfirm --clean

echo "→ Verifying signature…"
codesign -dv dist/Maksim.app 2>&1 | grep -E "Identifier|TeamIdentifier"

echo "→ Installing to /Applications/Maksim.app…"
rm -rf /Applications/Maksim.app
cp -R dist/Maksim.app /Applications/

echo "✓ Done. Launch with:  open /Applications/Maksim.app"
