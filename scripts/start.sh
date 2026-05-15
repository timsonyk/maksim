#!/bin/zsh
# Foreground run. Used directly and as the LaunchAgent target.
cd "$(dirname "$0")/.."
exec /opt/homebrew/bin/uv run maksim
