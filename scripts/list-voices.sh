#!/bin/zsh
# List voices available on your ElevenLabs account (filter by language).
# Usage:  ./scripts/list-voices.sh          → all voices
#         ./scripts/list-voices.sh ru       → voices that declare Russian
set -e
cd "$(dirname "$0")/.."

if [ -f .env.local ]; then
    source .env.local
elif [ -f .env ]; then
    source .env
fi

if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "ELEVENLABS_API_KEY not set"; exit 1
fi

FILTER="${1:-}"
exec uv run python - <<PY
import os, httpx
key = os.environ["ELEVENLABS_API_KEY"]
filt = "$FILTER".lower()
r = httpx.get(
    "https://api.elevenlabs.io/v2/voices",
    headers={"xi-api-key": key},
    params={"page_size": 100},
    timeout=20,
)
r.raise_for_status()
voices = r.json().get("voices", [])
hits = 0
for v in voices:
    langs = v.get("verified_languages") or []
    lang_codes = [l.get("language", "").lower() for l in langs]
    if filt and filt not in " ".join(lang_codes):
        continue
    hits += 1
    print(f"{v['voice_id']}  {v['name']:<22}  langs={','.join(lang_codes) or '?'}")
print(f"\n{hits} voices shown.")
PY
