"""Generate macOS .icns icon for Maksim.app from a source PNG.

By default reads ``scripts/icon-source.png``. To use a different source,
pass it as an argument:

    uv run python scripts/make_icon.py /path/to/photo.png

Output: ``scripts/maksim.icns``.

The source image is center-cropped to a square (if not already), then
rendered at every macOS app-icon size. macOS rounds the corners
automatically at display time.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "icon-source.png"
ICONSET = HERE / "maksim.iconset"
ICNS = HERE / "maksim.icns"


def _square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _make_png(base: Image.Image, size: int) -> Image.Image:
    return base.resize((size, size), Image.LANCZOS)


def main() -> None:
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src_path.exists():
        print(f"ERROR: source image not found at {src_path}", file=sys.stderr)
        sys.exit(1)

    base = Image.open(src_path).convert("RGBA")
    print(f"source: {src_path}  ({base.size[0]}×{base.size[1]})")
    base = _square_crop(base)
    base = base.resize((1024, 1024), Image.LANCZOS)

    if ICONSET.exists():
        shutil.rmtree(ICONSET)
    ICONSET.mkdir(parents=True)

    # macOS .iconset wants ten specific names.
    pairs = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    for size, name in pairs:
        _make_png(base, size).save(ICONSET / name, format="PNG")
        print(f"  {name}  ({size}×{size})")

    subprocess.run(["iconutil", "-c", "icns", str(ICONSET), "-o", str(ICNS)], check=True)
    shutil.rmtree(ICONSET)
    print(f"\n✓ {ICNS}  ({ICNS.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
