# PyInstaller spec for building Maksim.app (macOS arm64).
#
# Build:  uv run pyinstaller scripts/maksim.spec --noconfirm --clean
# Output: dist/Maksim.app
# After:  codesign --force --deep --sign - dist/Maksim.app
#         open dist/Maksim.app
#
# Runtime config (API keys + wake word path):
#   The app reads ~/.maksim/.env on startup.
#   Create it before running the app:
#     mkdir -p ~/.maksim && cp .env.local ~/.maksim/.env
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

PROJECT_ROOT = Path(SPECPATH).resolve().parent

# Bundle the wake-word model files alongside the executable so the relative
# WAKE_WORD=models/maksim.onnx path resolves inside the .app.
datas = [
    (str(PROJECT_ROOT / "models" / "maksim.onnx"), "models"),
    (str(PROJECT_ROOT / "models" / "maksim.onnx.data"), "models"),
]
binaries = []
hiddenimports = []

# Heavy native packages need full collection — bins, datas, AND their own hidden
# submodules — or PyInstaller silently misses dylibs and ONNX models at runtime.
for pkg in (
    "torch", "torchaudio",            # PyTorch + audio (silero needs torch)
    "faster_whisper", "ctranslate2",  # local STT
    "openwakeword",                   # wake-word engine + bundled models
    "silero_vad",                     # VAD
    "onnxruntime",                    # used by wake + vad
    "sounddevice", "soundfile",       # audio I/O (PortAudio dylib)
    "anthropic", "groq", "elevenlabs",  # cloud SDKs
    "pydantic", "pydantic_settings",
    "scipy",                          # used in resampling
    "rumps",                          # menu-bar UI
    "AppKit", "Foundation",           # pyobjc — rumps depends on these
    "omegaconf",                      # Silero TTS imports this dynamically inside torch.hub
    "antlr4",                         # omegaconf runtime dep
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[spec] warning: collect_all({pkg!r}) failed: {e}")

# Extra hidden imports that PyInstaller analysis often misses.
hiddenimports += [
    "_cffi_backend",
    "rich",
    "websockets",
    "httpx",
    "dotenv",
    # Silero TTS imports omegaconf at runtime from inside torch.hub-loaded code,
    # which PyInstaller's static analysis can't follow.
    "omegaconf",
    "omegaconf.dictconfig",
    "omegaconf.listconfig",
    "antlr4",
    "antlr4.atn",
]


a = Analysis(
    [str(PROJECT_ROOT / "maksim_app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Cut huge unused deps if PyInstaller pulls them in.
        "tkinter", "test", "tests", "pytest", "matplotlib",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="maksim",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # no terminal window — pure GUI app
    disable_windowed_traceback=False,
    target_arch="arm64",
    # Apple Development identity. Stable Team ID (BH5W66Q8G5) means macOS TCC
    # remembers permissions (mic / Accessibility / Screen Recording) between
    # rebuilds — adhoc signing reset them every time.
    codesign_identity="Apple Development: timsonyk@gmail.com (YLUF3LQ7MY)",
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="maksim",
)

app = BUNDLE(
    coll,
    name="Maksim.app",
    icon=None,
    bundle_identifier="com.timofey.maksim",
    version="0.1.0",
    info_plist={
        "CFBundleName": "Maksim",
        "CFBundleDisplayName": "Maksim",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "12.0",
        # CRITICAL: macOS TCC will silently deny mic access without this string.
        "NSMicrophoneUsageDescription":
            "Maksim слушает микрофон чтобы распознать wake-word и команды.",
        # Screen Recording permission — used by look_at_screen tool.
        "NSScreenCaptureDescription":
            "Maksim может сделать скриншот экрана когда ты попросишь посмотреть на него.",
        # AppleEvents permission — used by run_applescript / open_app / notify.
        "NSAppleEventsUsageDescription":
            "Maksim использует AppleScript для управления приложениями.",
        # Hide Dock icon — Maksim runs in background, output via voice.
        "LSUIElement": True,
        # arm64 only.
        "LSArchitecturePriority": ["arm64"],
    },
)
