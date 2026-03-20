# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Desktop Agent
Build: pyinstaller main.spec
"""

import sys
from pathlib import Path
import customtkinter

CTK_PATH = Path(customtkinter.__file__).parent

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        # CustomTkinter themes, images, fonts
        (str(CTK_PATH), "customtkinter"),
    ],
    hiddenimports=[
        # sqlite-vec native extension loader
        "sqlite_vec",
        # Ollama client internals
        "ollama",
        "httpx",
        "httpcore",
        # Pillow image formats used by show_screenshot
        "PIL._tkinter_finder",
        "PIL.ImageTk",
        "PIL.Image",
        # Optional deps — wrapped in try/except in code so safe to list
        "paramiko",
        "pdfplumber",
        "pytesseract",
        "cv2",
        "playwright.sync_api",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "numpy", "scipy", "pandas",
        "IPython", "notebook", "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir mode — faster startup
    name="DesktopAgent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # no terminal window
    disable_windowed_traceback=False,
    icon=None,               # set to "icon.ico" if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DesktopAgent",
)
