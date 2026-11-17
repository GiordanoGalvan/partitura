# PyInstaller: app Windows a cartella (avvio rapido, nessuna estrazione a ogni lancio)
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hidden = [], [], []
for pkg in ("verovio", "transkun", "moduleconf", "music21", "librosa", "svglib",
            "reportlab", "soundfile", "soxr", "pretty_midi", "mir_eval", "yt_dlp",
            "audioread", "mido", "piano_transcription_inference"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
    except Exception:
        pass
hidden += collect_submodules("transkun") + collect_submodules("scipy")
binaries += [("ffmpeg.exe", ".")]

a = Analysis(["../partitura.py"], pathex=[".."], datas=datas, binaries=binaries,
             hiddenimports=hidden, excludes=["matplotlib", "IPython", "jupyter", "tkinter.test"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Partitura",
          icon="partitura.ico", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Partitura", upx=False)
