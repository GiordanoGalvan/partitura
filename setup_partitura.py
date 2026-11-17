#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SETUP PARTITURA 2 — installatore automatico.

Fa tutto da solo:
  1. Trova un Python 3.10-3.13 (su Windows lo installa con winget se manca)
  2. Crea, o riusa, l'ambiente virtuale ./venv
  3. Installa PyTorch (solo CPU), il motore Transkun e il resto
  4. Installa FFmpeg (serve solo per i link YouTube)
  5. Controlla che tutto si importi e genera un PDF di prova
  6. Scrive PARTITURA.bat e avvia il programma

Si lancia con QUALSIASI Python (usa solo la libreria standard):
    python setup_partitura.py

MuseScore non serve più: il PDF lo genera il programma.
Tutto viene registrato in setup_partitura.log.
"""

import os
import re
import sys
import time
import shutil
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / "venv"
LOGFILE = HERE / "setup_partitura.log"
IS_WIN = platform.system() == "Windows"

# 3.12 è la versione con cui tutto lo stack è stato verificato per Windows.
TARGET_VERSIONS = ["3.12", "3.11", "3.13", "3.10"]

# Pacchetti indispensabili, uno per riga: se uno salta lo si vede subito.
BASE = [
    "numpy", "scipy", "soundfile", "audioread", "soxr", "librosa",
    "music21", "mido", "verovio", "svglib", "reportlab",
]
# Transkun: il motore di trascrizione. Va installato SENZA dipendenze perché
# ne dichiara una (ncls) che su Windows richiede un compilatore C e che per
# trascrivere non serve. Le dipendenze che servono davvero sono queste tre.
TRANSKUN = "transkun==2.0.1"
TRANSKUN_DIP = ["moduleconf", "pretty_midi", "mir_eval"]
# Facoltativi: motore di riserva (CRNN) e download da YouTube.
FACOLTATIVI = ["piano_transcription_inference", "yt-dlp"]

_log_handle = None


for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def log(msg="", level="·"):
    global _log_handle
    line = f"  {level} {msg}" if msg else ""
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        if _log_handle is None:
            _log_handle = open(LOGFILE, "w", encoding="utf-8")
        _log_handle.write(line + "\n")
        _log_handle.flush()
    except Exception:
        pass


def section(title):
    log()
    log("─" * 64, " ")
    log(title, " ")
    log("─" * 64, " ")


def pausa(testo="\n  Premi INVIO per chiudere..."):
    try:
        if sys.stdin and sys.stdin.isatty():
            input(testo)
    except Exception:
        pass


def run(cmd, timeout=3600, quiet=False):
    """Esegue un comando e ne registra l'esito. Non solleva mai."""
    if not quiet:
        log("$ " + " ".join(str(c) for c in cmd)[:170], " ")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        log("tempo scaduto", "!")
        return None
    except FileNotFoundError:
        return None
    if res.returncode != 0 and not quiet:
        tail = (res.stderr or res.stdout or "").strip()[-600:]
        if tail:
            log(f"esito {res.returncode}: {tail}", "!")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 1. PYTHON
# ─────────────────────────────────────────────────────────────────────────────

def python_version_of(exe) -> str:
    res = run([str(exe), "-c",
               "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
              timeout=30, quiet=True)
    return res.stdout.strip() if res and res.returncode == 0 else ""


def candidate_pythons():
    found = []
    if IS_WIN:
        res = run(["py", "-0p"], timeout=30, quiet=True)
        if res and res.returncode == 0:
            for line in res.stdout.splitlines():
                m = re.search(r"([A-Za-z]:\\[^\r\n]*?python\.exe)", line, re.I)
                if m:
                    found.append(Path(m.group(1)))
        roots = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Python",
            Path(os.environ.get("ProgramFiles", "C:/Program Files")),
            Path("C:/"),
        ]
        for root in roots:
            if not root.exists():
                continue
            try:
                for child in root.iterdir():
                    if child.is_dir() and re.search(r"python", child.name, re.I):
                        for sub in (child / "python.exe", child / "bin" / "python.exe"):
                            if sub.exists():
                                found.append(sub)
            except Exception:
                pass
    else:
        for v in TARGET_VERSIONS:
            w = shutil.which(f"python{v}")
            if w:
                found.append(Path(w))
    w = shutil.which("python3") or shutil.which("python")
    if w:
        found.append(Path(w))
    found.append(Path(sys.executable))

    seen, out = set(), []
    for p in found:
        key = str(p).lower()
        if key not in seen and p.exists():
            seen.add(key)
            out.append(p)
    return out


def pick_python():
    log("Cerco un Python adatto...")
    by_version = {}
    for p in candidate_pythons():
        v = python_version_of(p)
        if v:
            by_version.setdefault(v, p)
            log(f"trovato Python {v} — {p}")
    for want in TARGET_VERSIONS:
        if want in by_version:
            log(f"Scelgo Python {want}", "+")
            return by_version[want]
    return None


def ensure_python():
    section("PYTHON")
    py = pick_python()
    if py:
        return py
    if IS_WIN and shutil.which("winget"):
        log("Nessun Python adatto: installo il 3.12 con winget (qualche minuto)...", "+")
        run(["winget", "install", "-e", "--id", "Python.Python.3.12",
             "--accept-package-agreements", "--accept-source-agreements",
             "--disable-interactivity", "--scope", "user"], timeout=1800)
        time.sleep(3)
        py = pick_python()
        if py:
            return py
    log("Non trovo e non riesco a installare un Python 3.10-3.13.", "!")
    log("Scarica Python 3.12 da python.org, spunta «Add to PATH» e rilancia.", "!")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. AMBIENTE VIRTUALE E PACCHETTI
# ─────────────────────────────────────────────────────────────────────────────

def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def ensure_venv(base_python: Path):
    section("AMBIENTE VIRTUALE")
    vp = venv_python()
    if vp.exists():
        v = python_version_of(vp)
        if v in TARGET_VERSIONS:
            log(f"Riuso l'ambiente esistente (Python {v}): aggiungo solo ciò che manca", "+")
            return vp
        log(f"L'ambiente esistente è su Python {v or '?'}: lo rifaccio.", "!")
        shutil.rmtree(VENV, ignore_errors=True)
    log(f"Creo l'ambiente in {VENV}")
    run([str(base_python), "-m", "venv", str(VENV)], timeout=600)
    if not vp.exists():
        log("Creazione dell'ambiente virtuale fallita.", "!")
        return None
    log("Ambiente creato", "+")
    return vp


def pip_install(vp: Path, args, label) -> bool:
    log(f"Installo {label}...")
    res = run([str(vp), "-m", "pip", "install", "--no-input",
               "--disable-pip-version-check"] + args, timeout=5400)
    ok = res is not None and res.returncode == 0
    log(f"{label}: {'fatto' if ok else 'FALLITO'}", "+" if ok else "!")
    return ok


def ensure_packages(vp: Path):
    """Ritorna (falliti_indispensabili, falliti_facoltativi)."""
    section("PACCHETTI")
    run([str(vp), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        timeout=900, quiet=True)
    falliti, facolt = [], []

    # PyTorch solo CPU: l'indice normale su Windows scarica 2,5 GB di librerie
    # per schede grafiche che qui non servono.
    cpu = ["--index-url", "https://download.pytorch.org/whl/cpu"]
    if not pip_install(vp, ["torch", "torchaudio"] + cpu, "PyTorch (CPU)"):
        log("Ritento dall'indice standard...", "!")
        if not pip_install(vp, ["torch", "torchaudio"], "PyTorch (indice standard)"):
            falliti.append("torch")

    for pkg in BASE:
        if not pip_install(vp, [pkg], pkg):
            falliti.append(pkg)

    ok_tk = pip_install(vp, ["--no-deps", TRANSKUN], "Transkun (motore di trascrizione)")
    for pkg in TRANSKUN_DIP:
        ok_tk = pip_install(vp, [pkg], pkg) and ok_tk
    if not ok_tk:
        facolt.append("transkun")

    for pkg in FACOLTATIVI:
        if not pip_install(vp, [pkg], pkg):
            facolt.append(pkg)
    return falliti, facolt


# ─────────────────────────────────────────────────────────────────────────────
# 3. FFMPEG
# ─────────────────────────────────────────────────────────────────────────────

def ensure_ffmpeg():
    section("FFMPEG (per i link YouTube)")
    if shutil.which("ffmpeg"):
        log("FFmpeg già presente", "+")
        return True
    if IS_WIN and shutil.which("winget"):
        log("Installo FFmpeg con winget...")
        run(["winget", "install", "-e", "--id", "Gyan.FFmpeg",
             "--accept-package-agreements", "--accept-source-agreements",
             "--disable-interactivity"], timeout=1800)
        log("FFmpeg installato: sarà visibile alla prossima apertura del programma.", "+")
        return True
    log("FFmpeg non installato: i link YouTube non funzioneranno.", "!")
    log("File audio e MIDI sul disco funzionano comunque.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFICA
# ─────────────────────────────────────────────────────────────────────────────

def verify(vp: Path):
    """Ritorna (indispensabili_ok, motori_disponibili)."""
    section("VERIFICA")
    code = (
        "import importlib\n"
        "base=['numpy','scipy','soundfile','librosa','music21','mido','verovio',"
        "'svglib','reportlab','torch']\n"
        "bad=[]\n"
        "for m in base:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: bad.append(m+' ('+type(e).__name__+')')\n"
        "mot=[]\n"
        "try:\n"
        "    import transkun, moduleconf, pretty_midi, mir_eval, torchaudio; mot.append('transkun')\n"
        "except Exception as e: print('NOTA transkun: '+type(e).__name__+': '+str(e)[:120])\n"
        "try:\n"
        "    import piano_transcription_inference; mot.append('crnn')\n"
        "except Exception as e: print('NOTA crnn: '+type(e).__name__+': '+str(e)[:120])\n"
        "print('BASE '+('OK' if not bad else 'MANCA: '+', '.join(bad)))\n"
        "print('MOTORI '+','.join(mot))\n"
    )
    res = run([str(vp), "-c", code], timeout=600, quiet=True)
    out = res.stdout if res else ""
    base_ok, motori = False, []
    for line in out.splitlines():
        if line.startswith("NOTA"):
            log(line[5:], "!")
        elif line.startswith("BASE"):
            base_ok = line.strip() == "BASE OK"
            log("Librerie di base: " + line[5:], "+" if base_ok else "!")
        elif line.startswith("MOTORI"):
            motori = [m for m in line[7:].strip().split(",") if m]
    if motori:
        log("Motori di trascrizione disponibili: " + ", ".join(motori), "+")
    else:
        log("Nessun motore di trascrizione dall'audio: funzioneranno solo i file MIDI.", "!")
    if res is None or (res.returncode != 0 and not out):
        log((res.stderr[-400:] if res else "nessuna risposta"), "!")
    return base_ok, motori


def prova_a_secco(vp: Path) -> bool:
    """Genera un PDF da un brano di prova: collauda notazione e impaginazione."""
    section("PROVA A SECCO")
    banco = HERE / "banco_prova.py"
    if not banco.exists():
        log("banco_prova.py non c'è: salto la prova.", "·")
        return True
    log("Scrivo e impagino un brano di prova (un minuto circa)...")
    res = run([str(vp), str(banco), "--rapido"], timeout=900, quiet=True)
    ok = res is not None and res.returncode == 0 and "PDF generati .................. 1 su 1" in res.stdout
    if ok:
        log("La catena note → partitura → PDF funziona su questo computer.", "+")
    else:
        log("La prova non è riuscita: guarda setup_partitura.log.", "!")
        if res:
            for line in (res.stdout + res.stderr).strip().splitlines()[-12:]:
                log(line.strip(), "!")
    return ok


def ensure_checkpoint_crnn(vp: Path, motori):
    """Solo se Transkun manca: scarica il modello del motore di riserva."""
    if "transkun" in motori or "crnn" not in motori:
        return
    section("MODELLO CRNN (motore di riserva)")
    code = ("import sys; sys.path.insert(0, r'%s'); import partitura; "
            "partitura.ensure_checkpoint()" % str(HERE))
    res = run([str(vp), "-c", code], timeout=3600, quiet=True)
    ok = res is not None and res.returncode == 0
    log("Modello pronto" if ok else "Download non riuscito: verrà ritentato all'uso.",
        "+" if ok else "!")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LANCIATORE
# ─────────────────────────────────────────────────────────────────────────────

def write_launcher(vp: Path) -> Path:
    section("LANCIATORE")
    if IS_WIN:
        path = HERE / "PARTITURA.bat"
        path.write_text(
            "@echo off\r\n"
            "title PARTITURA\r\n"
            "set PYTHONUTF8=1\r\n"
            f'cd /d "{HERE}"\r\n'
            f'"{vp}" "{HERE / "partitura.py"}" %*\r\n'
            "if errorlevel 1 pause\r\n",
            encoding="utf-8")
    else:
        path = HERE / "partitura.sh"
        path.write_text(f'#!/bin/bash\ncd "{HERE}"\n"{vp}" "{HERE / "partitura.py"}" "$@"\n',
                        encoding="utf-8")
        os.chmod(path, 0o755)
    log(f"Creato {path.name}: da qui in poi lanci solo questo.", "+")
    for vecchio in ("RIPARA.bat", "ripara.py", "SCARICA_MODELLO.bat", "scarica_modello.py"):
        if (HERE / vecchio).exists():
            log(f"{vecchio} non serve più: puoi cancellarlo.")
    return path


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n" + "=" * 64)
    print("  SETUP PARTITURA 2 — installazione automatica")
    print("=" * 64)
    log(f"Sistema: {platform.system()} {platform.release()}")
    log(f"Cartella: {HERE}")

    if not (HERE / "partitura.py").exists():
        log("partitura.py non è in questa cartella. Mettilo qui e rilancia.", "!")
        pausa()
        return 1
    try:
        base = ensure_python()
        if not base:
            pausa()
            return 1
        vp = ensure_venv(base)
        if not vp:
            pausa()
            return 1
        falliti, facolt = ensure_packages(vp)
        ensure_ffmpeg()
        base_ok, motori = verify(vp)
        secco_ok = prova_a_secco(vp) if base_ok else False
        ensure_checkpoint_crnn(vp, motori)
        launcher = write_launcher(vp)

        section("RISULTATO")
        tutto = base_ok and secco_ok and bool(motori)
        if tutto:
            log("Installazione completata e collaudata.", "+")
        else:
            log("Installazione completata con problemi:", "!")
            if falliti:
                log("pacchetti indispensabili falliti: " + ", ".join(falliti), "!")
            if not motori:
                log("nessun motore di trascrizione dall'audio", "!")
            if base_ok and not secco_ok:
                log("la prova a secco non è riuscita", "!")
            log(f"Mandami {LOGFILE.name}: c'è scritto tutto.", "!")
        if facolt:
            log("Facoltativi non installati: " + ", ".join(facolt)
                + " (il programma funziona lo stesso)")
        log()
        log(f"Da adesso lancia: {launcher.name}", "+")
        if base_ok and "--no-avvio" not in sys.argv:
            log("Avvio il programma...", "+")
            try:
                subprocess.Popen([str(vp), str(HERE / "partitura.py")], cwd=str(HERE))
                time.sleep(2)
            except Exception as e:
                log(f"Avvialo a mano ({e})", "!")
        pausa("\n  Premi INVIO per chiudere questa finestra...")
        return 0 if tutto else 2
    except Exception as e:
        import traceback
        log(f"ERRORE: {e}", "!")
        log(traceback.format_exc(), "!")
        pausa()
        return 1


if __name__ == "__main__":
    sys.exit(main())
