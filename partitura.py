#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PARTITURA 2 — da audio di pianoforte a partitura PDF.

Riscrittura completa del nucleo di notazione. Cosa cambia rispetto alla 1.x:

  * La griglia dei movimenti si ricava dalle NOTE trascritte, non dall'audio:
    su un pianoforte gli attacchi sono l'informazione ritmica più pulita.
  * La quantizzazione decide una suddivisione PER MOVIMENTO (binaria o
    terzinata) e non nota per nota: le durate risultano scrivibili per
    costruzione, niente più «inexpressible durations».
  * Le misure si costruiscono a mano, con legature di valore, pause e
    terzine esplicite: nessun passaggio alla cieca dentro makeNotation.
  * Abbellimenti riconosciuti prima della griglia: acciaccature (anche
    multiple), trilli, mordenti superiori e inferiori, gruppetti, arpeggiati.
  * In più: tonalità con alterazioni coerenti, dinamiche (pp…ff) dalle
    velocità, staccati, segni di pedale, anacrusi.
  * Il PDF si genera da solo (Verovio): MuseScore diventa facoltativo.
  * In ingresso anche un file MIDI, oltre ad audio e link YouTube.
  * In uscita anche il MIDI della partitura, per riascoltarla.

Uso:
    python partitura.py                      interfaccia grafica
    python partitura.py brano.mp3 -t Titolo  riga di comando
    python partitura.py --help
"""

import os
import re
import sys
import json
import math
import shutil
import tempfile
import argparse
import threading
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import List, Optional, Tuple, Dict

import numpy as np

# Su Windows la console e i tubi fra processi usano cp1252: senza questo, un
# carattere come «─» o «→» fa fallire la stampa.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

VERSIONE = "3.0"

# App compilata (installatore Windows): niente interprete Python separato, e
# FFmpeg viaggia nella stessa cartella dell'eseguibile.
CONGELATO = bool(getattr(sys, "frozen", False))
if CONGELATO:
    _cartella_app = os.path.dirname(sys.executable)
    os.environ["PATH"] = _cartella_app + os.pathsep + os.environ.get("PATH", "")

SR = 16000                  # frequenza di campionamento per il modello CRNN
HOP = 512

# ─────────────────────────────────────────────────────────────────────────────
# SOGLIE (secondi). Tutte verificate sul banco di prova sintetico.
# ─────────────────────────────────────────────────────────────────────────────

CHORD_SPREAD   = 0.035      # note entro 35 ms = stesso attacco (accordo)
ARP_MAX_STEP   = 0.090      # passo massimo fra le note di un arpeggiato
ARP_MAX_SPAN   = 0.400      # durata massima dell'intero arpeggiato
GRACE_MAX_GAP  = 0.072      # distanza massima acciaccatura → nota reale (oltre, è una biscroma scritta)
ORN_MAX_IOI    = 0.135      # passo massimo dentro trilli / mordenti / gruppetti
TRILL_MIN      = 4          # note minime perché sia un trillo
ISOLATION      = 2.5        # quante volte più lunga deve essere la nota reale

# Profili di quantizzazione: penalità per suddivisione del movimento.
# «lam» pesa la complessità contro la fedeltà all'esecuzione.
QUANT = {
    #            lam     penalità per divisione (tempi semplici)
    "lieve": dict(lam=0.014, pen={1: 0, 2: 1, 4: 2, 3: 2.6, 8: 3.2, 6: 3.6}),
    "media": dict(lam=0.030, pen={1: 0, 2: 1, 4: 2, 3: 2.8, 8: 5.5, 6: 6.0}),
    "forte": dict(lam=0.060, pen={1: 0, 2: 1, 4: 2.2, 3: 4.0, 8: 8.0, 6: 8.0}),
}
# tempi composti (movimento = semiminima puntata): 3 = crome, 6 = semicrome
QUANT_COMP = {
    "lieve": {1: 0, 3: 1, 6: 2, 12: 3.4},
    "media": {1: 0, 3: 1, 6: 2.2, 12: 6.0},
    "forte": {1: 0, 3: 1, 6: 3.5, 12: 9.0},
}
PESO_EQUILIBRIO = 2.5      # quanto conta che le due mani si dividano le note
COLLISION = 0.45            # costo di due attacchi distinti finiti sullo stesso punto

TEMPI = ["4/4", "3/4", "2/4", "6/8", "9/8", "12/8", "5/4"]


STAMPA = True               # il banco di prova la spegne


def log(msg: str, cb=None):
    line = f"  {msg}"
    if STAMPA:
        try:
            print(line, flush=True)
        except Exception:
            pass
    if cb:
        cb(line)


# ─────────────────────────────────────────────────────────────────────────────
# DATI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Nota:
    onset: float
    offset: float
    pitch: int
    vel: int = 80
    hand: int = 0                    # 0 destra, 1 sinistra
    verified: bool = True            # ha riscontro nell'audio originale
    orn: Optional[str] = None        # trill | mordent | inv_mordent | turn
    arp: int = 0                     # id del gruppo arpeggiato (0 = nessuno)
    graces: List[int] = field(default_factory=list)
    raw_onset: float = 0.0           # attacco originale, prima degli allineamenti
    pos: float = 0.0                 # posizione in movimenti (continua)
    end_pos: float = 0.0
    q: F = F(0)                      # attacco quantizzato, in quarti dalla battuta 1


@dataclass
class Metrica:
    beats: np.ndarray                # istanti dei movimenti (secondi)
    bpm: float
    num: int
    den: int
    beat_ql: F                       # durata del movimento in quarti (1 o 3/2)
    per_bar: int                     # movimenti per battuta
    origin: int                      # indice del movimento dove inizia la battuta 1

    @property
    def sig(self) -> str:
        return f"{self.num}/{self.den}"

    @property
    def bar_ql(self) -> F:
        return self.beat_ql * self.per_bar

    def to_pos(self, t):
        """secondi → movimenti (interpolando la griglia reale: assorbe il rubato)."""
        idx = np.arange(len(self.beats), dtype=float)
        b = self.beats
        t = np.asarray(t, dtype=float)
        out = np.interp(t, b, idx)
        lo = t < b[0]
        hi = t > b[-1]
        if np.any(lo):
            out = np.where(lo, (t - b[0]) / (b[1] - b[0]), out)
        if np.any(hi):
            out = np.where(hi, idx[-1] + (t - b[-1]) / (b[-1] - b[-2]), out)
        return out - self.origin


# ─────────────────────────────────────────────────────────────────────────────
# 1. INGRESSO: audio, YouTube, MIDI
# ─────────────────────────────────────────────────────────────────────────────

YT_RE = re.compile(r"(youtube\.com|youtu\.be)", re.I)


def is_youtube(s: str) -> bool:
    return bool(YT_RE.search(s or ""))


def is_midi(path: str) -> bool:
    return path.lower().endswith((".mid", ".midi"))


def _ytdlp_cmd() -> List[str]:
    probe = subprocess.run([sys.executable, "-c", "import yt_dlp"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        return [exe]
    raise RuntimeError("yt-dlp non disponibile. Installalo con:\n"
                       f"  {sys.executable} -m pip install yt-dlp")


def _fetch_youtube_interno(url: str, workdir: str, cb=None) -> Tuple[str, Optional[str]]:
    """Nell'app compilata yt-dlp si usa come libreria, nello stesso processo."""
    import yt_dlp
    log("Scarico l'audio da YouTube...", cb)
    opz = {"format": "bestaudio/best", "noplaylist": True, "quiet": True,
           "outtmpl": os.path.join(workdir, "yt_audio.%(ext)s"),
           "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}]}
    with yt_dlp.YoutubeDL(opz) as ydl:
        info = ydl.extract_info(url, download=True)
    cands = sorted(f for f in os.listdir(workdir) if f.startswith("yt_audio"))
    if not cands:
        raise RuntimeError("Download completato ma file non trovato.")
    path = os.path.join(workdir, next((c for c in cands if c.endswith(".mp3")), cands[0]))
    title = (info or {}).get("title")
    log(f"Scaricato: {title or os.path.basename(path)}", cb)
    return path, title


def fetch_youtube(url: str, workdir: str, cb=None) -> Tuple[str, Optional[str]]:
    if CONGELATO:
        return _fetch_youtube_interno(url, workdir, cb)
    log("Scarico l'audio da YouTube...", cb)
    out_tmpl = os.path.join(workdir, "yt_audio.%(ext)s")
    cmd = _ytdlp_cmd() + ["-x", "--audio-format", "mp3", "--audio-quality", "0",
                          "--no-playlist", "--print-json", "--no-simulate",
                          "-o", out_tmpl, url]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp fallito:\n{res.stderr[-800:]}")
    title = None
    for line in res.stdout.splitlines():
        if line.strip().startswith("{"):
            try:
                title = json.loads(line).get("title")
            except Exception:
                pass
    cands = sorted(f for f in os.listdir(workdir) if f.startswith("yt_audio"))
    if not cands:
        raise RuntimeError("Download completato ma file non trovato.")
    path = os.path.join(workdir, "yt_audio.mp3")
    if not os.path.exists(path):
        path = os.path.join(workdir, cands[0])
    log(f"Scaricato: {title or os.path.basename(path)}", cb)
    return path, title


def load_audio(path: str, cb=None) -> np.ndarray:
    import librosa
    log("Carico l'audio...", cb)
    y, _ = librosa.load(path, sr=SR, mono=True)
    log(f"Durata: {len(y) / SR:.1f} s", cb)
    return y


def leggi_midi(path: str, cb=None) -> Tuple[List[Nota], List[Tuple[float, float]]]:
    """Legge note e pedale di risonanza da un file MIDI."""
    import mido
    mid = mido.MidiFile(path)
    t = 0.0
    attive: Dict[Tuple[int, int], Tuple[float, int]] = {}
    note: List[Nota] = []
    pedali: List[Tuple[float, float]] = []
    ped_on: Optional[float] = None
    for msg in mid:                       # i tempi arrivano già in secondi
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            k = (getattr(msg, "channel", 0), msg.note)
            if k in attive:               # riattacco senza note_off: chiudi la precedente
                t0, v0 = attive.pop(k)
                note.append(Nota(t0, max(t, t0 + 0.02), msg.note, v0))
            attive[k] = (t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            k = (getattr(msg, "channel", 0), msg.note)
            if k in attive:
                t0, v0 = attive.pop(k)
                note.append(Nota(t0, max(t, t0 + 0.02), msg.note, v0))
        elif msg.type == "control_change" and msg.control == 64:
            if msg.value >= 64 and ped_on is None:
                ped_on = t
            elif msg.value < 64 and ped_on is not None:
                pedali.append((ped_on, t))
                ped_on = None
    for (ch, p), (t0, v0) in attive.items():
        note.append(Nota(t0, max(t, t0 + 0.05), p, v0))
    if ped_on is not None:
        pedali.append((ped_on, t))
    note.sort(key=lambda n: (n.onset, n.pitch))
    log(f"MIDI: {len(note)} note, {len(pedali)} pedalate", cb)
    return note, pedali


def scrivi_midi(note: List[Nota], pedali, path: str):
    """Salva l'esecuzione trascritta (tempi reali) come MIDI."""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tempo = 500000                                    # 120 bpm → 960 tick/s
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    ev = []
    for n in note:
        ev.append((n.raw_onset or n.onset, 1, "note_on", n.pitch, max(1, min(127, n.vel))))
        ev.append((n.offset, 0, "note_off", n.pitch, 0))
    for a, b in pedali:
        ev.append((a, 2, "cc", 64, 127))
        ev.append((b, 0, "cc", 64, 0))
    ev.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for t, _, kind, a, b in ev:
        tick = int(round(max(t, 0) * 960))
        dt, last = max(tick - last, 0), max(tick, last)
        if kind == "cc":
            tr.append(mido.Message("control_change", control=a, value=b, time=dt))
        else:
            tr.append(mido.Message(kind, note=a, velocity=b, time=dt))
    mid.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRASCRIZIONE — due motori
# ─────────────────────────────────────────────────────────────────────────────

CKPT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
CKPT_URL = ("https://zenodo.org/record/4034264/files/"
            "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1")
CKPT_MIN_SIZE = 160_000_000


def checkpoint_path() -> str:
    from pathlib import Path
    return str(Path.home() / "piano_transcription_inference_data" / CKPT_NAME)


def ensure_checkpoint(cb=None) -> str:
    """
    Il pacchetto CRNN scarica il modello con os.system('wget'), che su Windows
    non esiste. Lo scarichiamo noi nello stesso percorso.
    """
    import urllib.request
    path = checkpoint_path()
    if os.path.exists(path) and os.path.getsize(path) >= CKPT_MIN_SIZE:
        return path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log("Scarico il modello CRNN (165 MB, solo la prima volta)...", cb)
    tmp = path + ".part"
    last = [-1]

    def progress(blocks, bsize, total):
        if total > 0:
            pct = min(int(blocks * bsize * 100 / total), 100)
            if pct >= last[0] + 10:
                last[0] = pct
                log(f"  {pct}%", cb)

    try:
        op = urllib.request.build_opener()
        op.addheaders = [("User-Agent", "partitura/2.0")]
        urllib.request.install_opener(op)
        urllib.request.urlretrieve(CKPT_URL, tmp, reporthook=progress)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Download del modello fallito: {e}\n"
                           f"Scaricalo da https://zenodo.org/record/4034264 e salvalo come:\n  {path}")
    if os.path.getsize(tmp) < CKPT_MIN_SIZE:
        os.remove(tmp)
        raise RuntimeError("Download del modello interrotto: rilancia.")
    os.replace(tmp, path)
    return path


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def trascrivi_crnn(audio_path: str, cb=None):
    """Motore CRNN ByteDance (note + pedale)."""
    from piano_transcription_inference import PianoTranscription
    y = load_audio(audio_path, cb)
    ckpt = ensure_checkpoint(cb)
    dev = _device()
    log(f"Trascrizione CRNN su {dev} (qualche minuto)...", cb)
    tr = PianoTranscription(device=dev, checkpoint_path=ckpt)
    out = tr.transcribe(y, None)
    note = [Nota(float(e["onset_time"]), float(e["offset_time"]),
                 int(e["midi_note"]), int(e.get("velocity", 80)))
            for e in out["est_note_events"]]
    pedali = [(float(p["onset_time"]), float(p["offset_time"]))
              for p in out.get("est_pedal_events", [])]
    note.sort(key=lambda n: (n.onset, n.pitch))
    return note, pedali


def transkun_disponibile() -> bool:
    import importlib.util
    try:
        return (importlib.util.find_spec("transkun") is not None
                and importlib.util.find_spec("moduleconf") is not None)
    except Exception:
        return False


def trascrivi_transkun(audio_path: str, workdir: str, cb=None):
    """
    Motore Transkun (Neural Semi-CRF): più preciso del CRNN soprattutto sulle
    durate. I pesi sono dentro il pacchetto pip, quindi nessun download.
    Non si passa dal comando «transkun» perché quello dipende da pkg_resources
    e da pydub: qui si carica il modello direttamente.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import librosa
    import moduleconf
    import transkun
    from transkun.Data import writeMidi

    base = os.path.join(os.path.dirname(transkun.__file__), "pretrained")
    conf_path = os.path.join(base, "2.0.conf")
    w_path = os.path.join(base, "2.0.pt")
    dev = _device()
    log(f"Trascrizione Transkun su {dev} (qualche minuto)...", cb)

    cm = moduleconf.parseFromFile(conf_path)
    TransKun = cm["Model"].module.TransKun
    conf = cm["Model"].config
    try:
        ck = torch.load(w_path, map_location=dev, weights_only=False)
    except TypeError:                     # torch vecchi non hanno weights_only
        ck = torch.load(w_path, map_location=dev)
    model = TransKun(conf=conf).to(dev)
    model.load_state_dict(ck.get("best_state_dict", ck.get("state_dict")), strict=False)
    model.eval()
    torch.set_grad_enabled(False)

    y, _ = librosa.load(audio_path, sr=model.fs, mono=False)
    if y.ndim == 1:
        y = y[None, :]
    x = torch.from_numpy(np.ascontiguousarray(y.T.astype(np.float32))).to(dev)
    est = model.transcribe(x, discardSecondHalf=False)
    out_mid = os.path.join(workdir, "transkun.mid")
    writeMidi(est).write(out_mid)
    return leggi_midi(out_mid)


def trascrivi(audio_path: str, motore: str, workdir: str, cb=None):
    """motore: auto | transkun | crnn. In auto: Transkun se c'è, altrimenti CRNN."""
    if motore in ("auto", "transkun") and transkun_disponibile():
        try:
            note, ped = trascrivi_transkun(audio_path, workdir, cb)
            if note:
                log(f"Note rilevate: {len(note)} | pedalate: {len(ped)}", cb)
                return note, ped, "transkun"
        except Exception as e:
            log(f"Transkun non è riuscito ({type(e).__name__}: {str(e)[:120]}).", cb)
            log("Passo al motore CRNN.", cb)
    elif motore == "transkun":
        log("Transkun non è installato: uso il motore CRNN.", cb)
    note, ped = trascrivi_crnn(audio_path, cb)
    log(f"Note rilevate: {len(note)} | pedalate: {len(ped)}", cb)
    return note, ped, "crnn"


# ─────────────────────────────────────────────────────────────────────────────
# 3. VERIFICA SULL'AUDIO ORIGINALE
# ─────────────────────────────────────────────────────────────────────────────

def verifica_audio(y: np.ndarray, note: List[Nota], cb=None) -> Dict:
    """
    Due controlli indipendenti dal modello di trascrizione:
      a) transienti — SuperFlux sull'audio: c'è un attacco dove la nota inizia?
      b) altezza    — nella trasformata a Q costante c'è energia su quella nota
                      (fondamentale o primi armonici) mentre suona?
    Una nota è «senza riscontro» solo se fallisce ENTRAMBI: così il rosso in
    partitura resta raro e significativo.
    """
    import librosa
    log("Verifica sull'audio originale...", cb)

    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP,
                                       fmax=8000, n_mels=138)
    odf = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max), sr=SR,
                                       hop_length=HOP, lag=2, max_size=3)
    fr = librosa.onset.onset_detect(onset_envelope=odf, sr=SR, hop_length=HOP,
                                    backtrack=False, units="frames", pre_max=3,
                                    post_max=3, pre_avg=10, post_avg=10,
                                    delta=0.07, wait=2)
    ref = librosa.frames_to_time(fr, sr=SR, hop_length=HOP)
    # il rilevatore segna l'attacco con qualche decina di ms di ritardo
    # (dipende dallo strumento e dalla registrazione): lo si misura e lo si toglie
    ons0 = np.array(sorted({round(n.onset, 3) for n in note}))
    if len(ref) and len(ons0):
        d = np.array([ons0[int(np.argmin(np.abs(ons0 - t)))] - t for t in ref])
        d = d[np.abs(d) < 0.12]
        if len(d) >= 10:
            ref = ref + float(np.median(d))

    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=27.5, n_bins=88,
                           bins_per_octave=12))
    nfr = C.shape[1]
    fps = SR / HOP

    tol = 0.060
    senza = 0
    for n in note:
        ha_attacco = bool(len(ref)) and float(np.min(np.abs(ref - n.onset))) <= tol
        f0 = int(round(n.onset * fps))
        f1 = int(round(min(n.offset, n.onset + 0.30) * fps))
        f0 = max(0, min(f0, nfr - 1))
        f1 = max(f0 + 1, min(f1 + 1, nfr))
        seg = C[:, f0:f1].mean(axis=1)
        picco = float(seg.max()) + 1e-9
        e = 0.0
        for h in (0, 12, 19):                       # fondamentale, 2° e 3° armonico
            b = n.pitch - 21 + h
            if 0 <= b < 88:
                e = max(e, float(seg[b]))
        ha_altezza = (e / picco) >= 0.06
        n.verified = ha_attacco or ha_altezza
        if not n.verified:
            senza += 1

    # attacchi presenti nell'audio ma non trascritti
    ons = np.array(sorted({round(n.onset, 3) for n in note}))
    persi = [float(t) for t in ref
             if not len(ons) or float(np.min(np.abs(ons - t))) > tol]
    coperti = len(ref) - len(persi)
    rec = coperti / len(ref) if len(ref) else 1.0
    log(f"Note senza riscontro nell'audio: {senza} su {len(note)} | "
        f"attacchi dell'audio non trascritti: {len(persi)} su {len(ref)}", cb)
    return {"note_trascritte": len(note), "note_senza_riscontro": senza,
            "attacchi_audio": int(len(ref)), "attacchi_coperti": int(coperti),
            "copertura": round(rec, 3), "_persi": persi}


# ─────────────────────────────────────────────────────────────────────────────
# 4. PULIZIA, ACCORDI, ARPEGGIATI
# ─────────────────────────────────────────────────────────────────────────────

def pulisci(note: List[Nota]) -> List[Nota]:
    """Toglie doppioni e note fantasma, e fissa l'attacco originale."""
    note = sorted(note, key=lambda n: (n.onset, n.pitch))
    out: List[Nota] = []
    ultimo: Dict[int, Nota] = {}
    for n in note:
        if n.offset - n.onset < 0.02:
            n.offset = n.onset + 0.02
        p = ultimo.get(n.pitch)
        if p is not None and n.onset - p.onset < 0.03:      # stesso tasto, stesso istante
            p.offset = max(p.offset, n.offset)
            p.vel = max(p.vel, n.vel)
            continue
        if p is not None and p.offset > n.onset:            # il tasto viene ribattuto
            p.offset = n.onset
        n.raw_onset = n.onset
        out.append(n)
        ultimo[n.pitch] = n
    return out


def trova_arpeggiati(note: List[Nota], pedali=None) -> int:
    """
    Accordo arpeggiato: 3+ note in ordine di altezza, a passo stretto, che
    restano tutte tenute insieme. Le scale veloci non ci cascano perché
    procedono per grado e ogni tasto viene rilasciato subito. Le note estranee
    che cadono in mezzo (la melodia dell'altra mano) vengono saltate.
    """
    n = len(note)
    gid = 0
    ped = sorted(pedali or [])

    def sotto_pedale(t: float) -> bool:
        return any(a - 0.02 <= t < b_ - 0.05 for a, b_ in ped)

    for i in range(n):
        if note[i].arp:
            continue
        fin = [k for k in range(i, n) if note[k].onset - note[i].onset <= ARP_MAX_SPAN]
        fin = [k for k in fin if not note[k].arp]
        migliore: List[int] = []
        for verso in (1, -1):
            # catena più lunga che parte da i: programmazione dinamica sulla finestra
            lung = {i: 1}
            prec = {i: None}
            for k in fin[1:]:
                for j in fin:
                    if j >= k or j not in lung:
                        continue
                    dt = note[k].onset - note[j].onset
                    dp = (note[k].pitch - note[j].pitch) * verso
                    if 0.012 <= dt <= ARP_MAX_STEP and 2 <= dp <= 12:
                        if lung[j] + 1 > lung.get(k, 0):
                            lung[k] = lung[j] + 1
                            prec[k] = j
            fine_c = max(lung, key=lambda k: lung[k])
            cat = []
            while fine_c is not None:
                cat.append(fine_c)
                fine_c = prec[fine_c]
            cat.reverse()
            if len(cat) > len(migliore):
                migliore = cat
        grp = migliore
        if len(grp) < 3:
            continue
        t0, t1 = note[grp[0]].onset, note[grp[-1]].onset
        if t1 - t0 <= CHORD_SPREAD:
            continue
        ampiezza = abs(note[grp[-1]].pitch - note[grp[0]].pitch)
        if ampiezza < 7 or ampiezza / (len(grp) - 1) < 2.5:
            continue
        # tenute insieme: con le dita, oppure dal pedale di risonanza
        if not all(note[k].offset > t1 + 0.10 or sotto_pedale(note[k].offset) for k in grp):
            continue
        gid += 1
        for k in grp:
            note[k].arp = gid
            note[k].onset = t0
    return gid


def allinea_accordi(note: List[Nota]):
    """Attacchi entro 35 ms diventano un unico istante: sono un accordo."""
    note.sort(key=lambda n: (n.onset, n.pitch))
    i, n = 0, len(note)
    while i < n:
        j = i + 1
        while j < n and note[j].onset - note[i].onset <= CHORD_SPREAD:
            j += 1
        if j - i > 1:
            t = float(np.median([note[k].onset for k in range(i, j)]))
            for k in range(i, j):
                note[k].onset = t
        i = j
    note.sort(key=lambda n: (n.onset, n.pitch))


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEPARAZIONE DELLE MANI
# ─────────────────────────────────────────────────────────────────────────────

def separa_mani(note: List[Nota], cb=None):
    """
    Per ogni attacco si sceglie un punto di divisione fra le mani; la sequenza
    dei punti è ottimizzata tutta insieme (Viterbi) pesando:
      - quanto il punto cade «in mezzo» alle note che suonano in quel momento,
      - che nessuna mano debba coprire più di una decima,
      - che il punto non salti su e giù da un attacco all'altro.
    Una soglia fissa sul Do centrale sbaglia appena la sinistra sale.
    """
    if not note:
        return
    tempi = sorted({n.onset for n in note})
    idx = {t: i for i, t in enumerate(tempi)}
    T = len(tempi)
    S = np.arange(33, 89)                         # divisioni candidate (La1 … Mi6)
    costo = np.zeros((T, len(S)))

    per_t: List[List[Nota]] = [[] for _ in range(T)]
    for n in note:
        per_t[idx[n.onset]].append(n)

    # note che suonano attorno a ogni attacco (contesto di ±0.6 s)
    on = np.array([n.onset for n in note])
    off = np.array([n.offset for n in note])
    pit = np.array([n.pitch for n in note], dtype=float)
    for i, t in enumerate(tempi):
        a = np.searchsorted(on, t - 0.6, "left")
        b = np.searchsorted(on, t + 0.6, "right")
        ctx = pit[a:b]
        w = np.exp(-np.abs(on[a:b] - t) / 0.3)
        # il punto di divisione va tenuto lontano dalle note
        d = np.abs(S[:, None] - 0.5 - ctx[None, :])
        costo[i] += (w[None, :] * np.exp(-d / 2.5)).sum(axis=1) * 1.2
        # le due mani si dividono il lavoro: una divisione che lascia quasi tutto
        # a una mano sola (basso isolato, accordi e melodia insieme) è sospetta
        sopra = (w[None, :] * (ctx[None, :] >= S[:, None])).sum(axis=1) / max(float(w.sum()), 1e-9)
        costo[i] += PESO_EQUILIBRIO * np.abs(sopra - 0.5)
        # note che stanno suonando in questo istante (iniziate ora o ancora tenute)
        a0 = np.searchsorted(on, t - 4.0, "left")
        b0 = np.searchsorted(on, t, "right")
        ora = np.sort(pit[a0:b0][off[a0:b0] > t + 0.04])
        vicine = np.sort(ctx[np.abs(on[a:b] - t) <= 0.45])
        for si, s in enumerate(S):
            # una mano non può coprire più di una decima nello stesso istante…
            sx, dx = ora[ora < s], ora[ora >= s]
            if len(sx) > 1:
                costo[i, si] += max(0.0, sx[-1] - sx[0] - 14) * 2.0
            if len(dx) > 1:
                costo[i, si] += max(0.0, dx[-1] - dx[0] - 14) * 2.0
            # …né saltare per più di un'ottava e mezza in mezzo secondo
            sx, dx = vicine[vicine < s], vicine[vicine >= s]
            if len(sx) > 1:
                costo[i, si] += max(0.0, sx[-1] - sx[0] - 17) * 0.6
            if len(dx) > 1:
                costo[i, si] += max(0.0, dx[-1] - dx[0] - 17) * 0.6
        costo[i] += np.abs(S - 60) * 0.035        # preferenza debole per il Do centrale

    trans = np.abs(S[:, None] - S[None, :]) * 0.22
    D = costo[0].copy()
    back = np.zeros((T, len(S)), dtype=np.int32)
    for i in range(1, T):
        tot = D[:, None] + trans
        back[i] = np.argmin(tot, axis=0)
        D = tot[back[i], np.arange(len(S))] + costo[i]
    s = int(np.argmin(D))
    div = np.zeros(T)
    for i in range(T - 1, -1, -1):
        div[i] = S[s]
        s = back[i, s]

    for n in note:
        n.hand = 1 if n.pitch < div[idx[n.onset]] else 0
    dx = sum(1 for n in note if n.hand == 0)
    log(f"Mani — destra: {dx} note, sinistra: {len(note) - dx}", cb)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ABBELLIMENTI  (prima della griglia: è il punto che decide tutto)
# ─────────────────────────────────────────────────────────────────────────────

def trova_abbellimenti(note: List[Nota], cb=None, met: Optional[Metrica] = None) -> Tuple[List[Nota], Dict[str, int]]:
    """
    Un'acciaccatura dura 30-60 ms. Se entra nella griglia metrica costringe il
    quantizzatore a suddivisioni assurde che rovinano l'intera battuta. Quindi
    gli abbellimenti escono dal flusso PRIMA che la griglia esista, e tornano
    come simboli.

    Ogni regola richiede che la figura sia isolata (preceduta e seguita da
    valori nettamente più lunghi): così una scala veloce o un basso albertino
    non vengono scambiati per abbellimenti. Nel dubbio le note restano note.
    """
    conta = {"acciaccature": 0, "trilli": 0, "mordenti": 0, "gruppetti": 0,
             "arpeggiati": len({n.arp for n in note if n.arp})}
    fuori = set()
    # Velocità in rapporto al tempo del brano. In un presto le semicrome scritte
    # vanno veloci quanto un abbellimento: la sola durata in millisecondi non
    # basta. passo_q() dà il passo fra due note in quarti; una figura che
    # procede esattamente a semicrome o a biscrome è musica scritta.
    bq_ = float(met.beat_ql) if met is not None else None

    def passo_q(a: Nota, b: Nota) -> float:
        return (b.pos - a.pos) * bq_ if bq_ else 0.0

    def a_griglia(pq: float) -> bool:
        """il passo coincide (entro il 5%) con una suddivisione binaria scritta?"""
        return bool(bq_) and (abs(pq - 0.25) < 0.0125 or abs(pq - 0.125) < 0.00625
                              or abs(pq - 0.5) < 0.025)

    for mano in (0, 1):
        hs = [n for n in note if n.hand == mano]
        hs.sort(key=lambda n: (n.onset, n.pitch))
        m = len(hs)
        if not m:
            continue
        usato = [False] * m
        ons = np.array([n.onset for n in hs])

        def attacco_dopo(t):
            k = int(np.searchsorted(ons, t + 1e-6, "right"))
            return ons[k] if k < m else t + 10.0

        def attacco_prima(t):
            k = int(np.searchsorted(ons, t - 1e-6, "left"))
            return ons[k - 1] if k > 0 else t - 10.0

        def catena(i, regola, quante=99):
            """
            Note successive e veloci a partire da i. regola(altezze_finora, p)
            dice se l'altezza p può proseguire la figura; le altre note della
            mano (un accordo tenuto sotto, per esempio) vengono saltate.
            """
            seq = [i]
            j = i + 1
            while j < m and len(seq) < quante and hs[j].onset - hs[seq[-1]].onset <= ORN_MAX_IOI + 0.03:
                dt = hs[j].raw_onset - hs[seq[-1]].raw_onset
                if (not usato[j] and 0.02 <= dt <= ORN_MAX_IOI and not hs[j].arp and not hs[j].orn
                        and regola([hs[k].pitch for k in seq], hs[j].pitch)):
                    seq.append(j)
                j += 1
            return seq

        def lunga(k, passo):
            """la nota k è nettamente più lunga del passo dell'abbellimento?"""
            n = hs[k]
            return (n.offset - n.onset >= ISOLATION * passo * 0.8
                    and attacco_dopo(n.onset) - n.onset >= ISOLATION * passo)

        def togli(seq):
            for k in seq:
                usato[k] = True
                fuori.add(id(hs[k]))

        def forza(k):
            """peso metrico (griglia preliminare) della posizione della nota k."""
            fr = hs[k].pos - math.floor(hs[k].pos)
            if min(fr, 1 - fr) < 0.11:
                return 3
            if abs(fr - 0.5) < 0.09:
                return 2
            if min(abs(fr - c) for c in (0.25, 0.75, 1 / 3, 2 / 3)) < 0.07:
                return 1
            return 0

        # ── gruppetto ────────────────────────────────────────────────────────
        def gruppetto4(pp, p):          # superiore, reale, inferiore, reale
            k = len(pp)
            if k == 1:
                return 1 <= pp[0] - p <= 2
            if k == 2:
                return 1 <= pp[1] - p <= 2
            return k == 3 and p == pp[1]

        def gruppetto5(pp, p):          # reale, superiore, reale, inferiore, reale
            k = len(pp)
            if k == 1:
                return 1 <= p - pp[0] <= 2
            if k == 2:
                return p == pp[0]
            if k == 3:
                return 1 <= pp[0] - p <= 2
            return k == 4 and p == pp[0]

        for i in range(m):
            if usato[i] or hs[i].arp:
                continue
            for regola, forma in ((gruppetto5, 5), (gruppetto4, 4)):
                seq = catena(i, regola, forma)
                if len(seq) != forma:
                    continue
                passi = np.diff([hs[k].raw_onset for k in seq])
                # un abbellimento parte SUL tempo: se la nota lunga finale cade
                # in posizione più forte dell'inizio, sono note normali
                if not lunga(seq[-1], float(passi.max())) or forza(seq[0]) < forza(seq[-1]):
                    continue
                pq = passo_q(hs[seq[0]], hs[seq[-1]]) / (forma - 1)
                if bq_ and pq >= 0.205:              # lento quanto una semicroma: è scritto
                    continue
                capo = hs[seq[0]]
                capo.pitch = hs[seq[1]].pitch if forma == 4 else capo.pitch
                capo.offset = hs[seq[-1]].offset
                capo.vel = max(hs[k].vel for k in seq)
                capo.orn = "turn"
                togli(seq[1:])
                conta["gruppetti"] += 1
                break

        # ── trillo e mordente: alternanza fra due altezze vicine ─────────────
        for i in range(m):
            if usato[i] or hs[i].arp or hs[i].orn:
                continue
            p0 = hs[i].pitch
            seq, dq_m = [i], 0
            for dq in (1, 2, -1, -2):
                c = catena(i, lambda pp, p, dq=dq: p == (p0 + dq if len(pp) % 2 else p0))
                if len(c) > len(seq):
                    seq, dq_m = c, dq
            if len(seq) < 3:
                continue
            passi = np.diff([hs[k].raw_onset for k in seq])
            capo = hs[seq[0]]
            passo_m = float(np.median(passi))
            regolare = float(np.percentile(passi, 85)) <= 1.6 * float(np.percentile(passi, 15))
            pq = passo_q(hs[seq[0]], hs[seq[-1]]) / (len(seq) - 1)
            if len(seq) >= TRILL_MIN and a_griglia(pq):
                continue                             # semicrome o biscrome alternate, scritte
            if len(seq) == 3 and bq_ and pq >= 0.205:
                continue
            if regolare and ((len(seq) >= 5 and passo_m <= 0.105)
                             or (len(seq) == TRILL_MIN and passo_m <= 0.088)):
                capo.pitch = min(p0, p0 + dq_m)        # nota reale = la più bassa
                capo.offset = max(hs[k].offset for k in seq)
                capo.orn = "trill"
                conta["trilli"] += 1
            elif (len(seq) == 3 and lunga(seq[2], float(passi.max()))
                  and forza(seq[0]) >= forza(seq[2])):
                capo.offset = hs[seq[2]].offset
                capo.orn = "inv_mordent" if dq_m > 0 else "mordent"
                conta["mordenti"] += 1
            else:
                continue
            capo.vel = max(hs[k].vel for k in seq)
            togli(seq[1:])

        # ── acciaccature: da 1 a 3 note brevissime prima di una nota reale ───
        gruppi: List[List[int]] = []
        i = 0
        while i < m:
            j = i
            while j + 1 < m and hs[j + 1].onset == hs[i].onset:
                j += 1
            vivi = [k for k in range(i, j + 1) if not usato[k]]
            if vivi:
                gruppi.append(vivi)
            i = j + 1
        for g in range(1, len(gruppi)):
            reale = [k for k in gruppi[g] if not usato[k]]
            if not reale:
                continue
            t_reale = hs[reale[0]].onset
            pos_reale = hs[reale[0]].pos
            for bersaglio in sorted(reale, key=lambda k: -hs[k].pitch):
                piccole: List[int] = []
                succ_t, succ_p = t_reale, hs[bersaglio].pitch
                h = g - 1
                while h >= 0 and len(piccole) < 3:
                    # nel gruppo precedente: una sola nota breve e vicina d'altezza;
                    # le altre (accompagnamento) devono essere note tenute
                    brevi = [k for k in gruppi[h] if not usato[k]
                             and hs[k].offset - hs[k].onset <= 0.13
                             and 1 <= abs(hs[k].pitch - succ_p) <= 12]
                    lunghe = [k for k in gruppi[h] if not usato[k] and k not in brevi]
                    if len(brevi) != 1 or any(hs[k].offset - hs[k].onset <= 0.13 for k in lunghe):
                        break
                    a = hs[brevi[0]]
                    gap = succ_t - a.onset
                    if a.orn or a.arp or a.graces:
                        break
                    if not (0.02 <= gap <= GRACE_MAX_GAP):
                        break
                    if a.offset > succ_t + 0.09:             # tenuta: non è un'acciaccatura
                        break
                    # lontana dal tempo (in movimenti): è una nota scritta, non un'acciaccatura.
                    # Più è lenta, più dev'essere a ridosso della nota reale.
                    if pos_reale - a.pos > (0.22 if gap <= 0.07 or piccole else 0.19):
                        break
                    piccole.insert(0, brevi[0])
                    succ_t, succ_p = a.onset, a.pitch
                    h -= 1
                if not piccole:
                    continue
                tt = [hs[k].onset for k in piccole] + [t_reale]
                passo = float(np.max(np.diff(tt)))
                prima = max((hs[k].onset for k in range(m)
                             if hs[k].onset < tt[0] - 1e-6 and not usato[k]
                             and abs(hs[k].pitch - hs[piccole[0]].pitch) <= 12), default=tt[0] - 10.0)
                # isolamento: non deve essere la coda di una volata a passo costante
                if tt[0] - prima >= 1.2 * (tt[1] - tt[0]) and lunga(bersaglio, passo):
                    hs[bersaglio].graces = [hs[k].pitch for k in piccole] + hs[bersaglio].graces
                    togli(piccole)
                    conta["acciaccature"] += len(piccole)
                    break

    resto = [n for n in note if id(n) not in fuori]
    log("Abbellimenti — " + ", ".join(f"{k}: {v}" for k, v in conta.items()), cb)
    return resto, conta


# ─────────────────────────────────────────────────────────────────────────────
# 7. METRICA: movimenti, tempo, battute
# ─────────────────────────────────────────────────────────────────────────────

def _inviluppo(note: List[Nota], fr: int = 100) -> np.ndarray:
    """
    Curva degli accenti su cui si cerca la pulsazione. Non conta quante note
    cadono in un istante né quanto sono forti (nei brani sincopati gli accordi
    in levare vincerebbero sui bassi in battere): conta ciò che nella musica
    scritta sta davvero sui tempi forti, cioè le note lunghe e le più gravi.
    Con la vecchia curva, sui pezzi veri, il tracciatore finiva in controtempo
    quattro volte più spesso.
    """
    from scipy.ndimage import gaussian_filter1d
    T = max(n.offset for n in note) + 3.0
    env = np.zeros(int(T * fr) + 2)
    on = np.array([n.onset for n in note])
    pit = np.array([n.pitch for n in note])
    peso: Dict[float, float] = {}
    quante: Dict[float, int] = {}
    for n in note:
        a = int(np.searchsorted(on, n.onset - 0.6, "left"))
        b = int(np.searchsorted(on, n.onset + 0.6, "right"))
        grave = 1.0 if n.pitch <= pit[a:b].min() + 2 else 0.0
        w = 0.3 + 0.7 * min((n.offset - n.onset) / 0.5, 1.5) + grave + 0.3 * n.vel / 127.0
        peso[n.onset] = max(peso.get(n.onset, 0.0), w)
        quante[n.onset] = quante.get(n.onset, 0) + 1
    for t, w in peso.items():
        env[int(round(t * fr))] += w + 0.15 * min(quante[t], 4)
    return gaussian_filter1d(env, 1.8)


def _raffina(beats: np.ndarray, note: List[Nota]) -> np.ndarray:
    """Avvicina ogni movimento all'attacco reale più vicino (segue il rubato)."""
    ons = np.array(sorted({n.onset for n in note}))
    out = beats.copy()
    med = float(np.median(np.diff(beats)))
    for i, b in enumerate(beats):
        k = int(np.argmin(np.abs(ons - b)))
        d = ons[k] - b
        if abs(d) <= min(0.06, 0.12 * med):
            out[i] = b + 0.8 * d
    # la griglia deve restare strettamente crescente
    for i in range(1, len(out)):
        if out[i] <= out[i - 1] + 0.3 * med:
            out[i] = out[i - 1] + 0.5 * med
    return out


def _regola_bordi(beats: np.ndarray) -> np.ndarray:
    """
    Ai due estremi del brano il tracciatore non ha contesto e spesso piazza
    male il primo o l'ultimo movimento. Se un intervallo di bordo si discosta
    troppo da quelli vicini, lo si ricalcola dal passo locale.
    """
    b = beats.copy()
    n = len(b)
    if n < 8:
        return b
    for i in (2, 1, 0):                              # inizio, dall'interno verso fuori
        med = float(np.median(np.diff(b[i + 1: i + 6])))
        if abs((b[i + 1] - b[i]) - med) > 0.15 * med:
            b[i] = b[i + 1] - med
    for i in (n - 3, n - 2, n - 1):                  # fine
        med = float(np.median(np.diff(b[i - 5: i])))
        if abs((b[i] - b[i - 1]) - med) > 0.15 * med:
            b[i] = b[i - 1] + med
    return b


def _evidenza_fase(beats: np.ndarray, note: List[Nota]) -> np.ndarray:
    """
    Corregge la fase della griglia tratto per tratto. Nei brani sincopati il
    tracciatore passa dal battere al levare e viceversa anche più volte; qui,
    per ogni movimento, si guarda dove stanno le note più gravi e più lunghe
    (sul movimento o a metà?), si decide con un Viterbi a due stati quali
    tratti sono in controtempo, e solo quelli vengono spostati di mezzo
    movimento. Poi si richiudono i buchi e si tolgono i doppioni.
    """
    n = len(beats)
    idx = np.arange(n, dtype=float)
    on = np.array([x.onset for x in note])
    pos = np.interp(on, beats, idx)
    fin = np.interp(np.array([x.offset for x in note]), beats, idx)
    pit = np.array([x.pitch for x in note], dtype=float)
    dur = np.minimum(fin - pos, 2.0)
    grave = np.full((2, n), np.nan)
    lunga = np.zeros((2, n))
    for lato, sfas in ((0, 0.0), (1, 0.5)):
        fr = (pos - sfas) - np.round(pos - sfas)
        sel = np.abs(fr) < 0.13
        k = np.clip(np.round(pos - sfas)[sel].astype(int), 0, n - 1)
        for kk, p_, d_ in zip(k, pit[sel], dur[sel]):
            if np.isnan(grave[lato, kk]) or p_ < grave[lato, kk]:
                grave[lato, kk] = p_
            lunga[lato, kk] = max(lunga[lato, kk], d_)
    e = np.zeros(n)                       # > 0: in fase; < 0: controtempo
    for k in range(n):
        b_, l_ = grave[0, k], grave[1, k]
        if not np.isnan(b_) and not np.isnan(l_):
            e[k] = (np.clip((l_ - b_) / 4.0, -1.5, 1.5)
                    + np.clip((lunga[0, k] - lunga[1, k]) / 0.25, -1.5, 1.5))
        elif not np.isnan(b_):
            e[k] = 0.5
        elif not np.isnan(l_):
            e[k] = -0.5
    return e


def _fase_locale(beats: np.ndarray, note: List[Nota]) -> np.ndarray:
    """Sposta di mezzo movimento i soli tratti in controtempo (vedi _evidenza_fase)."""
    n = len(beats)
    if n < 12:
        return beats
    e = _evidenza_fase(beats, note)
    # Costo del cambio di fase. Normalmente alto: l'indizio locale, nei brani
    # sincopati, oscilla. Ma dove la griglia stessa ha un'anomalia che somma a
    # circa mezzo movimento (la firma di una scivolata del tracciatore) il
    # cambio è atteso: lì costa poco, e costa qualcosa NON cambiare.
    ioi = np.diff(beats)
    loc = np.array([np.median(ioi[max(0, i - 8): i + 9]) for i in range(len(ioi))])
    dev = (ioi - loc) / loc
    cambio = np.full(n, 14.0)
    resto = np.zeros(n)
    for k in range(1, n):
        cum = abs(float(dev[max(0, k - 4): k + 3].sum()))
        if 0.28 <= cum <= 0.72 and np.abs(dev[max(0, k - 4): k + 3]).max() > 0.12:
            cambio[k], resto[k] = 0.5, 0.4
    costo = np.stack([-e, e])             # stato 0 = in fase, 1 = controtempo
    D = costo[:, 0].copy()
    back = np.zeros((2, n), dtype=int)
    for k in range(1, n):
        nuovo = np.zeros(2)
        for st in (0, 1):
            resta, cambia = D[st] + resto[k], D[1 - st] + cambio[k]
            back[st, k] = st if resta <= cambia else 1 - st
            nuovo[st] = min(resta, cambia) + costo[st, k]
        D = nuovo
    st = int(np.argmin(D))
    stato = np.zeros(n, dtype=int)
    for k in range(n - 1, -1, -1):
        stato[k] = st
        st = back[st, k]
    if not stato.any():
        return beats
    T = float(np.median(np.diff(beats)))
    out = []
    for k in range(n):
        if stato[k] == 0:
            out.append(beats[k])
        else:
            passo = (beats[k + 1] - beats[k]) if k + 1 < n else T
            out.append(beats[k] + passo / 2)
    return _regolarizza(np.array(sorted(out)))


def _regolarizza(b: np.ndarray) -> np.ndarray:
    """
    Dove la griglia ha intervalli anomali (dopo una correzione di fase, o per
    una scivolata del tracciatore) il tratto compreso fra due movimenti
    regolari viene ridiviso in parti uguali, tante quante ne servono a
    coprirlo col passo locale. Così non si perde né si inventa un movimento.
    """
    n = len(b)
    if n < 6:
        return b
    ioi = np.diff(b)
    T = float(np.median(ioi))
    strano = (ioi < 0.8 * T) | (ioi > 1.25 * T)
    if not strano.any():
        return b
    out = [b[0]]
    i = 0
    while i < len(ioi):
        if not strano[i]:
            out.append(b[i + 1])
            i += 1
            continue
        j = i
        while j + 1 < len(ioi) and strano[j + 1]:
            j += 1
        vicini = np.concatenate([ioi[max(0, i - 4): i], ioi[j + 1: j + 5]])
        vicini = vicini[(vicini > 0.8 * T) & (vicini < 1.25 * T)]
        t_loc = float(np.median(vicini)) if len(vicini) else T
        span = b[j + 1] - out[-1]
        k = max(1, int(round(span / t_loc)))
        for m in range(1, k + 1):
            out.append(out[-1] + (b[j + 1] - out[-1]) / (k - m + 1))
        i = j + 1
    return np.array(out)


def _estendi(beats: np.ndarray, t0: float, t1: float) -> np.ndarray:
    b = list(beats)
    passo = float(np.median(np.diff(beats[:5]))) if len(beats) > 5 else float(beats[1] - beats[0])
    while b[0] > t0 - passo:
        b.insert(0, b[0] - passo)
    passo = float(np.median(np.diff(beats[-5:]))) if len(beats) > 5 else passo
    while b[-1] < t1 + passo:
        b.append(b[-1] + passo)
    return np.array(b)


def _z(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _autocorr(a: np.ndarray, lag: int) -> float:
    if len(a) <= lag + 2:
        return 0.0
    x = a - a.mean()
    den = float((x * x).sum()) + 1e-9
    return float((x[:-lag] * x[lag:]).sum() / den)


def _caratteri(note: List[Nota], pos: np.ndarray, fine: np.ndarray, nb: int) -> Dict[str, np.ndarray]:
    """
    Cosa succede in ogni movimento, visto da quattro lati diversi:
      acc   volume degli attacchi sul movimento
      basso quanto è grave la nota più bassa (il basso cade sui tempi forti)
      dens  quanti attacchi contiene (il disegno ritmico si ripete a ogni battuta)
      dur   la nota più lunga che vi inizia (le note lunghe partono sui tempi forti)
      pc    istogramma delle 12 note (l'armonia cambia sulla stanghetta)
    """
    acc = np.zeros(nb)
    dens = np.zeros(nb)
    dur = np.zeros(nb)
    grave = np.full(nb, np.nan)
    pc = np.zeros((nb, 12))
    for n, p, e in zip(note, pos, fine):
        k = int(math.floor(p + 0.12))
        if not (0 <= k < nb):
            continue
        dens[k] += 1
        pc[k, n.pitch % 12] += min(e - p, 2.0) + 0.2
        if abs(p - k) < 0.16:                       # attacco sul movimento
            acc[k] += n.vel / 127.0
            dur[k] = max(dur[k], min(e - p, 4.0))
            if np.isnan(grave[k]) or n.pitch < grave[k]:
                grave[k] = n.pitch
    vuoti = np.isnan(grave)
    if vuoti.all():
        grave[:] = 60.0
    else:
        grave[vuoti] = float(np.nanmax(grave))      # nessun basso = «acuto»
    # gravità relativa: rispetto ai movimenti vicini, così un cambio d'armonia
    # non nasconde il disegno basso-accordo-accordo
    rel = np.array([grave[k] - np.median(grave[max(0, k - 4): k + 5]) for k in range(nb)])
    return {"acc": acc, "basso": -rel, "dens": dens, "dur": dur, "pc": pc, "grave": grave}


def _somiglianza(pc: np.ndarray, lag: int) -> float:
    """quanto l'armonia del movimento k somiglia a quella del movimento k+lag."""
    if len(pc) <= lag + 2:
        return 0.0
    a, b = pc[:-lag], pc[lag:]
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-9
    return float(((a * b).sum(axis=1)[ok] / den[ok]).mean()) if ok.any() else 0.0


def _periodo(car: Dict[str, np.ndarray], lag: int) -> float:
    return (_autocorr(car["basso"], lag) + 0.7 * _autocorr(car["dens"], lag)
            + 0.5 * _autocorr(car["acc"], lag) + 0.5 * _autocorr(car["dur"], lag))


def _battere(car: Dict[str, np.ndarray], g: int) -> int:
    """Su quale movimento (0…g-1) cade il battere."""
    pc = car["pc"]
    nb = len(pc)
    nov = np.zeros(nb)                              # novità armonica sulla stanghetta
    for k in range(g, nb - g):
        a, b = pc[k - g:k].sum(axis=0), pc[k:k + g].sum(axis=0)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        nov[k] = 1.0 - (float(a @ b) / den if den > 1e-9 else 1.0)
    tot = (1.2 * _z(nov) + 1.0 * _z(car["basso"]) + 0.6 * _z(car["acc"])
           + 0.6 * _z(car["dur"]))
    punti = np.array([tot[f::g].mean() if len(tot[f::g]) else -9.0 for f in range(g)])
    # l'indizio più affidabile: la nota più grave della battuta sta sul battere
    # (basso del valzer, fondamentale dell'accompagnamento). Non dipende dal
    # volume, che nelle registrazioni spesso tradisce.
    grave = car["grave"]
    quota = np.zeros(g)
    for f in range(g):
        bar_ = [grave[k: k + g] for k in range(f, nb - g + 1, g)]
        bar_ = [x for x in bar_ if x.max() - x.min() >= 1.0]
        if bar_:
            quota[f] = float(np.mean([x[0] <= x[1:].min() + 0.5 for x in bar_]))
    # fa da filtro: restano in gara solo le fasi col basso sul battere quasi
    # quanto la migliore; fra quelle decidono armonia, accento e durate
    punti[quota < quota.max() - 0.25] = -9.0
    return int(np.argmax(punti))


def _controtempo(note: List[Nota], pos: np.ndarray, fine: np.ndarray) -> float:
    """
    La griglia è in fase o cade sui levare? Nei brani sincopati (ragtime,
    polacche) gli accordi in levare sono più numerosi e più forti dei bassi in
    battere, e il tracciatore si aggancia a quelli. Due indizi che non
    dipendono dal volume: sui battere stanno le note più gravi e quelle più
    lunghe. Ritorna un punteggio: positivo = in fase, negativo = controtempo.
    """
    pit = np.array([n.pitch for n in note], dtype=float)
    dur = np.minimum(fine - pos, 2.0)
    dati = []
    for sfas in (0.0, 0.5):
        fr = (pos - sfas) - np.round(pos - sfas)
        sel = np.abs(fr) < 0.13
        if sel.sum() < 8:
            return 1.0                               # troppo poco per giudicare
        k = np.round(pos - sfas)[sel].astype(int)
        gravi: Dict[int, float] = {}
        for kk, p_ in zip(k, pit[sel]):
            gravi[kk] = min(gravi.get(kk, 999.0), p_)
        dati.append((float(np.mean(list(gravi.values()))), float(dur[sel].mean())))
    (g_b, d_b), (g_l, d_l) = dati
    return (g_l - g_b) / 4.0 + (d_b - d_l) / 0.12


def _note_perse(note: List[Nota], pos: np.ndarray) -> float:
    """
    Quota di attacchi che con questa pulsazione non si potrebbero scrivere,
    perché più fitti di otto per movimento (finirebbero l'uno sull'altro).
    """
    perse = tot = 0
    for mano in (0, 1):
        pp = np.unique(np.round([p for n, p in zip(note, pos) if n.hand == mano], 4))
        if not len(pp):
            continue
        k = np.floor(pp + 0.07).astype(int)
        for kk in np.unique(k):
            fr = pp[k == kk] - kk
            tot += len(fr)
            perse += min(len(fr) - len(set(np.clip(np.round(fr * d), 0, d).tolist()))
                         for d in (8, 6))
    return perse / max(tot, 1)


def _costo_notazione(note: List[Nota], pos: np.ndarray) -> float:
    """
    Quanto «costa» scrivere il brano con questa pulsazione: per ogni mano e
    ogni movimento la suddivisione più economica (come farà il quantizzatore),
    più un fisso per movimento. Serve a scegliere fra una pulsazione e la sua
    metà: le terzine lette a tempo doppio costano il doppio.
    """
    pen = QUANT["media"]["pen"]
    lam = QUANT["media"]["lam"]
    tot = 0.0
    for mano in (0, 1):
        pp = np.unique(np.round([p for n, p in zip(note, pos) if n.hand == mano], 4))
        if not len(pp):
            continue
        k = np.floor(pp + 0.07).astype(int)
        for kk in np.unique(k):
            fr = pp[k == kk] - kk
            best = 1e9
            for d, pd in pen.items():
                snap = np.clip(np.round(fr * d), 0, d)
                c = (float(np.abs(fr - snap / d).sum()) / lam + pd
                     + COLLISION / lam * (len(snap) - len(set(snap.tolist()))))
                best = min(best, c)
            tot += best
    return tot + 0.5 * (float(pos.max()) - float(pos.min()))


def stima_metrica(note: List[Nota], bpm_manuale: Optional[float],
                  tempo_manuale: str, cb=None) -> Metrica:
    import librosa
    env = _inviluppo(note)
    FR = 100
    kw = dict(onset_envelope=env, sr=FR, hop_length=1, units="frames",
              trim=False, tightness=300)
    t0 = min(n.onset for n in note)
    t1 = max(n.offset for n in note)
    on = np.array([n.onset for n in note])
    off = np.array([n.offset for n in note])
    ons_u = np.array(sorted(set(on.tolist())))

    def posizioni(b):
        m_ = Metrica(b, 0.0, 4, 4, F(1), 4, 0)
        return m_.to_pos(on), m_.to_pos(off)

    def sistema(b):
        """raffina, ripara le scivolate, taglia ciò che sta fuori dal brano."""
        if len(b) < 4:
            passo = 60.0 / (bpm_manuale or 100.0)
            b = np.arange(t0, t1 + passo, passo)
            log("Pulsazione non riconoscibile: uso una griglia uniforme.", cb)
        b = _raffina(np.asarray(b, dtype=float), note)
        b = _raffina(_fase_locale(b, note), note)
        b = _regola_bordi(b)
        # prima della prima nota e dopo l'ultima il tracciatore inventa: si tengono
        # solo i movimenti compresi fra il primo e l'ultimo che cadono su un attacco
        vicino = np.array([float(np.min(np.abs(ons_u - x))) <= 0.05 for x in b])
        if vicino.sum() >= 4:
            iv = np.where(vicino)[0]
            b = b[iv[0]: iv[-1] + 1]
        return _estendi(b, t0, t1)

    def raggruppa(b, k):
        """un movimento ogni k, partendo dalla fase più accentata."""
        p_, f_ = posizioni(b)
        fase = _battere(_caratteri(note, p_, f_, len(b)), k)
        return _estendi(b[fase::k], t0, t1)

    # La pulsazione si cerca sempre «al naturale»: è il livello a cui il
    # tracciatore resta più stabile. Il livello giusto si sceglie dopo.
    _, fr = librosa.beat.beat_track(start_bpm=110.0, **kw)
    beats = sistema(np.asarray(fr, dtype=float) / FR)
    # Il tracciatore preferisce pulsazioni vicine a 110: un presto a 160 può
    # agganciarlo a 107 (tre crome per movimento) e il brano uscirebbe in 6/8.
    # Si prova la pulsazione una volta e mezza più rapida e quella più lenta:
    # quando la prima è sbagliata, con l'altra il brano si scrive con molta
    # meno fatica (costo sotto l'80%); quando è giusta, le altre costano il 40%
    # in più. Il margine, sui brani di prova, è netto.
    c_nat = _costo_notazione(note, posizioni(beats)[0])
    nat = 60.0 / float(np.median(np.diff(beats)))
    for fattore in (1.5, 1 / 1.5):
        _, fr = librosa.beat.beat_track(bpm=nat * fattore, **kw)
        alt = sistema(np.asarray(fr, dtype=float) / FR)
        c_alt = _costo_notazione(note, posizioni(alt)[0])
        if c_alt < 0.8 * c_nat:
            beats, c_nat = alt, c_alt
    pos, fine = posizioni(beats)
    periodo = float(np.median(np.diff(beats)))

    if bpm_manuale:
        r = (60.0 / float(bpm_manuale)) / periodo
        if abs(r - 2) < 0.30:
            beats = raggruppa(beats, 2)
        elif abs(r - 3) < 0.40:
            beats = raggruppa(beats, 3)
        elif abs(r - 0.5) < 0.08:
            beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
        elif abs(r - 1) > 0.18:
            # il BPM indicato non è un multiplo semplice di quello trovato:
            # si rifà la ricerca vincolata a quel valore
            _, fr = librosa.beat.beat_track(bpm=float(bpm_manuale), **kw)
            beats = sistema(np.asarray(fr, dtype=float) / FR)
        pos, fine = posizioni(beats)
    else:
        # Distanze di 2/3 di movimento fra note consecutive non esistono nella
        # scrittura normale: sono la firma di terzine lette a tempo doppio.
        n23 = n13 = tot = 0
        for mano in (0, 1):
            pp = np.unique(np.round([p_ for n, p_ in zip(note, pos) if n.hand == mano], 4))
            d = np.diff(pp)
            tot += len(d)
            n23 += int(np.sum(np.abs(d - 2 / 3) < 0.055))
            n13 += int(np.sum(np.abs(d - 1 / 3) < 0.055))
        if n23 >= max(4, 0.05 * tot) and n23 > 2 * n13:
            prove = []
            for f in (0, 1):
                b2 = _estendi(beats[f::2], t0, t1)
                prove.append((_costo_notazione(note, posizioni(b2)[0]), f, b2))
            beats = min(prove, key=lambda x: x[0])[2]
        elif 60.0 / periodo > 144:
            # pulsazione molto rapida: quasi sempre è la suddivisione, non il
            # movimento. A gruppi di tre → tempo composto (6/8); a due → metà tempo.
            car = _caratteri(note, pos, fine, len(beats))
            A = {L: _periodo(car, L) + 1.5 * _somiglianza(car["pc"], L) for L in (2, 3, 4)}
            if A[3] > max(A[2], A[4]) + 0.05:
                if 60.0 / periodo > 168:
                    beats = raggruppa(beats, 3)
            else:
                # …a meno che il brano sia davvero un presto: se a metà tempo le
                # note più rapide non si potessero più scrivere, resta com'è
                meta = raggruppa(beats, 2)
                if _note_perse(note, posizioni(meta)[0]) <= 0.02:
                    beats = meta
        pos, fine = posizioni(beats)

    # in fase o sui levare?
    if _controtempo(note, pos, fine) < -0.5:
        beats = _estendi((beats[:-1] + beats[1:]) / 2, t0, t1)
        pos, fine = posizioni(beats)

    manuale = tempo_manuale if tempo_manuale and tempo_manuale != "auto" else None
    num = den = 0
    composto_forzato = None
    if manuale:
        num, den = (int(x) for x in manuale.split("/"))
        composto_forzato = (den == 8 and num in (6, 9, 12))

    # suddivisione prevalente del movimento: binaria o ternaria?
    def ternario_(pos_):
        fr_ = pos_ - np.floor(pos_)
        vic = lambda c: int(np.sum(np.abs(fr_ - c) < 0.075))
        tern = vic(1 / 3) + vic(2 / 3)
        bina = vic(0.5) + 0.5 * (vic(0.25) + vic(0.75))
        return tern > 2.0 * bina and tern >= 0.12 * len(pos_)

    ternario = ternario_(pos)
    composto = ternario if composto_forzato is None else composto_forzato

    if composto and not ternario and composto_forzato:
        # tempo composto chiesto a mano, ma la pulsazione trovata è quella delle
        # crome: si raggruppano a tre scegliendo la fase più accentata
        car = _caratteri(note, pos, fine, len(beats))
        f = _battere(car, 3)
        beats = _estendi(beats[f::3], t0, t1)
        pos, fine = posizioni(beats)

    nb = len(beats)
    car = _caratteri(note, pos, fine, nb)
    beat_ql = F(3, 2) if composto else F(1)
    A = {L: _periodo(car, L) + 1.5 * _somiglianza(car["pc"], L) for L in (2, 3, 4, 6)}

    if manuale:
        per_bar = num // 3 if composto else num
    elif composto:
        if A[3] > max(A[2], A[4]) + 0.10:
            per_bar = 3
        elif A[4] > A[2] + 0.35:
            per_bar = 4
        else:
            per_bar = 2
        num, den = per_bar * 3, 8
    else:
        per_bar = 3 if A[3] > max(A[2], A[4]) + 0.05 else 4
        num, den = per_bar, 4

    fase = _battere(car, per_bar)
    primo = float(pos.min())
    origin = fase
    while origin > primo + 0.12:
        origin -= per_bar
    while origin + per_bar <= primo + 0.12:
        origin += per_bar

    bpm = 60.0 / float(np.median(np.diff(beats)))
    met = Metrica(beats, bpm, num, den, beat_ql, per_bar, origin)
    pos = met.to_pos(on)
    fine = met.to_pos(off)
    for n, p, e in zip(note, pos, fine):
        n.pos, n.end_pos = float(p), float(e)

    unita = "semiminima puntata" if composto else "semiminima"
    log(f"Tempo: {met.sig}{' (indicato)' if manuale else ' (rilevato)'} | "
        f"{unita} = {bpm:.0f}{' (indicato)' if bpm_manuale else ''}", cb)
    return met


# ─────────────────────────────────────────────────────────────────────────────
# 8. QUANTIZZAZIONE PER MOVIMENTO
# ─────────────────────────────────────────────────────────────────────────────

def quantizza(note: List[Nota], met: Metrica, livello: str, cb=None) -> Dict[int, Dict[int, int]]:
    """
    Per ogni mano e per ogni movimento si sceglie UNA suddivisione (1, 2, 3, 4,
    6, 8) minimizzando  errore + lam·complessità + collisioni.  Tutti gli
    attacchi del movimento vanno su quella griglia: binario e terzinato non si
    mescolano mai dentro lo stesso movimento, quindi ogni durata è scrivibile.
    Ritorna, per mano, la suddivisione scelta in ogni movimento.
    """
    composto = met.beat_ql != 1
    prof = QUANT[livello]
    pen = QUANT_COMP[livello] if composto else prof["pen"]
    lam = prof["lam"]
    divs: Dict[int, Dict[int, int]] = {0: {}, 1: {}}

    for mano in (0, 1):
        hs = [n for n in note if n.hand == mano]
        per_beat: Dict[int, Dict[float, List[Nota]]] = {}
        for n in hs:
            k = max(int(math.floor(n.pos + 0.07)), 0)
            per_beat.setdefault(k, {}).setdefault(round(n.pos, 5), []).append(n)

        for k, gruppi in per_beat.items():
            fr = np.array(sorted(gruppi.keys())) - k
            best = None
            for d, pd in pen.items():
                snap = np.clip(np.round(fr * d), 0, d)
                err = float(np.abs(fr - snap / d).sum())
                coll = len(snap) - len(set(snap.tolist()))
                c = err + lam * pd + COLLISION * coll
                if best is None or c < best[0] - 1e-9:
                    best = (c, d, snap)
            _, d, snap = best
            divs[mano][k] = d
            for f, j in zip(sorted(gruppi.keys()), snap):
                q_beats = F(k) + F(int(j), d)
                for n in gruppi[f]:
                    n.q = q_beats * met.beat_ql
    primo = min((n.q for n in note), default=F(0))
    if primo < 0:                                # sicurezza: niente posizioni negative
        sp = (-primo // met.bar_ql + (1 if (-primo) % met.bar_ql else 0)) * met.bar_ql
        for n in note:
            n.q += sp
        for mano in (0, 1):
            divs[mano] = {k + int(sp / met.beat_ql): v for k, v in divs[mano].items()}
    usate: Dict[int, int] = {}
    for mano in (0, 1):
        for d in divs[mano].values():
            usate[d] = usate.get(d, 0) + 1
    log(f"Quantizzazione «{livello}» — suddivisioni usate: "
        + ", ".join(f"{d}→{c}" for d, c in sorted(usate.items())), cb)
    return divs


# ─────────────────────────────────────────────────────────────────────────────
# 9. TONALITÀ E NOMI DELLE NOTE
# ─────────────────────────────────────────────────────────────────────────────

_KK_MAG = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOMI_IT = ["Do", "Do♯", "Re", "Mi♭", "Mi", "Fa", "Fa♯", "Sol", "La♭", "La", "Si♭", "Si"]


def stima_tonalita(note: List[Nota]) -> Tuple[int, str, str]:
    """Krumhansl-Schmuckler. Ritorna (diesis, modo, nome)."""
    h = np.zeros(12)
    for n in note:
        h[n.pitch % 12] += min(n.offset - n.onset, 2.0) + 0.1
    best = (-9.0, 0, "major")
    for t in range(12):
        for modo, prof in (("major", _KK_MAG), ("minor", _KK_MIN)):
            r = float(np.corrcoef(h, np.roll(prof, t))[0, 1])
            if r > best[0]:
                best = (r, t, modo)
    _, t, modo = best
    rel = t if modo == "major" else (t + 3) % 12
    diesis = next(x for x in range(-5, 7) if (7 * x) % 12 == rel)
    nome = f"{_NOMI_IT[t]} {'maggiore' if modo == 'major' else 'minore'}"
    return diesis, modo, nome


def nome_nota(midi: int, diesis: int, modo: str) -> str:
    """Alterazioni coerenti con la tonalità, lungo la linea delle quinte."""
    lo = diesis - (3 if modo == "minor" else 4)
    pc = midi % 12
    x = next(v for v in range(lo, lo + 12) if (7 * v) % 12 == pc)
    lettera = "FCGDAEB"[(x + 1) % 7]
    alt = (x + 1) // 7
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[lettera]
    ottava = (midi - base - alt) // 12 - 1
    segno = {0: "", 1: "#", 2: "##", -1: "-", -2: "--"}[alt]
    return f"{lettera}{segno}{ottava}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. EVENTI: accordi, durate, staccati
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Evento:
    inizio: F
    fine: F
    note: List[Nota]                 # vuoto = pausa
    stacc: bool = False


def _griglia_ql(q: F, mano_divs: Dict[int, int], met: Metrica) -> F:
    """passo di griglia (in quarti) del movimento che contiene q."""
    k = int(q / met.beat_ql)
    return met.beat_ql / mano_divs.get(k, 1)


def _snap_fine(t: F, mano_divs, met: Metrica) -> F:
    k = int(t / met.beat_ql)
    passo = met.beat_ql / mano_divs.get(k, 1)
    base = k * met.beat_ql
    return base + round((t - base) / passo) * passo


def costruisci_eventi(note: List[Nota], met: Metrica, divs, pedali,
                      staccati: bool, due_voci: bool = True):
    """
    Ritorna (voce principale, seconda voce), ciascuna un dizionario per mano.

    Seconda voce: una nota che resta tenuta mentre la stessa mano suona altro
    (il basso sotto gli accordi, il canto sopra l'accompagnamento) non può
    stare nella stessa voce delle note che si muovono, o verrebbe troncata al
    primo attacco successivo. Va in una voce a parte, con la sua durata vera.

    Voce principale: una sequenza di accordi e pause senza sovrapposizioni.
    Durata di un accordo:
      - se è tenuto (o il pedale lo sostiene) fino al successivo → riempie tutto;
      - se è rilasciato molto presto e il valore è breve → staccato;
      - altrimenti → durata reale arrotondata, e una pausa per il resto.
    """
    ped = sorted(pedali)
    ped_on = np.array([a for a, _ in ped]) if ped else np.zeros(0)
    ped_off = np.array([b for _, b in ped]) if ped else np.zeros(0)

    def pedale_giu(t: float) -> bool:
        if not len(ped_on):
            return False
        i = int(np.searchsorted(ped_on, t, "right")) - 1
        return i >= 0 and ped_off[i] > t + 0.03

    def fine_q(n: Nota) -> F:
        """fine della nota in quarti, non ancora arrotondata."""
        return n.q + F(n.end_pos - n.pos).limit_denominator(96) * met.beat_ql

    out: Dict[int, List[Evento]] = {}
    out2: Dict[int, List[Evento]] = {}
    for mano in (0, 1):
        acc: Dict[F, List[Nota]] = {}
        for n in note:
            if n.hand == mano:
                acc.setdefault(n.q, []).append(n)
        tempi = sorted(acc)

        # ── note tenute sotto (o sopra) altre che si muovono → seconda voce
        tenute: Dict[F, List[Nota]] = {}
        if due_voci:
            libera_da = F(0)
            for i, q in enumerate(tempi[:-1]):
                ns = acc[q]
                if q < libera_da or any(n.arp or n.orn or n.graces for n in ns):
                    continue
                prossimo = tempi[i + 1]
                gap = prossimo - q
                # tenuta davvero oltre l'attacco successivo: almeno un passo di
                # griglia dopo l'arrotondamento (un legato di dita non basta),
                # lunga almeno mezzo movimento e una volta e mezza l'intervallo
                passo_p = _griglia_ql(prossimo, divs[mano], met)

                def arrot(n):
                    return _snap_fine(fine_q(n), divs[mano], met)

                lunghe = [n for n in ns
                          if arrot(n) >= prossimo + passo_p
                          and arrot(n) - q >= max(met.beat_ql / 2, gap * F(3, 2))]
                if not lunghe:
                    continue
                tenute[q] = lunghe
                libera_da = _snap_fine(max(fine_q(n) for n in lunghe), divs[mano], met)
                resto = [n for n in ns if all(n is not x for x in lunghe)]
                if resto:
                    acc[q] = resto
                else:
                    del acc[q]
            tempi = sorted(acc)
        ev2: List[Evento] = []
        cur2 = F(0)
        tq = sorted(tenute)
        for i, q in enumerate(tq):
            fine = _snap_fine(max(fine_q(n) for n in tenute[q]), divs[mano], met)
            if i + 1 < len(tq):
                fine = min(fine, tq[i + 1])
            fine = max(fine, q + _griglia_ql(q, divs[mano], met))
            if q > cur2:
                ev2.append(Evento(cur2, q, []))
            ev2.append(Evento(q, fine, sorted(tenute[q], key=lambda n: n.pitch)))
            cur2 = fine
        out2[mano] = ev2

        ev: List[Evento] = []
        cursore = F(0)
        for i, q in enumerate(tempi):
            gruppo = {}
            for n in acc[q]:                           # stesso tasto due volte: tieni il più lungo
                if n.pitch not in gruppo or n.offset > gruppo[n.pitch].offset:
                    if n.pitch in gruppo:
                        n.graces = n.graces or gruppo[n.pitch].graces
                        n.orn = n.orn or gruppo[n.pitch].orn
                    gruppo[n.pitch] = n
            ns = sorted(gruppo.values(), key=lambda n: n.pitch)
            pos0 = float(np.mean([n.pos for n in ns]))
            fine_cont = max(n.end_pos for n in ns)
            suonata = F(fine_cont - pos0).limit_denominator(96) * met.beat_ql
            t_ril = max(n.offset for n in ns)
            passo = _griglia_ql(q, divs[mano], met)

            ornato = any(n.orn for n in ns)
            stacc = False
            if i + 1 < len(tempi):
                gap = tempi[i + 1] - q
                rapporto = float(suonata / gap) if gap else 1.0
                tenuta = ornato or rapporto >= 0.62 or pedale_giu(t_ril)
                breve = _snap_fine(q + suonata, divs[mano], met)
                breve = max(q + passo, min(breve, tempi[i + 1]))
                if tenuta:
                    fine = tempi[i + 1]                # riempie fino al prossimo attacco
                elif gap <= met.beat_ql and rapporto < 0.30:
                    fine = tempi[i + 1]                # valore breve, suonato corto: staccato
                    stacc = staccati
                elif gap <= met.beat_ql and breve - q > gap / 2:
                    fine = tempi[i + 1]                # non legato, ma non c'è una pausa vera
                else:                                  # la nota dura metà dell'intervallo o meno:
                    fine = breve                       # nota + pausa
            else:
                meta = met.beat_ql / 2
                dur = max(meta, round(suonata / meta) * meta)
                fine = q + min(dur, 2 * met.bar_ql)
                stanghetta = -((-fine) // met.bar_ql) * met.bar_ql
                if stanghetta - fine <= met.beat_ql:   # accordo finale: fino alla stanghetta
                    fine = stanghetta
            if q > cursore:
                ev.append(Evento(cursore, q, []))
            if fine <= q:
                fine = q + passo
            ev.append(Evento(q, fine, ns, stacc))
            cursore = fine
        out[mano] = ev

    # tutte le voci devono finire sulla stessa stanghetta
    fine_tot = max((e[-1].fine for d in (out, out2) for e in d.values() if e),
                   default=met.bar_ql)
    nbar = int(math.ceil(fine_tot / met.bar_ql))
    fine_tot = nbar * met.bar_ql
    for d in (out, out2):
        for mano in (0, 1):
            ev = d[mano]
            if d is out2 and not any(e.note for e in ev):
                d[mano] = []
                continue
            cur = ev[-1].fine if ev else F(0)
            if cur < fine_tot:
                ev.append(Evento(cur, fine_tot, []))
    return out, out2


# ─────────────────────────────────────────────────────────────────────────────
# 11. SCOMPOSIZIONE IN VALORI SCRIVIBILI
# ─────────────────────────────────────────────────────────────────────────────

_VALORI = [F(4), F(3), F(2), F(3, 2), F(1), F(3, 4), F(1, 2), F(3, 8), F(1, 4),
           F(3, 16), F(1, 8), F(1, 16), F(1, 32)]


def _semplici(a: F, b: F, fino_al_battere: bool) -> List[Tuple[F, F]]:
    """Scompone [a,b) in valori ordinari (anche puntati)."""
    resto = b - a
    vals = []
    while resto > 0:
        v = next((x for x in _VALORI if x <= resto), None)
        if v is None:
            break
        vals.append(v)
        resto -= v
    if fino_al_battere:
        vals.reverse()                 # il valore piccolo prima: si allinea al movimento
    out, t = [], a
    for v in vals:
        out.append((t, t + v))
        t += v
    return out


def spezza(a: F, b: F, mano_divs: Dict[int, int], met: Metrica) -> List[Tuple[F, F]]:
    """
    Divide [a,b) in pezzi che esistono come figure musicali:
      - mai a cavallo di una stanghetta;
      - i movimenti interi si fondono in minime, minime puntate, semibrevi;
      - dentro un movimento terzinato i pezzi sono terzi (o sesti) di movimento.
    I pezzi vanno poi uniti da legature di valore.
    """
    bq, bar = met.beat_ql, met.bar_ql
    pezzi: List[Tuple[F, F]] = []

    def dentro(x: F, y: F, k: int):
        """[x,y) sta tutto nel movimento k."""
        d = mano_divs.get(k, 1)
        base = k * bq
        if bq == 1 and d in (3, 6):
            celle = [(base, base + 1)] if d == 3 else [(base, base + F(1, 2)),
                                                       (base + F(1, 2), base + 1)]
            for c0, c1 in celle:
                s, e = max(x, c0), min(y, c1)
                if s < e:
                    pezzi.append((s, e))       # 1 o 2 unità di terzina, o la cella intera
        else:
            pezzi.extend(_semplici(x, y, fino_al_battere=(y == base + bq and x != base)))

    t = a
    while t < b:
        fine_bar = (t // bar + 1) * bar
        seg_b = min(b, fine_bar)
        A = -((-t) // bq) * bq                 # primo movimento ≥ t
        B = (seg_b // bq) * bq                 # ultimo movimento ≤ seg_b
        if A > B:                              # tutto dentro un solo movimento
            dentro(t, seg_b, int(t // bq))
        else:
            if t < A:
                dentro(t, A, int(t // bq))
            if A < B:
                inizio_bar = (A // bar) * bar
                p = int((A - inizio_bar) / bq)
                n = int((B - A) / bq)
                x = A
                while n > 0:
                    L = 1
                    if n >= 4 and p % 4 == 0 and met.per_bar >= 4:
                        L = 4
                    elif (n >= 3 and bq == 1 and met.per_bar in (3, 4)
                          and (p == 0 or p + 3 == met.per_bar)):
                        L = 3
                    elif n >= 2 and (p % 2 == 0 or met.per_bar == 3):
                        L = 2
                    pezzi.append((x, x + L * bq))
                    x += L * bq
                    p += L
                    n -= L
            if B < seg_b:
                dentro(B, seg_b, int(B // bq))
        t = seg_b
    out, cur = [], a
    for s_, e_ in pezzi:
        if s_ == cur and e_ > s_:
            out.append((s_, e_))
            cur = e_
    if cur != b:                               # rete di sicurezza: non perdere mai durata
        out.extend(_semplici(cur, b, False))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 12. PARTITURA (music21, misure costruite a mano)
# ─────────────────────────────────────────────────────────────────────────────

_DIN = [(34, "pp"), (50, "p"), (66, "mp"), (82, "mf"), (100, "f"), (128, "ff")]


def _dinamica(v: float) -> str:
    return next(nome for soglia, nome in _DIN if v < soglia)


def costruisci_partitura(eventi: Dict[int, List[Evento]], divs, met: Metrica,
                         tonalita: Tuple[int, str, str], titolo: str, autore: str,
                         pedali, opz: Dict, cb=None, eventi2=None):
    from music21 import (stream, note as m21n, chord as m21c, meter as m21m,
                         tempo as m21t, metadata, layout, clef, expressions,
                         duration, key as m21k, tie, articulations, dynamics, bar)

    diesis, modo, _ = tonalita
    bar_ql = met.bar_ql
    nbar = int(max(e[-1].fine for e in eventi.values() if e) / bar_ql)

    sc = stream.Score()
    sc.insert(0, metadata.Metadata())
    sc.metadata.title = titolo or "Senza titolo"
    sc.metadata.composer = autore or ""

    parti = []
    primi: Dict[int, Dict[int, object]] = {0: {}, 1: {}}      # nota iniziale per battuta
    tutti: Dict[int, List[Tuple[F, F, object]]] = {0: [], 1: []}

    for mano in (0, 1):
        part = stream.PartStaff()
        part.id = "destra" if mano == 0 else "sinistra"
        misure = []
        for b in range(nbar):
            m = stream.Measure(number=b + 1)
            if b == 0:
                m.insert(0, clef.TrebleClef() if mano == 0 else clef.BassClef())
                m.insert(0, m21k.KeySignature(diesis))
                m.insert(0, m21m.TimeSignature(met.sig))
                if mano == 0:
                    ref = duration.Duration(float(met.beat_ql))
                    m.insert(0, m21t.MetronomeMark(number=int(round(met.bpm)), referent=ref))
            misure.append(m)

        def elementi(lista, seconda: bool):
            """Trasforma una voce in elementi music21, battuta per battuta."""
            per_b: List[list] = [[] for _ in range(nbar)]
            for ev in lista:
                pezzi = spezza(ev.inizio, ev.fine, divs[mano], met)
                for i, (s, e) in enumerate(pezzi):
                    b = int(s // bar_ql)
                    if b >= nbar:
                        continue
                    d = duration.Duration(quarterLength=e - s)
                    if not ev.note:
                        el = m21n.Rest()
                        el.duration = d
                        if seconda:                    # le pause della seconda voce non si stampano
                            el.style.hideObjectOnPrint = True
                    else:
                        nomi = [nome_nota(n.pitch, diesis, modo) for n in ev.note]
                        if len(nomi) == 1:
                            el = m21n.Note(nomi[0])
                        else:
                            el = m21c.Chord(nomi)
                        el.duration = d
                        el.volume.velocity = int(np.mean([n.vel for n in ev.note]))
                        if len(pezzi) > 1:
                            tipo = "start" if i == 0 else ("stop" if i == len(pezzi) - 1 else "continue")
                            if len(nomi) == 1:
                                el.tie = tie.Tie(tipo)
                            else:
                                for cn in el.notes:
                                    cn.tie = tie.Tie(tipo)
                        if i == 0:
                            capo = ev.note
                            orn = next((n.orn for n in capo if n.orn), None)
                            if orn == "trill":
                                el.expressions.append(expressions.Trill())
                            elif orn == "turn":
                                el.expressions.append(expressions.Turn())
                            elif orn == "mordent":
                                el.expressions.append(expressions.Mordent())
                            elif orn == "inv_mordent":
                                el.expressions.append(expressions.InvertedMordent())
                            if len(capo) >= 2 and sum(1 for n in capo if n.arp) >= 2:
                                el.expressions.append(expressions.ArpeggioMark("normal"))
                            if ev.stacc and len(pezzi) == 1:
                                el.articulations.append(articulations.Staccato())
                            for gp in (g for n in capo for g in n.graces):
                                g = m21n.Note(nome_nota(gp, diesis, modo))
                                g.duration = duration.Duration("eighth")
                                per_b[b].append(g.getGrace())
                        if opz.get("rosso", True) and any(not n.verified for n in ev.note):
                            el.style.color = "red"
                            if len(nomi) > 1:
                                for cn in el.notes:
                                    cn.style.color = "red"
                    per_b[b].append(el)
                    if ev.note:
                        tutti[mano].append((s, e, el))
                        if not seconda:
                            primi[mano].setdefault(b, el)
            return per_b

        v1 = elementi(eventi[mano], False)
        v2 = elementi(eventi2.get(mano) or [], True) if eventi2 else [[] for _ in range(nbar)]
        for b in range(nbar):
            ha_note2 = any(getattr(x, "isNote", False) or getattr(x, "isChord", False) for x in v2[b])
            if not ha_note2:
                for x in v1[b]:
                    misure[b].append(x)
                continue
            # due voci: la più acuta coi gambi in su, l'altra in giù
            alt = lambda xs: [p.midi for x in xs if not x.isRest for p in x.pitches]
            a1, a2 = alt(v1[b]), alt(v2[b])
            su1 = (np.mean(a1) if a1 else 0) >= (np.mean(a2) if a2 else 0)
            for xs, su, num in ((v1[b], su1, "1"), (v2[b], not su1, "2")):
                voce = stream.Voice()
                voce.id = num
                for x in xs:
                    if not x.isRest:
                        x.stemDirection = "up" if su else "down"
                    voce.append(x)
                misure[b].insert(0, voce)

        for m in misure:
            part.append(m)
        misure[-1].rightBarline = bar.Barline("final")
        parti.append(part)

    # ── dinamiche: una per battuta, solo quando il livello cambia davvero
    if opz.get("dinamiche", True):
        vol: Dict[int, List[int]] = {}
        for mano in (0, 1):
            for ev in eventi[mano]:
                for n in ev.note:
                    vol.setdefault(int(ev.inizio // bar_ql), []).append(n.vel)
        livelli = []
        for b in range(nbar):
            v = sorted(vol.get(b, []))
            livelli.append(float(np.mean(v[len(v) // 2:])) if v else None)
        liscio = []
        for b in range(nbar):
            fin = [x for x in livelli[max(0, b - 1): b + 2] if x is not None]
            liscio.append(float(np.median(fin)) if fin else None)
        attuale = None
        for b in range(nbar):
            if liscio[b] is None:
                continue
            nome = _dinamica(liscio[b])
            succ = next((liscio[k] for k in range(b + 1, nbar) if liscio[k] is not None), liscio[b])
            if nome != attuale and (attuale is None or _dinamica(succ) == nome):
                try:
                    parti[0].getElementsByClass(stream.Measure)[b].insert(0, dynamics.Dynamic(nome))
                    attuale = nome
                except Exception:
                    pass

    # ── pedale di risonanza: da nota a nota della mano sinistra (o destra)
    n_ped = 0
    if opz.get("pedale", True) and pedali and hasattr(expressions, "PedalMark"):
        elenco = sorted(tutti[1] or tutti[0], key=lambda x: x[0])
        inizi = np.array([float(s) for s, _, _ in elenco])
        for a, b_ in sorted(pedali):
            if b_ - a < 0.25:
                continue
            qa = F(float(met.to_pos(a))).limit_denominator(48) * met.beat_ql
            qb = F(float(met.to_pos(b_))).limit_denominator(48) * met.beat_ql
            i0 = int(np.searchsorted(inizi, float(qa) - 0.26, "left"))
            i1 = int(np.searchsorted(inizi, float(qb) - 0.05, "left")) - 1
            if i0 >= len(elenco) or i1 <= i0:
                continue
            try:
                pm = expressions.PedalMark(elenco[i0][2], elenco[i1][2])
                (parti[1] if tutti[1] else parti[0]).insert(0, pm)
                n_ped += 1
            except Exception:
                break

    for part in parti:
        for m in part.getElementsByClass(stream.Measure):
            for cont in (list(m.voices) or [m]):
                try:
                    stream.makeNotation.makeTupletBrackets(cont, inPlace=True)
                except Exception:
                    pass
        try:
            part.makeBeams(inPlace=True)
        except Exception:
            pass
        try:
            part.makeAccidentals(inPlace=True)
        except Exception:
            pass
        sc.insert(0, part)
    sc.insert(0, layout.StaffGroup(parti, symbol="brace", barTogether=True))
    log(f"Partitura: {nbar} battute" + (f", {n_ped} segni di pedale" if n_ped else ""), cb)
    return sc


def controlla_misure(sc, met: Metrica) -> List[str]:
    """Ogni battuta deve contenere esattamente la sua durata. Ritorna gli errori."""
    from music21 import stream
    err = []
    for part in sc.parts:
        for m in part.getElementsByClass(stream.Measure):
            for cont in (list(m.voices) or [m]):
                tot = sum((F(el.duration.quarterLength).limit_denominator(96)
                           for el in cont.notesAndRests if not el.duration.isGrace), F(0))
                if tot != met.bar_ql:
                    err.append(f"{part.id} battuta {m.number}: {tot} invece di {met.bar_ql}")
    return err


# ─────────────────────────────────────────────────────────────────────────────
# 13. USCITA: MusicXML, MIDI, PDF
# ─────────────────────────────────────────────────────────────────────────────

MUSESCORE_PATHS = [
    os.environ.get("MUSESCORE_PATH", ""),
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/usr/bin/mscore", "/usr/bin/musescore",
]


def find_musescore() -> Optional[str]:
    for p in MUSESCORE_PATHS:
        if p and os.path.exists(p):
            return p
    for name in ("mscore", "musescore", "MuseScore4", "MuseScore3"):
        w = shutil.which(name)
        if w:
            return w
    if sys.platform == "win32":
        for root in (os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
                     os.environ.get("ProgramFiles", r"C:\Program Files")):
            if not root or not os.path.isdir(root):
                continue
            try:
                for entry in os.listdir(root):
                    if "musescore" in entry.lower():
                        for sub in ("bin", ""):
                            d = os.path.join(root, entry, sub)
                            if os.path.isdir(d):
                                for f in os.listdir(d):
                                    if f.lower().startswith("musescore") and f.lower().endswith(".exe"):
                                        return os.path.join(d, f)
            except Exception:
                pass
    return None


def is_real_pdf(path: str) -> bool:
    try:
        if os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def _xml_per_verovio(xml_path: str) -> str:
    """
    Toglie l'indicazione metronomica dal MusicXML passato a Verovio: la nota
    del metronomo è un carattere di un font musicale che nel PDF non esiste.
    Il tempo viene scritto da noi nell'intestazione. Il file su disco resta
    completo, per MuseScore e gli altri programmi.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for parent in root.iter():
        for d in list(parent):
            if d.tag == "direction" and d.find("./direction-type/metronome") is not None:
                parent.remove(d)
    for tag in ("work", "movement-title", "identification", "credit"):
        for el in list(root.findall(tag)):
            if tag == "identification":
                for c in list(el.findall("creator")):
                    el.remove(c)
            else:
                root.remove(el)
    return ET.tostring(root, encoding="unicode")


def _svg_piatto(svg: str) -> str:
    """
    Verovio annida un secondo <svg> con le coordinate interne; svglib lo
    converte applicando due volte il rapporto pixel/punti, e la musica occupa
    solo tre quarti della pagina. Qui il riquadro interno diventa un gruppo e
    le sue coordinate passano all'elemento esterno.
    """
    m = re.search(r'<svg class="definition-scale"([^>]*?)viewBox="([^"]+)"([^>]*)>', svg)
    if not m:
        return svg
    interno = m.group(0)
    svg = svg.replace(interno, '<g class="definition-scale"' + m.group(1) + m.group(3) + '>', 1)
    fine = svg.rfind("</svg>", 0, svg.rfind("</svg>"))
    svg = svg[:fine] + "</g>" + svg[fine + len("</svg>"):]
    testa = re.search(r"<svg[^>]*>", svg).group(0)
    nuova = re.sub(r'\sviewBox="[^"]*"', "", testa)
    nuova = nuova[:-1] + f' viewBox="{m.group(2)}">'
    return svg.replace(testa, nuova, 1)


class _CatturaStderr:
    """Raccoglie ciò che una libreria C++ scrive sullo standard error."""

    def __enter__(self):
        self.testo = ""
        self._f = self._vecchio = None
        try:
            sys.stderr.flush()
            self._f = tempfile.TemporaryFile(mode="w+b")
            self._vecchio = os.dup(2)
            os.dup2(self._f.fileno(), 2)
        except Exception:
            self._vecchio = None
        return self

    def __exit__(self, *a):
        try:
            if self._vecchio is not None:
                os.dup2(self._vecchio, 2)
                os.close(self._vecchio)
                self._f.seek(0)
                self.testo = self._f.read().decode("utf-8", "replace").strip()
            if self._f:
                self._f.close()
        except Exception:
            pass
        return False


def _carica_in_verovio(tk, xml_path: str, cb=None) -> str:
    """
    Carica il MusicXML in Verovio provando più strade, e registra perché una
    strada fallisce. Su Windows il caricamento diretto può essere rifiutato
    (impostazioni locali del sistema, rilevamento del formato): le alternative
    passano da un file su disco e dal formato dichiarato esplicitamente.
    Ritorna il nome della strada riuscita, oppure stringa vuota.
    """
    import locale
    dati = _xml_per_verovio(xml_path)
    try:
        vecchia = locale.setlocale(locale.LC_ALL)
        locale.setlocale(locale.LC_ALL, "C")        # numeri col punto, non con la virgola
    except Exception:
        vecchia = None
    tmpdir = tempfile.mkdtemp(prefix="partitura_vrv_")
    tmpxml = os.path.join(tmpdir, "partitura.musicxml")
    try:
        with open(tmpxml, "w", encoding="utf-8", newline="\n") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + dati)

        def formato():
            tk.setInputFrom("musicxml")

        strade = [
            ("dati in memoria", lambda: tk.loadData(dati)),
            ("dati in memoria, formato dichiarato", lambda: (formato(), tk.loadData(dati))[1]),
            ("file temporaneo", lambda: (formato(), tk.loadFile(tmpxml))[1]),
            ("file originale", lambda: (formato(), tk.loadFile(xml_path))[1]),
        ]
        try:
            verovio_mod = sys.modules.get("verovio")
            verovio_mod.enableLog(verovio_mod.LOG_WARNING)
        except Exception:
            pass
        for nome, prova in strade:
            try:
                tk.setLocale()
            except Exception:
                pass
            with _CatturaStderr() as cat:
                try:
                    ok = bool(prova())
                    err = ""
                except Exception as e:
                    ok, err = False, f"{type(e).__name__}: {e}"
            if ok and tk.getPageCount() > 0:
                if nome != strade[0][0]:
                    log(f"Verovio: caricato per la via «{nome}».", cb)
                return nome
            motivo = " | ".join(x for x in (err, cat.testo[-400:].replace("\n", " / ")) if x)
            log(f"Verovio: via «{nome}» rifiutata. {motivo or '(nessun messaggio)'}", cb)
        return ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            tk.resetLocale()
        except Exception:
            pass
        if vecchia:
            try:
                locale.setlocale(locale.LC_ALL, vecchia)
            except Exception:
                pass


def pdf_in_altro_processo(xml_path: str, out_pdf: str, titolo: str, autore: str,
                          sottotitolo: str, cb=None) -> bool:
    """
    Ultima risorsa: impagina in un processo Python nuovo, senza finestra e
    senza i moduli della trascrizione caricati. È la condizione in cui gira
    la prova a secco dell'installatore.
    """
    log("Riprovo l'impaginazione in un processo separato...", cb)
    try:
        res = subprocess.run(
            ([sys.executable] if CONGELATO else [sys.executable, os.path.abspath(__file__)])
            + ["--impagina",
             xml_path, out_pdf, titolo, autore, sottotitolo],
            capture_output=True, text=True, timeout=1800, encoding="utf-8",
            errors="replace", env={**os.environ, "PYTHONUTF8": "1"})
    except Exception as e:
        log(f"Processo separato non avviato ({type(e).__name__}: {e})", cb)
        return False
    for r in (res.stdout + res.stderr).strip().splitlines()[-12:]:
        log("  [processo separato] " + r.strip(), cb)
    return is_real_pdf(out_pdf)


def pdf_verovio(xml_path: str, out_pdf: str, titolo: str, autore: str,
                sottotitolo: str, cb=None) -> bool:
    """MusicXML → SVG (Verovio) → PDF (reportlab). Nessun programma esterno."""
    import io
    import verovio
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    log("Impagino la partitura (Verovio)...", cb)
    # Verovio ricorda la cartella dei suoi font PER THREAD: il modulo viene
    # importato nel thread principale, ma dalla finestra l'impaginazione gira in
    # un thread secondario, che quella cartella non la conosce («font resources
    # are not available»). Va quindi indicata di nuovo qui, prima e dopo aver
    # creato il motore.
    risorse = os.path.join(os.path.dirname(verovio.__file__), "data")
    try:
        verovio.setDefaultResourcePath(risorse)
    except Exception:
        pass
    tk = verovio.toolkit()
    if os.path.isdir(risorse) and not tk.setResourcePath(risorse):
        log(f"Verovio: font non caricati da {risorse}", cb)
    W, H = A4
    testa1, testa = 92.0, 34.0                    # punti riservati in alto
    piede = 26.0
    tk.setOptions({
        "pageWidth": 2100, "pageHeight": 2760, "scale": 38,
        "pageMarginTop": 60, "pageMarginBottom": 60,
        "pageMarginLeft": 110, "pageMarginRight": 110,
        "header": "none", "footer": "none", "breaks": "auto",
        "adjustPageHeight": False, "spacingStaff": 10, "spacingSystem": 14,
        "justifyVertically": False, "svgViewBox": False,
    })
    esito = _carica_in_verovio(tk, xml_path, cb)
    if not esito:
        raise RuntimeError("Verovio non ha letto il MusicXML (dettagli nel registro).")
    npag = tk.getPageCount()
    c = canvas.Canvas(out_pdf, pagesize=A4)
    c.setTitle(titolo or "Partitura")
    c.setAuthor(autore or "")
    # Tutto in memoria: su Windows un file temporaneo riscritto a ogni pagina
    # può restare bloccato e far fallire l'impaginazione.
    for p in range(1, npag + 1):
        d = svg2rlg(io.BytesIO(_svg_piatto(tk.renderToSVG(p)).encode("utf-8")))
        if d is None:
            raise RuntimeError(f"pagina {p}: SVG non convertibile")
        alto = testa1 if p == 1 else testa
        sx = W / d.width
        sy = (H - alto - piede) / d.height
        s = min(sx, sy)
        d.scale(s, s)
        renderPDF.draw(d, c, (W - d.width * s) / 2, piede + (H - alto - piede - d.height * s))
        if p == 1:
            c.setFont("Times-Bold", 22)
            c.drawCentredString(W / 2, H - 48, (titolo or "Senza titolo")[:80])
            c.setFont("Times-Italic", 11)
            if autore:
                c.drawRightString(W - 44, H - 72, autore[:70])
            c.setFont("Times-Roman", 10)
            c.drawString(44, H - 72, sottotitolo)
        else:
            c.setFont("Times-Italic", 9)
            c.drawString(44, H - 24, (titolo or "")[:90])
        c.setFont("Times-Roman", 9)
        c.drawCentredString(W / 2, 14, f"{p} / {npag}")
        c.showPage()
    c.save()
    return is_real_pdf(out_pdf)


def pdf_musescore(xml_path: str, out_pdf: str, cb=None) -> bool:
    ms = find_musescore()
    if not ms:
        return False
    log(f"Impagino con MuseScore ({os.path.basename(ms)})...", cb)
    try:
        subprocess.run([ms, "-o", out_pdf, xml_path], capture_output=True,
                       text=True, timeout=900, errors="replace")
    except Exception as e:
        log(f"MuseScore non ha risposto ({e}).", cb)
        return False
    if not is_real_pdf(out_pdf):
        try:
            os.remove(out_pdf)
        except OSError:
            pass
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# NUCLEO: dalle note alla partitura (usato anche dal banco di prova)
# ─────────────────────────────────────────────────────────────────────────────

def note_in_partitura(note: List[Nota], pedali, titolo: str = "", autore: str = "",
                      livello: str = "media", bpm: Optional[float] = None,
                      tempo: str = "auto", opz: Optional[Dict] = None, cb=None) -> Dict:
    opz = opz or {}
    note = pulisci(note)
    if not note:
        raise RuntimeError("Nessuna nota da trascrivere.")
    trova_arpeggiati(note, pedali)
    allinea_accordi(note)
    separa_mani(note, cb)
    met0 = stima_metrica(note, bpm, tempo, None)   # griglia preliminare: serve agli abbellimenti
    note, conta = trova_abbellimenti(note, cb, met0)
    met = stima_metrica(note, bpm, tempo, cb)
    divs = quantizza(note, met, livello, cb)
    ton = stima_tonalita(note)
    log(f"Tonalità: {ton[2]}", cb)
    eventi, eventi2 = costruisci_eventi(note, met, divs, pedali, opz.get("staccati", True),
                                        opz.get("voci", True))
    sc = costruisci_partitura(eventi, divs, met, ton, titolo, autore, pedali, opz, cb,
                              eventi2=eventi2)
    errori = controlla_misure(sc, met)
    if errori:
        log(f"ATTENZIONE: {len(errori)} battute con durata sbagliata "
            f"(prima: {errori[0]})", cb)
    return {"score": sc, "note": note, "metrica": met, "divs": divs,
            "eventi": eventi, "eventi2": eventi2, "tonalita": ton, "abbellimenti": conta,
            "errori_misure": errori}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETA
# ─────────────────────────────────────────────────────────────────────────────

def _nome_file(titolo: str) -> str:
    s = re.sub(r"[^\w\s-]", "", titolo, flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", s)[:80] or "partitura"


def run_pipeline(source: str, out_dir: str, title: str = "", composer: str = "",
                 quant: str = "media", bpm: Optional[float] = None,
                 meter: str = "auto", motore: str = "auto",
                 opz: Optional[Dict] = None, cb=None) -> Dict:
    opz = {"rosso": True, "dinamiche": True, "pedale": True, "staccati": True,
           "musescore": False, **(opz or {})}
    # Tutto ciò che il programma dice finisce anche in un file accanto alle
    # partiture: se qualcosa va storto, la causa è scritta lì.
    righe: List[str] = []
    cb_utente = cb

    def cb(line):                                    # noqa: F811
        righe.append(line)
        if cb_utente:
            cb_utente(line)

    log(f"PARTITURA v{VERSIONE} — {os.path.abspath(__file__)}", cb)
    os.makedirs(out_dir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="partitura_")
    try:
        # 1 ─ ingresso
        audio_path = None
        if is_youtube(source):
            audio_path, yt_title = fetch_youtube(source, workdir, cb)
            title = title or (yt_title or "")
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"File non trovato: {source}")
            title = title or os.path.splitext(os.path.basename(source))[0]
            if not is_midi(source):
                audio_path = source

        # 2 ─ trascrizione
        if audio_path:
            note, pedali, usato = trascrivi(audio_path, motore, workdir, cb)
        else:
            note, pedali = leggi_midi(source, cb)
            usato = "midi"
        if not note:
            raise RuntimeError("Nessuna nota rilevata.")
        for n in note:
            n.raw_onset = n.onset

        # 3 ─ verifica sull'audio originale
        rapporto: Dict = {}
        persi: List[float] = []
        if audio_path:
            try:
                y = load_audio(audio_path, None)
                rapporto = verifica_audio(y, note, cb)
                persi = rapporto.pop("_persi", [])
            except Exception as e:
                log(f"Verifica sull'audio saltata ({type(e).__name__}: {str(e)[:90]})", cb)
        grezze = [Nota(n.onset, n.offset, n.pitch, n.vel, raw_onset=n.onset) for n in note]

        # 4 ─ notazione
        ris = note_in_partitura(note, pedali, title, composer, quant, bpm, meter, opz, cb)
        sc, met = ris["score"], ris["metrica"]

        # 5 ─ file
        base = os.path.join(out_dir, _nome_file(title))
        xml_path = base + ".musicxml"
        sc.write("musicxml", fp=xml_path, makeNotation=False)
        try:
            scrivi_midi(grezze, pedali, base + ".mid")
        except Exception as e:
            log(f"MIDI non scritto ({e})", cb)

        sotto = f"{'semiminima puntata' if met.beat_ql != 1 else 'semiminima'} = {met.bpm:.0f}"
        out_pdf = base + ".pdf"
        fatto = False
        if opz.get("musescore"):
            fatto = pdf_musescore(xml_path, out_pdf, cb)
            if not fatto:
                log("MuseScore non disponibile: uso l'impaginazione integrata.", cb)
        if not fatto:
            try:
                fatto = pdf_verovio(xml_path, out_pdf, title, composer, sotto, cb)
            except Exception as e:
                import traceback
                log(f"Impaginazione integrata fallita ({type(e).__name__}: {str(e)[:200]})", cb)
                for r in traceback.format_exc().splitlines():
                    righe.append("    " + r)
                fatto = pdf_in_altro_processo(xml_path, out_pdf, title, composer, sotto, cb)
                if not fatto:
                    fatto = pdf_musescore(xml_path, out_pdf, cb)
        if not fatto:
            log("PDF non generato: il motivo è in partitura_ultimo.log, nella cartella di uscita.", cb)
        finale = out_pdf if fatto else xml_path

        # 6 ─ rapporto
        def battuta(t: float) -> str:
            q = float(met.to_pos(t)) * float(met.beat_ql)
            b = int(q // float(met.bar_ql))
            mov = int((q - b * float(met.bar_ql)) / float(met.beat_ql)) + 1
            return f"batt. {b + 1}, mov. {mov}"

        rapporto.update({
            "titolo": title, "autore": composer, "motore": usato,
            "tempo": met.sig, "bpm": round(met.bpm, 1), "tonalita": ris["tonalita"][2],
            "quantizzazione": quant, "battute": int(len(sc.parts[0].getElementsByClass("Measure"))),
            "note_in_partitura": len(ris["note"]),
            "abbellimenti": ris["abbellimenti"],
            "battute_con_durata_errata": len(ris["errori_misure"]),
            "attacchi_audio_non_trascritti": [battuta(t) for t in persi[:80]],
        })
        rep_path = base + "_verifica.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rapporto, f, indent=2, ensure_ascii=False)

        log("", cb)
        log(f"FATTO — {finale}", cb)
        log(f"MusicXML (per correggere a mano): {xml_path}", cb)
        log(f"MIDI (per riascoltare): {base}.mid", cb)
        log(f"Rapporto di verifica: {rep_path}", cb)
        if rapporto.get("note_senza_riscontro"):
            log(f"In ROSSO le {rapporto['note_senza_riscontro']} note senza riscontro "
                f"nell'audio: controllale per prime.", cb)
        return {"pdf": finale, "xml": xml_path, "report": rapporto, "report_path": rep_path}
    except Exception:
        import traceback
        for r in traceback.format_exc().splitlines():
            righe.append("    " + r)
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            import platform
            with open(os.path.join(out_dir, "partitura_ultimo.log"), "w", encoding="utf-8") as f:
                f.write(f"PARTITURA {VERSIONE} | Python {sys.version.split()[0]} | "
                        f"{platform.system()} {platform.release()}\n")
                for mod in ("verovio", "svglib", "reportlab", "music21", "librosa", "torch", "numpy"):
                    try:
                        m_ = __import__(mod)
                        f.write(f"{mod} {getattr(m_, '__version__', getattr(m_, 'Version', '?'))}\n")
                    except Exception as e:
                        f.write(f"{mod} NON IMPORTABILE: {type(e).__name__}: {e}\n")
                f.write("\n".join(righe) + "\n")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACCIA GRAFICA
# ─────────────────────────────────────────────────────────────────────────────

# Avanzamento: ogni messaggio della pipeline segna una tappa (percentuale).
TAPPE = [("Scarico l'audio", 2), ("Carico l'audio", 4), ("Trascrizione", 6),
         ("MIDI:", 50), ("Note rilevate", 58), ("Verifica sull'audio", 60),
         ("Mani", 68), ("Abbellimenti", 72), ("Tempo:", 78), ("Quantizzazione", 83),
         ("Tonalità", 85), ("Partitura:", 88), ("Impagino", 90), ("FATTO", 100)]
NOME_TAPPA = {"Scarico l'audio": "Scarico da YouTube", "Carico l'audio": "Leggo l'audio",
              "Trascrizione": "Trascrivo le note (la fase più lunga)",
              "MIDI:": "Leggo il MIDI", "Note rilevate": "Note trascritte",
              "Verifica sull'audio": "Confronto con l'audio originale",
              "Mani": "Separo le mani", "Abbellimenti": "Cerco gli abbellimenti",
              "Tempo:": "Riconosco tempo e battute", "Quantizzazione": "Allineo le durate",
              "Tonalità": "Riconosco la tonalità", "Partitura:": "Scrivo la partitura",
              "Impagino": "Impagino il PDF", "FATTO": "Fatto"}


class Lettore:
    """Riproduce MIDI e audio con il lettore integrato di Windows (MCI), senza
    pacchetti aggiuntivi. Altrove non fa nulla."""

    def __init__(self):
        self.ok = sys.platform == "win32"
        self.aperto = False
        if self.ok:
            import ctypes
            self._mci = ctypes.windll.winmm.mciSendStringW
            self._buf = ctypes.create_unicode_buffer(256)

    def _cmd(self, s: str) -> str:
        err = self._mci(s, self._buf, 255, 0)
        if err:
            raise RuntimeError(f"MCI {err}: {s}")
        return self._buf.value

    def suona(self, path: str):
        if not self.ok:
            raise RuntimeError("La riproduzione è disponibile solo su Windows.")
        self.ferma()
        tipo = "sequencer" if is_midi(path) else "mpegvideo"
        self._cmd(f'open "{path}" type {tipo} alias brano')
        self.aperto = True
        self._cmd("set brano time format milliseconds")
        self._cmd("play brano")

    def stato(self) -> Tuple[str, int, int]:
        if not (self.ok and self.aperto):
            return "fermo", 0, 0
        try:
            return (self._cmd("status brano mode"), int(self._cmd("status brano position") or 0),
                    int(self._cmd("status brano length") or 0))
        except Exception:
            return "fermo", 0, 0

    def ferma(self):
        if self.ok and self.aperto:
            try:
                self._mci("close brano", None, 0, 0)
            except Exception:
                pass
        self.aperto = False


def _impostazioni_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "Partitura", "impostazioni.json")


def _leggi_impostazioni() -> Dict:
    try:
        with open(_impostazioni_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _salva_impostazioni(d: Dict):
    try:
        os.makedirs(os.path.dirname(_impostazioni_path()), exist_ok=True)
        with open(_impostazioni_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
    except Exception:
        pass


def _pagine_pdf(path: str, larghezza: int) -> List:
    """Pagine del PDF come immagini Tk, larghe «larghezza» pixel."""
    import pypdfium2 as pdfium
    from PIL import ImageTk
    doc = pdfium.PdfDocument(path)
    out = []
    try:
        for i in range(len(doc)):
            pg = doc[i]
            scala = larghezza / pg.get_width()
            out.append(ImageTk.PhotoImage(pg.render(scale=scala).to_pil()))
    finally:
        doc.close()
    return out


def launch_gui(prova: Optional[str] = None):
    """Finestra principale. Con «prova» = file MIDI o audio, l'app si collauda
    da sola: genera, mostra l'anteprima, apre il lettore e si chiude
    (codice d'uscita 0 se tutto è andato)."""
    import queue
    import time
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
        dnd = True
    except Exception:
        root = tk.Tk()
        dnd = False
    root.title(f"Partitura {VERSIONE}")
    root.geometry("1180x780")
    root.minsize(940, 620)
    try:
        ico = os.path.join(os.path.dirname(sys.executable if CONGELATO else __file__), "partitura.ico")
        if os.path.exists(ico):
            root.iconbitmap(ico)
    except Exception:
        pass

    st = ttk.Style()
    try:
        st.theme_use("vista" if sys.platform == "win32" else "clam")
    except Exception:
        pass
    BLU, GRIGIO, CHIARO = "#2457a6", "#6b6b6b", "#f4f6fa"
    st.configure("Titolo.TLabel", font=("Segoe UI", 15, "bold"))
    st.configure("Nota.TLabel", foreground=GRIGIO, font=("Segoe UI", 8))
    st.configure("Tappa.TLabel", font=("Segoe UI", 9, "bold"))
    st.configure("Grande.TButton", font=("Segoe UI", 11, "bold"), padding=8)

    imp = _leggi_impostazioni()
    v = dict(
        src=tk.StringVar(), tit=tk.StringVar(), aut=tk.StringVar(),
        quant=tk.StringVar(value=imp.get("quant", "media")), bpm=tk.StringVar(),
        tempo=tk.StringVar(value="auto"), motore=tk.StringVar(value=imp.get("motore", "auto")),
        out=tk.StringVar(value=imp.get("out") or os.path.join(os.path.expanduser("~"), "Partiture")),
        tappa=tk.StringVar(value="Pronto"), pct=tk.StringVar(value=""),
        pagina=tk.StringVar(value=""), tempo_ascolto=tk.StringVar(value=""),
    )
    b = {k: tk.BooleanVar(value=imp.get(k, d)) for k, d in
         [("rosso", True), ("dinamiche", True), ("pedale", True),
          ("staccati", True), ("musescore", False)]}
    stato = {"ris": None, "pagine": [], "pag": 0, "lavoro": False,
             "t0": 0.0, "pct": 0.0, "meta": 0.0, "esito": None, "audio": None}
    coda: "queue.Queue" = queue.Queue()
    lettore = Lettore()

    # ── struttura: colonna sinistra (comandi) | destra (anteprima e registro)
    pan = ttk.PanedWindow(root, orient="horizontal")
    pan.pack(fill="both", expand=True)
    sx = ttk.Frame(pan, padding=(14, 12))
    dx = ttk.Frame(pan, padding=(6, 12, 12, 6))
    pan.add(sx, weight=0)
    pan.add(dx, weight=1)

    ttk.Label(sx, text="Partitura", style="Titolo.TLabel").pack(anchor="w")
    ttk.Label(sx, text="audio di pianoforte → spartito PDF", style="Nota.TLabel").pack(anchor="w", pady=(0, 10))

    # zona di rilascio
    zona = tk.Label(sx, text="Trascina qui un file audio o MIDI\noppure clicca per sceglierlo",
                    bg=CHIARO, fg=BLU, font=("Segoe UI", 10), relief="groove", bd=2,
                    height=5, width=38, cursor="hand2", justify="center")
    zona.pack(fill="x", pady=(0, 6))
    ttk.Label(sx, text="…o incolla un link YouTube:", style="Nota.TLabel").pack(anchor="w")
    e_src = ttk.Entry(sx, textvariable=v["src"])
    e_src.pack(fill="x", pady=(0, 8))

    def imposta_sorgente(p: str):
        p = p.strip().strip('"').strip("{}")
        if not p:
            return
        v["src"].set(p)
        nome = p if is_youtube(p) else os.path.basename(p)
        zona.config(text=f"✓ {nome}\n(clicca o trascina per cambiarlo)", fg="#1d7a3a")
        if not is_youtube(p) and not v["tit"].get():
            v["tit"].set(os.path.splitext(os.path.basename(p))[0].replace("_", " "))

    def scegli(_=None):
        p = filedialog.askopenfilename(filetypes=[
            ("Audio e MIDI", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.mid *.midi"), ("Tutti", "*.*")])
        if p:
            imposta_sorgente(p)

    zona.bind("<Button-1>", scegli)
    if dnd:
        def rilascio(ev):
            files = root.tk.splitlist(ev.data)
            if files:
                imposta_sorgente(files[0])
            zona.config(bg=CHIARO)
            return ev.action
        for w in (zona, e_src):
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", rilascio)
        zona.dnd_bind("<<DropEnter>>", lambda e: (zona.config(bg="#dde7f7"), e.action)[1])
        zona.dnd_bind("<<DropLeave>>", lambda e: (zona.config(bg=CHIARO), e.action)[1])
    else:
        zona.config(text="Clicca per scegliere un file audio o MIDI")

    griglia = ttk.Frame(sx)
    griglia.pack(fill="x")
    griglia.columnconfigure(1, weight=1)

    def campo(r, testo, var, largo=True):
        ttk.Label(griglia, text=testo).grid(row=r, column=0, sticky="w", pady=3)
        e = ttk.Entry(griglia, textvariable=var, width=30 if largo else 8)
        e.grid(row=r, column=1, sticky="we" if largo else "w", pady=3, padx=(8, 0))
        return e

    def tendina(r, testo, var, valori):
        ttk.Label(griglia, text=testo).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Combobox(griglia, textvariable=var, values=valori, state="readonly",
                     width=10).grid(row=r, column=1, sticky="w", pady=3, padx=(8, 0))

    campo(0, "Titolo", v["tit"])
    campo(1, "Autore", v["aut"])
    tendina(2, "Quantizzazione", v["quant"], ["lieve", "media", "forte"])
    ttk.Label(griglia, text="lieve = fedele all'esecuzione · forte = più semplice da leggere",
              style="Nota.TLabel").grid(row=3, column=0, columnspan=2, sticky="w")
    campo(4, "BPM", v["bpm"], largo=False)
    ttk.Label(griglia, text="vuoto = automatico. Se il brano esce a velocità doppia,\n"
              "indica il BPM che senti battendo il piede.",
              style="Nota.TLabel").grid(row=5, column=0, columnspan=2, sticky="w")
    tendina(6, "Tempo", v["tempo"], ["auto"] + TEMPI)
    tendina(7, "Motore", v["motore"], ["auto", "transkun", "crnn"])

    opz = ttk.LabelFrame(sx, text="Nello spartito", padding=6)
    opz.pack(fill="x", pady=8)
    for i, (k, t) in enumerate([("rosso", "Note dubbie in rosso"), ("dinamiche", "Dinamiche"),
                                ("pedale", "Pedale"), ("staccati", "Staccati"),
                                ("musescore", "Impagina con MuseScore")]):
        ttk.Checkbutton(opz, text=t, variable=b[k]).grid(row=i // 2, column=i % 2, sticky="w", padx=4)

    cart = ttk.Frame(sx)
    cart.pack(fill="x")
    ttk.Label(cart, text="Salva in").pack(side="left")
    ttk.Entry(cart, textvariable=v["out"]).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(cart, text="…", width=3,
               command=lambda: v["out"].set(filedialog.askdirectory() or v["out"].get())).pack(side="left")

    btn = ttk.Button(sx, text="GENERA PARTITURA", style="Grande.TButton")
    btn.pack(fill="x", pady=(12, 6))
    barra = ttk.Progressbar(sx, maximum=100, mode="determinate")
    barra.pack(fill="x")
    rg = ttk.Frame(sx)
    rg.pack(fill="x")
    ttk.Label(rg, textvariable=v["tappa"], style="Tappa.TLabel").pack(side="left")
    ttk.Label(rg, textvariable=v["pct"], style="Nota.TLabel").pack(side="right")

    # ── destra: anteprima e registro
    nb = ttk.Notebook(dx)
    nb.pack(fill="both", expand=True)
    tab_ant = ttk.Frame(nb)
    tab_log = ttk.Frame(nb)
    nb.add(tab_ant, text="  Anteprima  ")
    nb.add(tab_log, text="  Registro  ")

    tela = tk.Canvas(tab_ant, bg="#8a8f98", highlightthickness=0)
    sb = ttk.Scrollbar(tab_ant, orient="vertical", command=tela.yview)
    tela.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    tela.pack(side="left", fill="both", expand=True)
    tela.create_text(20, 20, anchor="nw", fill="white", font=("Segoe UI", 12), tags="vuoto",
                     text="Qui comparirà lo spartito.\n\nScegli un brano a sinistra e premi "
                          "GENERA PARTITURA.")

    txt = tk.Text(tab_log, wrap="word", bg="#1e1e1e", fg="#d4d4d4", relief="flat",
                  insertbackground="#d4d4d4", font=("Consolas", 9))
    sbl = ttk.Scrollbar(tab_log, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sbl.set)
    sbl.pack(side="right", fill="y")
    txt.pack(fill="both", expand=True)

    # barra dei comandi sotto l'anteprima
    cmd = ttk.Frame(dx, padding=(0, 6, 0, 0))
    cmd.pack(fill="x")
    b_prec = ttk.Button(cmd, text="◀", width=3)
    b_succ = ttk.Button(cmd, text="▶", width=3)
    b_prec.pack(side="left")
    ttk.Label(cmd, textvariable=v["pagina"], width=12, anchor="center").pack(side="left")
    b_succ.pack(side="left")
    ttk.Separator(cmd, orient="vertical").pack(side="left", fill="y", padx=10)
    b_midi = ttk.Button(cmd, text="▶ Ascolta trascrizione")
    b_orig = ttk.Button(cmd, text="▶ Ascolta originale")
    b_stop = ttk.Button(cmd, text="■", width=3)
    for w in (b_midi, b_orig, b_stop):
        w.pack(side="left", padx=2)
    ttk.Label(cmd, textvariable=v["tempo_ascolto"], style="Nota.TLabel").pack(side="left", padx=6)
    b_cart = ttk.Button(cmd, text="Apri cartella")
    b_pdf = ttk.Button(cmd, text="Apri PDF")
    b_xml = ttk.Button(cmd, text="Apri MusicXML")
    for w in (b_cart, b_xml, b_pdf):
        w.pack(side="right", padx=2)
    risultato_btn = [b_prec, b_succ, b_midi, b_orig, b_stop, b_cart, b_pdf, b_xml]

    def abilita_risultati():
        r = stato["ris"]
        for w in risultato_btn:
            w.state(["disabled"])
        if not r:
            return
        b_cart.state(["!disabled"])
        if r.get("xml"):
            b_xml.state(["!disabled"])
        if r["pdf"].lower().endswith(".pdf"):
            b_pdf.state(["!disabled"])
        if lettore.ok:
            if os.path.exists(os.path.splitext(r["xml"])[0] + ".mid"):
                b_midi.state(["!disabled"])
            if stato["audio"]:
                b_orig.state(["!disabled"])
            b_stop.state(["!disabled"])
        if len(stato["pagine"]) > 1:
            b_prec.state(["!disabled"])
            b_succ.state(["!disabled"])

    abilita_risultati()

    # ── anteprima
    def mostra_pagina():
        tela.delete("all")
        pg = stato["pagine"]
        if not pg:
            return
        i = stato["pag"] = max(0, min(stato["pag"], len(pg) - 1))
        x = max(10, (tela.winfo_width() - pg[i].width()) // 2)
        tela.create_image(x, 10, anchor="nw", image=pg[i])
        tela.configure(scrollregion=(0, 0, tela.winfo_width(), pg[i].height() + 20))
        tela.yview_moveto(0)
        v["pagina"].set(f"pagina {i + 1} di {len(pg)}")

    def carica_anteprima():
        r = stato["ris"]
        if not r or not r["pdf"].lower().endswith(".pdf"):
            return
        try:
            larg = max(400, min(tela.winfo_width() - 30, 1000))
            stato["pagine"] = _pagine_pdf(r["pdf"], larg)
            stato["larg"] = larg
            mostra_pagina()
        except Exception as e:
            tela.delete("all")
            tela.create_text(20, 20, anchor="nw", fill="white", font=("Segoe UI", 11),
                             text=f"Anteprima non disponibile ({type(e).__name__}).\n"
                                  "Il PDF è comunque pronto: «Apri PDF».")

    def sposta(d):
        stato["pag"] += d
        mostra_pagina()

    b_prec.config(command=lambda: sposta(-1))
    b_succ.config(command=lambda: sposta(+1))
    tela.bind("<MouseWheel>", lambda e: tela.yview_scroll(int(-e.delta / 120), "units"))
    root.bind("<Prior>", lambda e: sposta(-1))
    root.bind("<Next>", lambda e: sposta(+1))

    ridim = {"id": None}

    def al_ridimensionamento(_):
        if not stato["pagine"]:
            return
        if ridim["id"]:
            root.after_cancel(ridim["id"])
        def rifai():
            if abs(min(tela.winfo_width() - 30, 1000) - stato.get("larg", 0)) > 40:
                carica_anteprima()
            else:
                mostra_pagina()
        ridim["id"] = root.after(250, rifai)

    tela.bind("<Configure>", al_ridimensionamento)

    # ── apertura file e ascolto
    def apri(path):
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Apertura", f"Non riesco ad aprire:\n{path}\n\n{e}")

    b_pdf.config(command=lambda: apri(stato["ris"]["pdf"]))
    b_xml.config(command=lambda: apri(stato["ris"]["xml"]))
    b_cart.config(command=lambda: apri(os.path.dirname(stato["ris"]["xml"])))

    def ascolta(path):
        try:
            lettore.suona(path)
        except Exception as e:
            messagebox.showerror("Ascolto", f"Non riesco a riprodurre il file.\n\n{e}")

    b_midi.config(command=lambda: ascolta(os.path.splitext(stato["ris"]["xml"])[0] + ".mid"))
    b_orig.config(command=lambda: ascolta(stato["audio"]))
    b_stop.config(command=lambda: (lettore.ferma(), v["tempo_ascolto"].set("")))

    def orologio():
        modo, pos, lung = lettore.stato()
        if modo == "playing" and lung:
            v["tempo_ascolto"].set(f"{pos // 60000}:{pos // 1000 % 60:02d} / "
                                   f"{lung // 60000}:{lung // 1000 % 60:02d}")
        elif v["tempo_ascolto"].get():
            v["tempo_ascolto"].set("")
        root.after(500, orologio)

    orologio()

    # ── lavoro in sottofondo: i messaggi passano da una coda
    def worker(par):
        try:
            res = run_pipeline(cb=lambda line: coda.put(("riga", line)), **par)
            coda.put(("fatto", res))
        except Exception as e:
            import traceback
            coda.put(("riga", "\nERRORE: " + str(e)))
            coda.put(("riga", traceback.format_exc()))
            coda.put(("errore", str(e)))

    def avanza(line: str):
        for chiave, pct in TAPPE:
            if chiave in line and pct >= stato["pct"]:
                stato["pct"] = pct
                v["tappa"].set(NOME_TAPPA[chiave])
                succ = [p for _, p in TAPPE if p > pct]
                stato["meta"] = succ[0] - 1 if succ else 100
                break

    def ciclo():
        try:
            while True:
                tipo, x = coda.get_nowait()
                if tipo == "riga":
                    txt.insert("end", x + "\n")
                    txt.see("end")
                    avanza(x)
                elif tipo == "fatto":
                    fine(x, None)
                elif tipo == "errore":
                    fine(None, x)
        except queue.Empty:
            pass
        if stato["lavoro"]:
            # dentro una tappa la barra avanza piano verso la successiva
            stato["pct"] += (stato["meta"] - stato["pct"]) * 0.004
            barra["value"] = stato["pct"]
            s = int(time.time() - stato["t0"])
            v["pct"].set(f"{int(stato['pct'])}% · {s // 60}:{s % 60:02d}")
        root.after(100, ciclo)

    def fine(res, errore):
        stato["lavoro"] = False
        btn.state(["!disabled"])
        btn.config(text="GENERA PARTITURA")
        if errore:
            stato["esito"] = False
            barra["value"] = 0
            v["tappa"].set("Errore — dettagli nel Registro")
            nb.select(tab_log)
            if not prova:
                messagebox.showerror("Errore", errore)
            return
        stato["esito"] = True
        stato["ris"] = res
        barra["value"] = 100
        rep = res.get("report", {})
        v["tappa"].set(f"Fatto · {rep.get('battute', '?')} battute · {rep.get('tempo', '')} · "
                       f"{rep.get('tonalita', '')}")
        v["pct"].set("")
        nb.select(tab_ant)
        carica_anteprima()
        abilita_risultati()
        if not res["pdf"].lower().endswith(".pdf"):
            nb.select(tab_log)
            if not prova:
                messagebox.showinfo("Partitura pronta (MusicXML)",
                                    "Il PDF non è stato generato: il motivo è nel Registro.\n"
                                    "La partitura è nel file MusicXML (si apre con MuseScore).")

    def start():
        if stato["lavoro"]:
            return
        src = v["src"].get().strip().strip('"')
        if not src:
            messagebox.showwarning("Brano", "Trascina un file audio o MIDI, oppure incolla un link YouTube.")
            return
        try:
            bpm = float(v["bpm"].get().replace(",", ".")) if v["bpm"].get().strip() else None
        except ValueError:
            messagebox.showerror("BPM", "Il BPM deve essere un numero (es. 72).")
            return
        par = dict(source=src, out_dir=v["out"].get().strip(),
                   title=v["tit"].get().strip(), composer=v["aut"].get().strip(),
                   quant=v["quant"].get(), bpm=bpm, meter=v["tempo"].get(),
                   motore=v["motore"].get(), opz={k: x.get() for k, x in b.items()})
        _salva_impostazioni({"out": par["out_dir"], "quant": par["quant"], "motore": par["motore"],
                             **par["opz"]})
        lettore.ferma()
        stato.update(ris=None, pagine=[], pag=0, lavoro=True, t0=time.time(), pct=0.0, meta=5.0,
                     audio=None if (is_youtube(src) or is_midi(src)) else src)
        abilita_risultati()
        txt.delete("1.0", "end")
        tela.delete("all")
        v["pagina"].set("")
        v["tappa"].set("Avvio…")
        btn.state(["disabled"])
        btn.config(text="elaborazione in corso…")
        threading.Thread(target=worker, args=(par,), daemon=True).start()

    btn.config(command=start)
    root.bind("<Return>", lambda e: start())

    def chiudi():
        if stato["lavoro"] and not prova and not messagebox.askyesno(
                "Uscire?", "La partitura è ancora in elaborazione. Uscire comunque?"):
            return
        lettore.ferma()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", chiudi)
    ciclo()

    if prova:
        # collaudo automatico dell'interfaccia
        esito = {"codice": 1}
        imposta_sorgente(prova)
        v["tit"].set("Collaudo interfaccia")
        v["bpm"].set("138")
        root.after(500, start)

        def controlla():
            if stato["esito"] is None and time.time() - stato["t0"] < 900:
                root.after(500, controlla)
                return
            ok = bool(stato["esito"]) and len(stato["pagine"]) > 0
            print(f"COLLAUDO GUI: esito={stato['esito']} pagine={len(stato['pagine'])} "
                  f"dnd={dnd} lettore={lettore.ok} barra={barra['value']}")
            if ok and lettore.ok:
                try:
                    lettore.suona(os.path.splitext(stato["ris"]["xml"])[0] + ".mid")
                    print("COLLAUDO GUI: lettore MIDI", lettore.stato())
                except Exception as e:
                    print(f"COLLAUDO GUI: lettore MIDI non disponibile qui ({e})")
                lettore.ferma()
            esito["codice"] = 0 if ok else 1
            root.after(300, root.destroy)

        root.after(1000, controlla)
        root.mainloop()
        return esito["codice"]
    root.mainloop()
    return 0


# ─────────────────────────────────────────────────────────────────────────────

NECESSARI = [("numpy", "numpy"), ("scipy", "scipy"), ("librosa", "librosa"),
             ("music21", "music21"), ("mido", "mido"), ("verovio", "verovio"),
             ("svglib", "svglib"), ("reportlab", "reportlab")]


def controlla_dipendenze() -> List[str]:
    import importlib
    manca = []
    for mod, pkg in NECESSARI:
        try:
            importlib.import_module(mod)
        except Exception:
            manca.append(pkg)
    return manca


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--impagina":
        # uso interno: MusicXML → PDF in un processo a parte
        a = sys.argv[2:] + ["", "", ""]
        try:
            ok = pdf_verovio(a[0], a[1], a[2], a[3], a[4])
        except Exception as e:
            print(f"ERRORE {type(e).__name__}: {e}")
            ok = False
        sys.exit(0 if ok else 3)
    if len(sys.argv) >= 3 and sys.argv[1] == "--prova-gui":
        sys.exit(launch_gui(prova=sys.argv[2]))
    ap = argparse.ArgumentParser(description="Audio di pianoforte → partitura PDF.")
    ap.add_argument("source", nargs="?", help="file audio, file MIDI o link YouTube")
    ap.add_argument("-o", "--out", default="./partiture", help="cartella di uscita")
    ap.add_argument("-t", "--titolo", default="")
    ap.add_argument("-a", "--autore", default="")
    ap.add_argument("-q", "--quant", choices=["lieve", "media", "forte"], default="media")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--tempo", default="auto", help="auto, " + ", ".join(TEMPI))
    ap.add_argument("--motore", choices=["auto", "transkun", "crnn"], default="auto")
    ap.add_argument("--no-rosso", action="store_true")
    ap.add_argument("--no-dinamiche", action="store_true")
    ap.add_argument("--no-pedale", action="store_true")
    ap.add_argument("--no-staccati", action="store_true")
    ap.add_argument("--musescore", action="store_true", help="impagina con MuseScore")
    args = ap.parse_args()

    manca = controlla_dipendenze()
    if manca:
        print("\n  Mancano dei pacchetti: " + ", ".join(manca))
        print(f"  {sys.executable} -m pip install {' '.join(manca)}")
        print("  (oppure rilancia INSTALLA.bat)\n")
        sys.exit(1)

    if not args.source:
        launch_gui()
        return
    run_pipeline(args.source, args.out, args.titolo, args.autore, args.quant,
                 args.bpm, args.tempo, args.motore,
                 dict(rosso=not args.no_rosso, dinamiche=not args.no_dinamiche,
                      pedale=not args.no_pedale, staccati=not args.no_staccati,
                      musescore=args.musescore))


def _registra_crash():
    """Nell'app compilata non c'è console: un errore fatale va scritto su file."""
    import traceback
    testo = traceback.format_exc()
    for cartella in (os.environ.get("LOCALAPPDATA", ""), os.path.expanduser("~"), os.getcwd()):
        try:
            d = os.path.join(cartella, "Partitura") if cartella else "Partitura"
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "errore.log"), "w", encoding="utf-8") as f:
                f.write(f"PARTITURA {VERSIONE}\n{' '.join(sys.argv)}\n\n{testo}")
            break
        except Exception:
            continue
    try:
        print(testo, file=sys.stderr)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        _registra_crash()
        sys.exit(1)
