"""Crea tests/prova.wav e tests/prova.mid: un breve brano di pianoforte sintetico."""
import os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import banco_prova as B, partitura as P
P.STAMPA = False
qui = os.path.dirname(os.path.abspath(__file__))
nome, tempo, bpm, ton, modo, batt, kw, dai = [b for b in B.BRANI if b[0] == "Valzer"][0]
ev, bq, bar = B.componi(11, tempo, ton, modo, 8, **kw)
note, _ = B.esegui(ev, bq, bpm, 11, pedale=False, bar=bar)
SR = 22050
y = np.zeros(int((max(n.offset for n in note) + 1.5) * SR))
for n in note:
    f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
    t = np.arange(int((n.offset - n.onset + 0.2) * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h ** 1.3 * np.exp(-t * (1.5 + 0.6 * h)) for h in range(1, 7))
    x *= np.minimum(t / 0.004, 1) * (n.vel / 127) ** 1.5
    i = int(n.onset * SR)
    y[i:i + len(x)] += x[:len(y) - i]
sf.write(os.path.join(qui, "prova.wav"), y / np.abs(y).max() * 0.8, SR)
P.scrivi_midi([P.Nota(n.onset, n.offset, n.pitch, n.vel, raw_onset=n.onset) for n in note], [],
              os.path.join(qui, "prova.mid"))
print("ok", len(note), "note")
