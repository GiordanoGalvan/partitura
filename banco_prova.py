#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BANCO DI PROVA — misura quanto la notazione prodotta coincide con quella vera.

Si scrivono partiture di cui si conosce ogni nota, le si «esegue» come farebbe
un pianista (rubato, attacchi imprecisi di qualche millisecondo, accordi non
perfettamente insieme, abbellimenti suonati per esteso) e si chiede a
partitura.py di ricostruirle. Poi si confronta nota per nota.

Non collauda il riconoscimento delle note dall'audio (quello è il modello
neurale): collauda tutto ciò che viene dopo, cioè la scrittura.

    python banco_prova.py            tutti i brani
    python banco_prova.py --pdf      salva anche i PDF in ./prova_uscita
"""

import os
import sys
import random
import argparse
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import List, Optional, Dict, Tuple

import numpy as np

# Su Windows la console e i tubi fra processi usano cp1252: senza questo, un
# carattere come «─» o «→» fa fallire la stampa.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import partitura as P
P.STAMPA = False


@dataclass
class GT:
    q: F
    dur: F
    pitches: List[int]
    hand: int
    orn: Optional[str] = None            # trill | mordent | inv_mordent | turn | arp
    graces: List[int] = field(default_factory=list)
    stacc: bool = False


SCALE = {"magg": [0, 2, 4, 5, 7, 9, 11], "min": [0, 2, 3, 5, 7, 8, 10]}


def grado(tonica: int, modo: str, g: int) -> int:
    sc = SCALE[modo]
    return tonica + 12 * (g // 7) + sc[g % 7]


def componi(seme: int, tempo: str, tonica: int, modo: str, battute: int,
            p_sedicesimi=0.15, p_terzine=0.0, p_abbell=0.0, p_arp=0.0) -> Tuple[List[GT], F, F]:
    """Scrive un brano a due mani. Ritorna (eventi, durata del movimento, durata della battuta)."""
    rnd = random.Random(seme)
    num, den = (int(x) for x in tempo.split("/"))
    composto = den == 8
    bq = F(3, 2) if composto else F(1)
    per_bar = num // 3 if composto else num
    bar = bq * per_bar
    ev: List[GT] = []

    if composto:
        celle = [([F(3, 2)], 2), ([F(1), F(1, 2)], 4), ([F(1, 2)] * 3, 4),
                 ([F(1, 2), F(1)], 1), ([F(1, 4)] * 6, 1 if p_sedicesimi > 0.1 else 0)]
    else:
        celle = [([F(1)], 4), ([F(1, 2)] * 2, 5), ([F(3, 4), F(1, 4)], 2),
                 ([F(1, 2), F(1, 4), F(1, 4)], 2), ([F(1, 4), F(1, 4), F(1, 2)], 2),
                 ([F(1, 4)] * 4, int(20 * p_sedicesimi)),
                 ([F(1, 3)] * 3, int(20 * p_terzine))]
    celle = [(c, w) for c, w in celle if w > 0]

    g = 7 + rnd.randint(0, 4)                       # grado della melodia (ottava centrale)
    for b in range(battute):
        accordo = rnd.choice([0, 3, 4, 0, 5]) if b % 4 else 0
        # ── mano destra
        k = 0
        while k < per_bar:
            t = b * bar + k * bq
            if (not composto and k + 2 <= per_bar and k % 2 == 0 and rnd.random() < 0.18):
                durate, passo = [F(2)], 2
            else:
                durate, passo = rnd.choices([c for c, _ in celle], [w for _, w in celle])[0], 1
            if b == battute - 1 and k == 0:
                durate, passo = [bar], per_bar
            for d in durate:
                g += rnd.choice([-2, -1, -1, 1, 1, 2, 0])
                g = max(7, min(g, 17))
                p = grado(tonica + 60, modo, g - 7)
                e = GT(t, d, [p], 0)
                r = rnd.random()
                if d >= 1 and r < p_abbell:
                    e.orn = rnd.choice(["trill", "mordent", "inv_mordent", "turn"])
                elif d >= F(1, 2) and r < p_abbell * 2.2:
                    e.graces = [grado(tonica + 60, modo, g - 7 + rnd.choice([-1, 1]))]
                    # la doppia acciaccatura vuole spazio: solo dopo una nota di almeno una croma
                    if rnd.random() < 0.25 and ev and ev[-1].hand == 0 and ev[-1].dur >= F(1, 2):
                        e.graces = [grado(tonica + 60, modo, g - 9), grado(tonica + 60, modo, g - 8)]
                elif d <= F(1, 2) and d >= F(1, 2) and rnd.random() < 0.10:
                    e.stacc = True
                ev.append(e)
                t += d
            k += passo
        # ── mano sinistra
        base = grado(tonica + 36, modo, accordo)
        terza = grado(tonica + 36, modo, accordo + 2)
        quinta = grado(tonica + 36, modo, accordo + 4)
        if b == battute - 1:
            ev.append(GT(b * bar, bar, [base, quinta, base + 12], 1,
                         orn="arp" if p_arp > 0 else None))
            continue
        if rnd.random() < p_arp:
            ev.append(GT(b * bar, bar, [base, quinta, terza + 12, base + 24], 1, orn="arp"))
            continue
        if composto:
            for k in range(per_bar):
                for j, p in enumerate([base, terza, quinta]):
                    ev.append(GT(b * bar + k * bq + j * F(1, 2), F(1, 2), [p], 1))
        elif per_bar == 3:
            ev.append(GT(b * bar, F(1), [base if base <= 45 else base - 12], 1))
            # accordi nell'ottava sotto il Do centrale: le mani non si incrociano
            acc_ = sorted([48 + terza % 12, 48 + quinta % 12])
            ev.append(GT(b * bar + 1, F(1), acc_, 1))
            ev.append(GT(b * bar + 2, F(1), acc_, 1))
        else:
            for k in range(per_bar * 2):
                p = [base, quinta, terza, quinta][k % 4]
                ev.append(GT(b * bar + k * F(1, 2), F(1, 2), [p], 1))
    return ev, bq, bar


def componi_virtuoso(seme: int, tonica: int, modo: str, battute: int) -> Tuple[List[GT], F, F]:
    """
    Studio da concerto inventato per mettere alle corde il programma: salti di
    due ottave in semicrome, volate di biscrome, note ribattute, trilli lunghi,
    acciaccature, e una sinistra che salta fra basso e accordo.
    """
    rnd = random.Random(seme)
    bar = F(4)
    ev: List[GT] = []
    gr = lambda g, base=60: grado(tonica + base, modo, g)
    for b in range(battute):
        accordo = rnd.choice([0, 3, 4, 5]) if b % 4 else 0
        t0 = b * bar
        ultima = b == battute - 1
        # ── sinistra: basso profondo sul tempo, accordo due ottave sopra in levare
        basso = grado(tonica + 36, modo, accordo)
        basso = basso if basso <= 45 else basso - 12
        acc_ = sorted([48 + grado(tonica, modo, accordo + 2) % 12 + 7,
                       48 + grado(tonica, modo, accordo + 4) % 12 + 7,
                       48 + grado(tonica, modo, accordo) % 12 + 12])
        if ultima:
            ev.append(GT(t0, bar, [basso, basso + 12], 1))
        else:
            for k in range(4):
                ev.append(GT(t0 + k, F(1, 2), [basso, basso + 12] if k % 2 == 0 else [basso + 7], 1))
                ev.append(GT(t0 + k + F(1, 2), F(1, 2), acc_, 1))
        # ── destra: una figura per movimento
        if ultima:
            ev.append(GT(t0, bar, [gr(accordo + 7), gr(accordo + 9), gr(accordo + 11)], 0))
            continue
        k = 0
        while k < 4:
            t = t0 + k
            fig = rnd.choice(["salti", "salti", "volata", "ribattute", "trillo", "canto", "canto"])
            g = accordo + rnd.choice([7, 9, 11])
            if fig == "salti":                      # semicrome: giù–su di due ottave
                su = gr(g + 7)
                giu = gr(g - 5)
                for j in range(4):
                    ev.append(GT(t + F(j, 4), F(1, 4), [giu if j % 2 == 0 else su], 0))
            elif fig == "volata":                   # otto biscrome per grado
                verso = rnd.choice([1, -1])
                p0 = g if verso == 1 else g + 7
                for j in range(8):
                    ev.append(GT(t + F(j, 8), F(1, 8), [gr(p0 + verso * j)], 0))
            elif fig == "ribattute":                # la stessa nota quattro volte
                for j in range(4):
                    ev.append(GT(t + F(j, 4), F(1, 4), [gr(g + 4)], 0))
            elif fig == "trillo" and k <= 2:        # trillo lungo due movimenti
                ev.append(GT(t, F(2), [gr(g + 2)], 0, orn="trill"))
                k += 2
                continue
            else:                                   # due crome cantabili, la prima con acciaccatura
                ev.append(GT(t, F(1, 2), [gr(g + 3)], 0,
                             graces=[gr(g + 2)] if rnd.random() < 0.5 else []))
                ev.append(GT(t + F(1, 2), F(1, 2), [gr(g + 1)], 0))
            k += 1
    return ev, F(1), bar


def esegui(ev: List[GT], bq: F, bpm: float, seme: int, rubato=0.05, jitter=0.010,
           pedale=False, bar: F = F(4)) -> Tuple[List[P.Nota], list]:
    """Trasforma la partitura in un'esecuzione: tempi reali, imprecisi, con rubato."""
    rnd = random.Random(seme + 1000)
    fine = float(max(e.q + e.dur for e in ev))
    passi = np.arange(0, fine + 1.0, 1 / 48)
    lento = 1 + rubato * np.sin(2 * np.pi * passi / (float(bar) * 4)) \
        + 0.4 * rubato * np.sin(2 * np.pi * passi / (float(bar) * 1.3) + 1.0)
    sec_per_q = (60.0 / bpm) / float(bq) * lento
    tempo_di = np.concatenate([[0.0], np.cumsum(sec_per_q[:-1] / 48)])
    T = lambda q: float(np.interp(float(q), passi, tempo_di)) + 0.5

    note: List[P.Nota] = []
    for e in ev:
        t0 = T(e.q) + rnd.gauss(0, jitter)
        t1 = T(e.q + e.dur)
        tenuta = 0.30 if e.stacc else 0.92
        vel = int(max(30, min(120, (78 if e.hand == 0 else 62) + rnd.gauss(0, 6)
                              + (10 if (e.q % bar) == 0 else 0))))
        fine_n = t0 + (t1 - t0) * tenuta
        p0 = e.pitches[0]
        if e.orn == "arp":
            for i, p in enumerate(sorted(e.pitches)):
                note.append(P.Nota(t0 + i * 0.045, fine_n, p, vel))
            continue
        for i, gp in enumerate(e.graces):
            tg = t0 - 0.055 * (len(e.graces) - i)
            note.append(P.Nota(tg, tg + 0.05, gp, vel - 8))
        if e.orn == "trill":
            t, i = t0, 0
            while t < fine_n - 0.12:
                note.append(P.Nota(t, t + 0.08, p0 + (2 if i % 2 else 0), vel - 5))
                t += 0.085
                i += 1
            if i % 2 == 0:                              # chiude sulla nota reale
                note.append(P.Nota(t, fine_n, p0, vel - 5))
            else:
                note[-1].offset = fine_n
        elif e.orn in ("mordent", "inv_mordent"):
            aux = p0 + 2 if e.orn == "inv_mordent" else p0 - 1
            note.append(P.Nota(t0, t0 + 0.065, p0, vel))
            note.append(P.Nota(t0 + 0.07, t0 + 0.135, aux, vel - 6))
            note.append(P.Nota(t0 + 0.14, fine_n, p0, vel))
        elif e.orn == "turn":
            for i, p in enumerate([p0 + 2, p0, p0 - 1]):
                note.append(P.Nota(t0 + i * 0.075, t0 + i * 0.075 + 0.07, p, vel - 4))
            note.append(P.Nota(t0 + 0.225, fine_n, p0, vel))
        else:
            for p in e.pitches:
                note.append(P.Nota(t0 + rnd.uniform(-0.006, 0.006), fine_n, p, vel))
    pedali = []
    if pedale:
        q = F(0)
        while q < fine:
            pedali.append((T(q) + 0.05, T(q + bar) - 0.05))
            q += bar
    note.sort(key=lambda n: (n.onset, n.pitch))
    return note, pedali


def confronta(ev: List[GT], ris: Dict) -> Dict:
    met = ris["metrica"]
    vero = {(p, e.q, e.hand): e for e in ev for p in e.pitches}
    fuori = []
    for voci in (ris["eventi"], ris.get("eventi2") or {}):
        for mano, evs in voci.items():
            for x in evs:
                for n in x.note:
                    fuori.append((n.pitch, x.inizio, mano, x, n))

    # la battuta 1 del risultato può non coincidere con quella vera: si cerca lo scarto
    def punteggio(sc):
        chiavi = {(p, q - sc) for p, q, _, _, _ in fuori}
        return sum(1 for (p, q, _h) in vero if (p, q) in chiavi)

    scarti = [k * met.beat_ql for k in range(-2 * met.per_bar, 2 * met.per_bar + 1)]
    sc = max(scarti, key=lambda s: (punteggio(s), -abs(s)))
    per_chiave = {}
    for p, q, mano, x, n in fuori:
        per_chiave.setdefault((p, q - sc), (mano, x, n))

    giuste = mani = durate = 0
    orn_ok = orn_tot = gr_ok = gr_tot = arp_ok = arp_tot = 0
    for (p, q, mano), e in vero.items():
        if e.orn in ("trill", "mordent", "inv_mordent", "turn") and p == e.pitches[0]:
            orn_tot += 1
        if e.graces and p == e.pitches[0]:
            gr_tot += 1
        if e.orn == "arp" and p == min(e.pitches):
            arp_tot += 1
        hit = per_chiave.get((p, q))
        if not hit:
            continue
        giuste += 1
        m2, x, n = hit
        mani += int(m2 == mano)
        durate += int((x.fine - x.inizio) == e.dur)
        if e.orn in ("trill", "mordent", "inv_mordent", "turn") and p == e.pitches[0]:
            orn_ok += int(n.orn == e.orn)
        if e.graces and p == e.pitches[0]:
            gr_ok += int(sorted(n.graces) == sorted(e.graces))
        if e.orn == "arp" and p == min(e.pitches):
            arp_ok += int(sum(1 for y in x.note if y.arp) >= 2)
    falsi_orn = sum(1 for _, _, _, _, n in fuori if (n.orn or n.graces)) - orn_ok - gr_ok
    return {"note": len(vero), "giuste": giuste, "in_piu": max(0, len(fuori) - giuste),
            "mani": mani, "durate": durate, "scarto": sc,
            "orn": (orn_ok, orn_tot), "acc": (gr_ok, gr_tot), "arp": (arp_ok, arp_tot),
            "falsi_abbell": max(0, falsi_orn)}


# I brani lenti si provano col BPM indicato: sotto gli 80 circa la pulsazione
# è ambigua per chiunque (♩=60 a crome e ♩=120 a semiminime sono la stessa musica).
BRANI = [
    # nome,              tempo, bpm, tonica, modo, batt, kw, bpm indicato
    ("Sonatina",          "4/4", 104, 0, "magg", 16, dict(p_sedicesimi=0.20), False),
    ("Valzer",            "3/4", 138, 7, "magg", 24, dict(p_sedicesimi=0.05), False),
    ("Barcarola",         "6/8", 58, 9, "min", 16, dict(), True),
    ("Studio in terzine", "4/4", 88, 5, "magg", 16, dict(p_terzine=0.35, p_sedicesimi=0.05), False),
    ("Aria ornata",       "4/4", 76, 2, "min", 16, dict(p_abbell=0.30, p_arp=0.25, p_sedicesimi=0.10), True),
    ("Minuetto ornato",   "3/4", 112, 10, "magg", 24, dict(p_abbell=0.25, p_arp=0.15), False),
    ("Allegro",           "4/4", 132, 4, "min", 16, dict(p_sedicesimi=0.30), False),
    ("Adagio",            "4/4", 60, 3, "magg", 12, dict(p_abbell=0.20, p_terzine=0.15), True),
    ("Studio virtuosistico", "4/4", 116, 8, "min", 16, dict(virtuoso=True), False),
    ("Studio virtuos. presto", "4/4", 160, 1, "min", 16, dict(virtuoso=True), False),
]


def prova(nome, tempo, bpm, tonica, modo, batt, kw, seme, tempo_dato="auto",
          bpm_dato=None, livello="media", pdf_dir=None, rubato=0.05, jitter=0.010):
    if kw.get("virtuoso"):
        ev, bq, bar = componi_virtuoso(seme, tonica, modo, batt)
    else:
        ev, bq, bar = componi(seme, tempo, tonica, modo, batt, **kw)
    note, pedali = esegui(ev, bq, bpm, seme, rubato=rubato, jitter=jitter,
                          pedale=(seme % 2 == 0), bar=bar)
    muto = lambda *_: None
    ris = P.note_in_partitura(note, pedali, nome, "Banco di prova", livello,
                              bpm_dato, tempo_dato, {}, cb=muto)
    c = confronta(ev, ris)
    c["tempo_ok"] = ris["metrica"].sig == tempo
    c["tempo"] = ris["metrica"].sig
    c["bar"] = ris["metrica"].bar_ql
    c["bpm"] = ris["metrica"].bpm
    c["errori_misure"] = len(ris["errori_misure"])
    c["xml_ok"] = c["pdf_ok"] = False
    d = pdf_dir or P.tempfile.mkdtemp(prefix="banco_")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{nome.replace(' ', '_')}_{seme}")
    try:
        ris["score"].write("musicxml", fp=base + ".musicxml", makeNotation=False)
        c["xml_ok"] = True
        sotto = f"semiminima{' puntata' if bq != 1 else ''} = {ris['metrica'].bpm:.0f}"
        c["pdf_ok"] = P.pdf_verovio(base + ".musicxml", base + ".pdf", nome,
                                    "Banco di prova", sotto, cb=muto)
    except Exception as e:
        c["errore"] = f"{type(e).__name__}: {e}"
    if not pdf_dir:
        P.shutil.rmtree(d, ignore_errors=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="salva i PDF in ./prova_uscita")
    ap.add_argument("--semi", type=int, default=3, help="esecuzioni diverse per brano")
    ap.add_argument("--tempo-dato", action="store_true", help="passa il tempo a mano")
    ap.add_argument("--senza-bpm", action="store_true", help="non indica mai il BPM")
    ap.add_argument("--rapido", action="store_true",
                    help="un solo brano: serve all'installatore per collaudare il PDF")
    args = ap.parse_args()
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prova_uscita") if args.pdf else None

    tot = dict(note=0, giuste=0, mani=0, durate=0, in_piu=0, orn=[0, 0], acc=[0, 0],
               arp=[0, 0], falsi=0, tempo=0, fase=0, n=0, misure=0, pdf=0)
    print(f"\n  {'brano':<20}{'tempo':>12}{'note':>7}{'attacco':>9}{'durata':>8}"
          f"{'mani':>7}{'abbell.':>9}{'acciacc.':>9}{'arp.':>7}{'falsi':>6}  PDF")
    print("  " + "─" * 100)
    brani = [b for b in BRANI if b[0] == "Aria ornata"] if args.rapido else BRANI
    for nome, tempo, bpm, tonica, modo, batt, kw, dai_bpm in brani:
        for s in range(1 if args.rapido else args.semi):
            c = prova(nome, tempo, bpm, tonica, modo, batt, kw, seme=11 + s * 7,
                      tempo_dato=tempo if args.tempo_dato else "auto",
                      bpm_dato=bpm if (dai_bpm and not args.senza_bpm) else None, pdf_dir=pdf_dir)
            pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "—"
            fr = lambda t: f"{t[0]}/{t[1]}" if t[1] else "—"
            segno = "" if c["tempo_ok"] else " ✗"
            fase = "" if c["scarto"] % c["bar"] == 0 else f" (battere sfasato di {c['scarto']})"
            print(f"  {nome[:19]:<20}{c['tempo'] + segno + ' ' + format(c['bpm'], '.0f'):>12}{c['note']:>7}"
                  f"{pct(c['giuste'], c['note']):>9}{pct(c['durate'], c['giuste']):>8}"
                  f"{pct(c['mani'], c['giuste']):>7}{fr(c['orn']):>9}{fr(c['acc']):>9}"
                  f"{fr(c['arp']):>7}{c['falsi_abbell']:>6}  "
                  f"{'ok' if c['pdf_ok'] else 'NO'}{fase}"
                  f"{' MISURE ERRATE: ' + str(c['errori_misure']) if c['errori_misure'] else ''}"
                  f"{'  ' + c['errore'] if 'errore' in c else ''}")
            for k in ("note", "giuste", "mani", "durate", "in_piu"):
                tot[k] += c[k]
            for k in ("orn", "acc", "arp"):
                tot[k][0] += c[k][0]
                tot[k][1] += c[k][1]
            tot["falsi"] += c["falsi_abbell"]
            tot["tempo"] += int(c["tempo_ok"])
            tot["fase"] += int(c["scarto"] % c["bar"] == 0)
            tot["misure"] += c["errori_misure"]
            tot["pdf"] += int(c["pdf_ok"])
            tot["n"] += 1
    print("  " + "─" * 100)
    n = tot["n"]
    print(f"\n  Esecuzioni provate ............ {n}")
    print(f"  Note al posto giusto .......... {tot['giuste']} su {tot['note']} "
          f"({100 * tot['giuste'] / tot['note']:.1f}%)")
    print(f"  Durata scritta giusta ......... {100 * tot['durate'] / max(tot['giuste'], 1):.1f}%")
    print(f"  Mano giusta ................... {100 * tot['mani'] / max(tot['giuste'], 1):.1f}%")
    print(f"  Trilli/mordenti/gruppetti ..... {tot['orn'][0]} su {tot['orn'][1]}")
    print(f"  Acciaccature .................. {tot['acc'][0]} su {tot['acc'][1]}")
    print(f"  Arpeggiati .................... {tot['arp'][0]} su {tot['arp'][1]}")
    print(f"  Abbellimenti inventati ........ {tot['falsi']}")
    print(f"  Tempo riconosciuto ............ {tot['tempo']} su {n}")
    print(f"  Battere al posto giusto ....... {tot['fase']} su {n}")
    print(f"  Battute con durata errata ..... {tot['misure']}")
    print(f"  PDF generati .................. {tot['pdf']} su {n}\n")
    if pdf_dir:
        print(f"  PDF in: {pdf_dir}\n")


if __name__ == "__main__":
    main()
