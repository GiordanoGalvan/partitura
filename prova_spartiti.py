#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROVA SU SPARTITI VERI — il collaudo più severo della scrittura.

Prende spartiti di pianoforte veri (tutti di pubblico dominio, inclusi nella
libreria music21), li «esegue» come farebbe un pianista — rubato, attacchi
imprecisi, accordi non perfettamente insieme, acciaccature suonate — e chiede
a partitura.py di riscriverli. Poi confronta nota per nota con l'originale.

A differenza di banco_prova.py qui la musica è vera: più voci nella stessa
mano, sincopi, salti, legature, anacrusi.

    python prova_spartiti.py
"""

import os
import sys
import argparse
from fractions import Fraction as F
from typing import List, Tuple

# Su Windows la console e i tubi fra processi usano cp1252: senza questo, un
# carattere come «─» o «→» fa fallire la stampa.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import partitura as P
import banco_prova as B

P.STAMPA = False

# spartito, titolo, BPM dell'esecuzione (semiminima)
SPARTITI = [
    ("mozart/k545/movement1_exposition", "Mozart, Sonata K545 (esposizione)", 126),
    ("joplin/maple_leaf_rag", "Joplin, Maple Leaf Rag", 96),
    ("chopin/mazurka06-2", "Chopin, Mazurka op. 6 n. 2", 132),
    ("cpebach/h186", "C.P.E. Bach, H 186", 84),
    ("schumann_clara/polonaise_op1n1", "C. Schumann, Polacca op. 1 n. 1", 96),
    ("schumann_clara/polonaise_op1n2", "C. Schumann, Polacca op. 1 n. 2", 96),
    ("schumann_clara/polonaise_op1n3", "C. Schumann, Polacca op. 1 n. 3", 92),
    ("schumann_clara/polonaise_op1n4", "C. Schumann, Polacca op. 1 n. 4", 96),
]


def leggi_spartito(nome: str) -> Tuple[List[B.GT], F, F, str]:
    """Ritorna (eventi veri, durata del movimento, durata della battuta, tempo)."""
    from music21 import corpus, meter, stream
    sc = corpus.parse(nome)
    ts = sc.recurse().getElementsByClass(meter.TimeSignature)[0]
    tempo = ts.ratioString
    bar = F(ts.barDuration.quarterLength).limit_denominator(96)
    composto = ts.denominator == 8 and ts.numerator in (6, 9, 12)
    bq = F(3, 2) if composto else F(1)

    # anacrusi: la prima battuta può essere incompleta
    prima = sc.parts[0].getElementsByClass(stream.Measure)[0]
    lung = F(prima.duration.quarterLength).limit_denominator(96)
    sposta = (bar - lung) if 0 < lung < bar else F(0)

    ev: List[B.GT] = []
    for mano, part in enumerate(sc.parts[:2]):
        piatto = part.stripTies().flatten().notes
        attesa: List[int] = []
        gruppi = {}
        for n in piatto:
            alt = [p.midi for p in n.pitches]
            if n.duration.isGrace:
                attesa.extend(alt)
                continue
            q = F(n.offset).limit_denominator(96) + sposta
            d = F(n.duration.quarterLength).limit_denominator(96)
            if d <= 0:
                continue
            k = (q, d)
            if k not in gruppi:
                gruppi[k] = B.GT(q, d, [], mano)
            g = gruppi[k]
            for p in alt:
                if p not in g.pitches:
                    g.pitches.append(p)
            if attesa and not g.graces:
                g.graces = attesa[-3:]
            attesa = []
        for g in gruppi.values():
            g.pitches.sort()
            # l'acciaccatura si attacca alla nota più vicina: nel confronto
            # conta la nota a cui il programma dovrebbe appenderla
            ev.append(g)
    # stesso tasto scritto due volte nello stesso istante (due voci all'unisono)
    visti, puliti = set(), []
    for e in sorted(ev, key=lambda e: (e.q, e.hand, -e.dur)):
        e.pitches = [p for p in e.pitches if (p, e.q) not in visti]
        for p in e.pitches:
            visti.add((p, e.q))
        if e.pitches:
            puliti.append(e)
    return puliti, bq, bar, tempo


def equivalente(vero: str, trovato: str) -> bool:
    """2/4 e 4/4 sono la stessa musica con le stanghette a distanza doppia."""
    pari = {"2/4", "4/4"}
    return vero == trovato or (vero in pari and trovato in pari)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semi", type=int, default=2, help="esecuzioni diverse per spartito")
    ap.add_argument("--pdf", action="store_true", help="salva i PDF in ./prova_uscita")
    ap.add_argument("--solo", default="", help="solo gli spartiti il cui nome contiene questo testo")
    args = ap.parse_args()
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prova_uscita") if args.pdf else None
    if pdf_dir:
        os.makedirs(pdf_dir, exist_ok=True)

    tot = dict(note=0, giuste=0, mani=0, durate=0, acc=[0, 0], falsi=0, tempo=0,
               fase=0, n=0, misure=0, pdf=0, in_piu=0)
    print(f"\n  {'spartito':<36}{'tempo':>10}{'note':>7}{'attacco':>9}{'durata':>8}"
          f"{'mani':>7}{'acciacc.':>10}{'falsi':>7}  PDF")
    print("  " + "─" * 98)
    for nome, titolo, bpm in SPARTITI:
        if args.solo and args.solo.lower() not in (nome + titolo).lower():
            continue
        ev, bq, bar, tempo = leggi_spartito(nome)
        for s in range(args.semi):
            seme = 5 + 9 * s
            note, pedali = B.esegui(ev, bq, bpm, seme, pedale=(s % 2 == 1), bar=bar)
            ris = P.note_in_partitura(note, pedali, titolo, "", "media", None, "auto", {})
            c = B.confronta(ev, ris)
            met = ris["metrica"]
            t_ok = equivalente(tempo, met.sig)
            f_ok = c["scarto"] % min(bar, met.bar_ql) == 0
            pdf_ok = False
            try:
                d = pdf_dir or P.tempfile.mkdtemp(prefix="spartiti_")
                base = os.path.join(d, nome.replace("/", "_") + f"_{seme}")
                ris["score"].write("musicxml", fp=base + ".musicxml", makeNotation=False)
                pdf_ok = P.pdf_verovio(base + ".musicxml", base + ".pdf", titolo, "",
                                       f"semiminima = {met.bpm:.0f}")
                if not pdf_dir:
                    P.shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                print("   errore PDF:", type(e).__name__, str(e)[:100])
            pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "—"
            print(f"  {titolo[:35]:<36}{met.sig + ('' if t_ok else ' ✗') + ' ' + format(met.bpm, '.0f'):>10}"
                  f"{c['note']:>7}{pct(c['giuste'], c['note']):>9}{pct(c['durate'], c['giuste']):>8}"
                  f"{pct(c['mani'], c['giuste']):>7}{str(c['acc'][0]) + '/' + str(c['acc'][1]):>10}"
                  f"{c['falsi_abbell']:>7}  {'ok' if pdf_ok else 'NO'}"
                  f"{'' if f_ok else ' (battere sfasato di ' + str(c['scarto']) + ')'}"
                  f"{' MISURE ERRATE: ' + str(len(ris['errori_misure'])) if ris['errori_misure'] else ''}")
            for k in ("note", "giuste", "mani", "durate", "in_piu"):
                tot[k] += c[k]
            tot["acc"][0] += c["acc"][0]
            tot["acc"][1] += c["acc"][1]
            tot["falsi"] += c["falsi_abbell"]
            tot["tempo"] += int(t_ok)
            tot["fase"] += int(f_ok)
            tot["misure"] += len(ris["errori_misure"])
            tot["pdf"] += int(pdf_ok)
            tot["n"] += 1
    print("  " + "─" * 98)
    n = max(tot["n"], 1)
    print(f"\n  Esecuzioni provate ............ {tot['n']}")
    print(f"  Note al posto giusto .......... {tot['giuste']} su {tot['note']} "
          f"({100 * tot['giuste'] / max(tot['note'], 1):.1f}%)")
    print(f"  Durata scritta giusta ......... {100 * tot['durate'] / max(tot['giuste'], 1):.1f}%")
    print(f"  Mano giusta ................... {100 * tot['mani'] / max(tot['giuste'], 1):.1f}%")
    print(f"  Acciaccature .................. {tot['acc'][0]} su {tot['acc'][1]}")
    print(f"  Abbellimenti inventati ........ {tot['falsi']}")
    print(f"  Tempo riconosciuto ............ {tot['tempo']} su {n}")
    print(f"  Battere al posto giusto ....... {tot['fase']} su {n}")
    print(f"  Battute con durata errata ..... {tot['misure']}")
    print(f"  PDF generati .................. {tot['pdf']} su {n}\n")


if __name__ == "__main__":
    main()
