# Partitura

Da una registrazione di pianoforte (file audio, MIDI o link YouTube) a una partitura PDF, con verifica sull'audio originale.

## Installazione

Scarica `Partitura-Setup-x.y.exe` dalla pagina **Releases** e lancialo. Non serve installare Python.

## Come escono le versioni

Ogni modifica caricata su `main` viene compilata e collaudata su un server Windows di GitHub (scheda **Actions**): l'installatore si scarica dagli *artifacts* dell'esecuzione.
Per pubblicare una versione ufficiale: alza `VERSIONE` in `partitura.py` e crea un tag `vX.Y` — l'installatore finisce in **Releases**.

Il collaudo automatico, a ogni modifica:
1. banco di prova sulla notazione (brani con soluzione nota);
2. spartiti veri di pubblico dominio;
3. trascrizione completa da audio con Transkun;
4. l'app compilata, avviata davvero, deve produrre il PDF da MIDI e da audio.

Se uno di questi fallisce, l'installatore non viene prodotto.

## Uso senza installatore (sviluppo)

`INSTALLA.bat` crea un ambiente Python locale; poi `PARTITURA.bat`.
