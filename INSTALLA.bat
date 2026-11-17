@echo off
title Setup PARTITURA 2
cd /d "%~dp0"

echo.
echo  ================================================================
echo   SETUP PARTITURA 2
echo  ================================================================
echo.
echo  Installa tutto il necessario e fa una prova. Puoi lasciarlo
echo  lavorare da solo: 10-20 minuti a seconda della connessione.
echo.

rem Serve un Python qualsiasi solo per far partire l'installatore:
rem la versione giusta se la procura lui.
where py >nul 2>&1
if %errorlevel%==0 (
    py setup_partitura.py
    goto :fine
)

where python >nul 2>&1
if %errorlevel%==0 (
    python setup_partitura.py
    goto :fine
)

echo.
echo  [!] Nessun Python trovato sul sistema.
echo.
echo      Scarica Python 3.12 da https://www.python.org/downloads/
echo      Durante l'installazione SPUNTA "Add python.exe to PATH".
echo      Poi rilancia questo file.
echo.
pause

:fine
