@echo off
REM OBJ Mesh Smoothing Tool - Batch-Datei für Windows
REM Diese Datei führt das Python-Skript mit Standard-Einstellungen aus

echo ===================================
echo OBJ Mesh Smoothing Tool
echo ===================================
echo.

REM Aktiviere die virtuelle Umgebung und führe das Skript aus
echo Starte Glättung aller OBJ-Dateien...
echo Verwende Taubin-Glättung mit 10 Iterationen
echo Ausgabe-Ordner: smoothed_obj_files
echo.

C:/Users/super/Documents/Github/sequoia/TestEvaluationPipeline/data/.venv/Scripts/python.exe smooth.py

echo.
echo Fertig! Überprüfen Sie den Ordner 'smoothed_obj_files' für die Ergebnisse.
pause
