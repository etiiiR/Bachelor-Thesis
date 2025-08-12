# OBJ Mesh Smoothing Tool - PowerShell-Skript
# Diese Datei führt das Python-Skript mit Standard-Einstellungen aus

Write-Host "===================================" -ForegroundColor Green
Write-Host "OBJ Mesh Smoothing Tool" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""

# Setze den Arbeitsordner
Set-Location "c:\Users\super\Documents\Github\sequoia\TestEvaluationPipeline\data"

Write-Host "Starte Glättung aller OBJ-Dateien..." -ForegroundColor Yellow
Write-Host "Verwende Taubin-Glättung mit 10 Iterationen" -ForegroundColor Yellow
Write-Host "Ausgabe-Ordner: smoothed_obj_files" -ForegroundColor Yellow
Write-Host ""

# Führe das Python-Skript aus
& "C:/Users/super/Documents/Github/sequoia/TestEvaluationPipeline/data/.venv/Scripts/python.exe" smooth.py

Write-Host ""
Write-Host "Fertig! Überprüfen Sie den Ordner 'smoothed_obj_files' für die Ergebnisse." -ForegroundColor Green
Read-Host "Drücken Sie Enter um fortzufahren"
