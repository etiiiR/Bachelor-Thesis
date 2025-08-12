#!/usr/bin/env python3
"""
Beispiel für die Verwendung des OBJ Smoothing Tools
Dieses Skript zeigt verschiedene Anwendungsbeispiele
"""

import os
import subprocess
from pathlib import Path

# Python-Pfad zur virtuellen Umgebung
PYTHON_PATH = "C:/Users/super/Documents/Github/sequoia/TestEvaluationPipeline/data/.venv/Scripts/python.exe"
SCRIPT_PATH = "smooth.py"

def run_smoothing(input_dir, output_dir, method="taubin", iterations=10, lambda_param=0.6):
    """
    Führt das Smoothing-Skript mit angegebenen Parametern aus
    """
    cmd = [
        PYTHON_PATH, SCRIPT_PATH,
        input_dir, output_dir,
        "--method", method,
        "--iterations", str(iterations),
        "--lambda", str(lambda_param)
    ]
    
    print(f"Führe aus: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Erfolgreich abgeschlossen")
    else:
        print("✗ Fehler aufgetreten:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    print("=== OBJ Smoothing Tool - Beispiele ===")
    print()
    
    examples = [
        {
            "name": "Hunyuan3D Ordner",
            "input": "Hunyuan3D",
            "output": "smoothed_hunyuan3d",
            "method": "taubin",
            "iterations": 8,
            "lambda": 0.5
        },
        {
            "name": "Spherical Prior Ordner (leichte Glättung)",
            "input": "refine_p2mp_augmentation_2_inputs_use_spherical_prior",
            "output": "smoothed_spherical_prior_light",
            "method": "taubin",
            "iterations": 5,
            "lambda": 0.3
        },
        {
            "name": "Two-Views Ordner (starke Glättung)",
            "input": "Hunyuan3D-two-views",
            "output": "smoothed_two_views_heavy",
            "method": "taubin",
            "iterations": 15,
            "lambda": 0.8
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Eingabe: {example['input']}")
        print(f"   Ausgabe: {example['output']}")
        print(f"   Methode: {example['method']}")
        print(f"   Iterationen: {example['iterations']}")
        print(f"   Lambda: {example['lambda']}")
        print()
    
    # Benutzer nach Auswahl fragen
    try:
        choice = input("Wählen Sie ein Beispiel (1-3) oder 'a' für alle: ").strip().lower()
        
        if choice == 'a':
            # Alle Beispiele ausführen
            for example in examples:
                print(f"\n--- Verarbeite: {example['name']} ---")
                run_smoothing(
                    example['input'],
                    example['output'],
                    example['method'],
                    example['iterations'],
                    example['lambda']
                )
        elif choice in ['1', '2', '3']:
            # Einzelnes Beispiel ausführen
            example = examples[int(choice) - 1]
            print(f"\n--- Verarbeite: {example['name']} ---")
            run_smoothing(
                example['input'],
                example['output'],
                example['method'],
                example['iterations'],
                example['lambda']
            )
        else:
            print("Ungültige Auswahl.")
            
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    main()
