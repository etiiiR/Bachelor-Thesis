# OBJ Mesh Smoothing Tool

Dieses Python-Skript glättet OBJ-Dateien und speichert die geglätteten Versionen in einem neuen Ordner.

## Installation

Stellen Sie sicher, dass Python 3.7+ installiert ist, und installieren Sie die erforderlichen Pakete:

```bash
pip install -r requirements.txt
```

## Verwendung

### Einfache Verwendung (ohne Argumente)

Führen Sie das Skript einfach aus dem Datenverzeichnis aus:

```bash
python smooth.py
```

Dies wird:
- Alle OBJ-Dateien im aktuellen Verzeichnis und Unterverzeichnissen finden
- Sie mit der Taubin-Glättung (10 Iterationen) glätten
- Die geglätteten Dateien im Ordner `smoothed_obj_files` speichern

### Erweiterte Verwendung (mit Argumenten)

```bash
python smooth.py <input_directory> <output_directory> [optionen]
```

#### Beispiele:

```bash
# Grundlegende Verwendung
python smooth.py ./Hunyuan3D ./smoothed_hunyuan

# Mit spezifischer Glättungsmethode
python smooth.py ./refine_p2mp_augmentation_2_inputs_use_spherical_prior ./smoothed --method taubin --iterations 15

# Nur oberste Ebene (nicht rekursiv)
python smooth.py ./data ./output --no-recursive

# Mit angepassten Parametern
python smooth.py ./input ./output --method laplacian --iterations 20 --lambda 0.3
```

## Optionen

- `--method`: Glättungsmethode (`laplacian`, `taubin`, `simple`)
  - `laplacian`: Standard Laplacian-Glättung
  - `taubin`: Taubin-Glättung (bewahrt Volumen besser) **Empfohlen**
  - `simple`: Einfache Glättung

- `--iterations`: Anzahl der Glättungsiterationen (Standard: 5)

- `--lambda`: Lambda-Parameter für die Glättung (Standard: 0.5)

- `--no-recursive`: Nur Dateien im Hauptverzeichnis verarbeiten, keine Unterordner

## Glättungsmethoden im Detail

### Laplacian-Glättung
- Schnell und einfach
- Kann das Volumen des Meshes reduzieren
- Gut für leichte Glättung

### Taubin-Glättung (Empfohlen)
- Bewahrt das ursprüngliche Volumen besser
- Reduziert Schrumpfung des Meshes
- Bessere Qualität für 3D-Modelle

### Simple-Glättung
- Sehr einfacher Algorithmus
- Am schnellsten
- Geringste Qualität

## Ausgabeformat

Die geglätteten OBJ-Dateien werden mit dem Suffix `_smoothed` gespeichert und behalten ihre ursprüngliche Verzeichnisstruktur bei.

Beispiel:
- Eingabe: `refine_p2mp/pollen_17781.obj`
- Ausgabe: `smoothed_obj_files/refine_p2mp/pollen_17781_smoothed.obj`

## Empfohlene Einstellungen

Für Pollen-Modelle:
```bash
python smooth.py ./input ./output --method taubin --iterations 10 --lambda 0.6
```

Für detaillierte Modelle (wenig Glättung):
```bash
python smooth.py ./input ./output --method taubin --iterations 5 --lambda 0.3
```

Für stark verrauschte Modelle:
```bash
python smooth.py ./input ./output --method taubin --iterations 20 --lambda 0.8
```
