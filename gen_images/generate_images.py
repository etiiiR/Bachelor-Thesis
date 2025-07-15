#!/usr/bin/env python3
"""
Skript zur Generierung verarbeiteter holographischer Pollenbilder.
Erstellt nur verarbeitete Bilder ohne Ripples in verschiedenen Größen.
"""

import os
import argparse
import shutil
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from holo_dataset import HolographicPollenDataset, RemoveRipples


def create_cameras_txt(target_dir: Path):
    """
    Erstellt eine cameras.txt Datei mit zwei orthogonalen Kamerawinkeln (90° Unterschied).
    Format für Pixel2Mesh/Pixel2MeshPlusPlus.
    
    Args:
        target_dir: Verzeichnis, in dem die cameras.txt erstellt werden soll
    """
    cameras_content = """0.000000 30.000000 0.000000 2.500000 30.000000
90.000000 30.000000 0.000000 2.500000 30.000000"""
    
    cameras_path = target_dir / "cameras.txt"
    with open(cameras_path, 'w') as f:
        f.write(cameras_content)
    print(f"    Erstellt cameras.txt mit orthogonalen Winkeln (0° und 90°)")


def copy_additional_files(target_dir: Path):
    """
    Kopiert pose, intrinsics.txt und near_far Dateien in das Zielverzeichnis.
    
    Args:
        target_dir: Zielverzeichnis für die zusätzlichen Dateien
    """
    current_dir = Path.cwd()
    files_to_copy = [
        ("pose", True),           # (filename, is_directory)
        ("intrinsics.txt", False),
        ("near_far", False)
    ]
    
    for filename, is_directory in files_to_copy:
        source_path = current_dir / filename
        target_path = target_dir / filename
        
        try:
            if source_path.exists():
                if is_directory:
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(source_path, target_path)
                    print(f"    Kopiert Ordner: {filename}")
                else:
                    shutil.copy2(source_path, target_path)
                    print(f"    Kopiert Datei: {filename}")
            else:
                print(f"    Warnung: {filename} nicht gefunden in {current_dir}")
        except Exception as e:
            print(f"    Fehler beim Kopieren von {filename}: {e}")


def process_and_resize_image(img_path: str, target_size: Tuple[int, int], 
                           to_rgb: bool = False) -> Image.Image:
    """
    Lädt ein Bild, normalisiert es, entfernt Ripples und skaliert es auf die Zielgröße.
    
    Args:
        img_path: Pfad zum Bild
        target_size: Zielgröße als (width, height)
        to_rgb: Ob das Bild zu RGB konvertiert werden soll
    
    Returns:
        Verarbeitetes PIL Image
    """
    # Lade und normalisiere das Bild
    img = Image.open(img_path)
    arr = np.array(img).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = np.zeros_like(arr)
    
    normalized_img = Image.fromarray(arr.astype(np.uint8), mode='L')
    
    # Entferne Ripples
    ripple_remover = RemoveRipples(method='otsu', max_scale=1.5)
    processed_img = ripple_remover(normalized_img)
    
    # Konvertiere zu RGB falls gewünscht
    if to_rgb:
        processed_img = processed_img.convert('RGB')
    
    # Skaliere auf Zielgröße
    resized_img = processed_img.resize(target_size, resample=Image.Resampling.LANCZOS)
    
    return resized_img


def generate_processed_images(output_dir: str = "processed_images", 
                            max_images_per_taxa: int = 6):
    """
    Generiert verarbeitete Bilder für alle Taxa in verschiedenen Größen.
    
    Args:
        output_dir: Ausgabeverzeichnis
        max_images_per_taxa: Maximale Anzahl Bilder pro Taxa
    """
    # Erstelle Ausgabeverzeichnisse
    output_path = Path(output_dir)
    size_224_path = output_path / "224x224"
    size_256_path = output_path / "256x256"
    rgb_224_path = output_path / "224x224rgb"  # Neue Struktur
    rgb_256_path = output_path / "rgb_256x256"
    
    size_224_path.mkdir(parents=True, exist_ok=True)
    size_256_path.mkdir(parents=True, exist_ok=True)
    rgb_224_path.mkdir(parents=True, exist_ok=True)
    rgb_256_path.mkdir(parents=True, exist_ok=True)
    
    # Setze DATA_DIR_PATH falls nicht gesetzt
    if not os.getenv("DATA_DIR_PATH"):
        os.environ["DATA_DIR_PATH"] = str(Path.cwd())
    
    print(f"DATA_DIR_PATH: {os.getenv('DATA_DIR_PATH')}")
    
    # Lade Dataset
    dataset = HolographicPollenDataset()
    print(f"Gefundene {len(dataset.pairs)} Bildpaare insgesamt")
    
    # Lade die Auswahl aus poleno_selection.json
    selection_file = Path("poleno_selection.json")
    if not selection_file.exists():
        print(f"❌ Fehler: {selection_file} nicht gefunden!")
        return
    
    with open(selection_file, "r", encoding="utf-8") as f:
        selection = json.load(f)
    
    print(f"Auswahl aus poleno_selection.json geladen:")
    for taxa, names in selection.items():
        print(f"  {taxa}: {len(names)} Bilder")
    
    # Filtere Paare basierend auf poleno_selection.json
    # Erstelle ein Set für O(1) Lookup
    allowed = {
        (taxa, name) for taxa, names in selection.items() for name in names
    }
    
    # Filtere die Bildpaare
    filtered_pairs = []
    for p0, p1, taxa in dataset.pairs:
        base_name_p0 = Path(p0).stem
        if (taxa, base_name_p0) in allowed:
            filtered_pairs.append((p0, p1, taxa))
    
    print(f"Nach Filterung: {len(filtered_pairs)} Bildpaare")
    
    # Gruppiere gefilterte Paare nach Taxa
    taxa_pairs = {}
    for p0, p1, taxa in filtered_pairs:
        if taxa not in taxa_pairs:
            taxa_pairs[taxa] = []
        taxa_pairs[taxa].append((p0, p1))
    
    print(f"Gefundene Taxa: {list(taxa_pairs.keys())}")
    
    total_processed = 0
    
    for taxa in taxa_pairs:
        print(f"\nVerarbeite Taxa: {taxa}")
        
        # Erstelle Unterordner für dieses Taxa
        taxa_224_path = size_224_path / taxa
        taxa_256_path = size_256_path / taxa
        taxa_224_path.mkdir(exist_ok=True)
        taxa_256_path.mkdir(exist_ok=True)
        
        # Begrenze auf max_images_per_taxa Paare (falls mehr als in der Auswahl vorhanden)
        pairs_to_process = taxa_pairs[taxa][:max_images_per_taxa]
        
        for i, (p0, p1) in enumerate(pairs_to_process):
            try:
                # Extrahiere die komplette Zeit aus dem ersten Bild des Paars
                base_name = Path(p0).stem
                datetime_part = ""
                if "image_pairs" in base_name:
                    # Suche nach dem Muster: poleno-XX_YYYY-MM-DD_HH.MM.SS.microseconds
                    parts = base_name.split("_")
                    datetime_components = []
                    for j, part in enumerate(parts):
                        if part.startswith("poleno-"):
                            # Sammle die nächsten 3 Teile: poleno-XX, YYYY-MM-DD, HH.MM.SS.microseconds
                            if j < len(parts) - 2:
                                poleno_part = part.split("-")[1]  # Extrahiere nur die Nummer nach poleno-
                                date_part = parts[j + 1]         # YYYY-MM-DD
                                time_part = parts[j + 2]         # HH.MM.SS.microseconds
                                datetime_part = f"{poleno_part}_{date_part}_{time_part}"
                            break
                
                # Erstelle RGB-Ordner für verschiedene Strukturen
                # 224x224rgb: pollenname_date_time.milliseconds/
                rgb_224_name = f"{taxa}_{datetime_part}" if datetime_part else f"{taxa}_{i+1:02d}"
                rgb_224_dir = rgb_224_path / rgb_224_name
                rgb_224_dir.mkdir(parents=True, exist_ok=True)
                
                # 256x256: taxa__nummer_date_time/rgb/
                pair_name = f"{taxa}__{i+1:02d}_{datetime_part}" if datetime_part else f"{taxa}__{i+1:02d}"
                rgb_pair_256_path = rgb_256_path / pair_name / "rgb"
                rgb_pair_256_path.mkdir(parents=True, exist_ok=True)
                
                # Kopiere zusätzliche Dateien in den 256x256 RGB-Ordner
                pair_256_root = rgb_256_path / pair_name
                copy_additional_files(pair_256_root)
                
                # Erstelle cameras.txt für 224x224rgb mit orthogonalen Winkeln
                create_cameras_txt(rgb_224_dir)
                
                # Verarbeite beide Bilder des Paars mit klarer Kennzeichnung
                for j, (img_path, pair_id) in enumerate([(p0, "pair0"), (p1, "pair1")]):
                    # Erstelle aussagekräftige Dateinamen
                    base_name = Path(img_path).stem
                    # Entferne den langen Teil und behalte nur das Wesentliche
                    if "image_pairs" in base_name:
                        short_name = base_name.split("image_pairs")[0].rstrip("_.")
                    else:
                        short_name = base_name
                    
                    filename_224 = f"{taxa}_{i+1:02d}_{pair_id}_224x224.png"
                    filename_256 = f"{taxa}_{i+1:02d}_{pair_id}_256x256.png"
                    
                    # RGB Dateinamen (000000.png für pair0, 000001.png für pair1)
                    rgb_filename = f"{j:06d}.png"
                    
                    # Verarbeite und speichere in 224x224 (Graustufen)
                    img_224 = process_and_resize_image(img_path, (224, 224), to_rgb=False)
                    img_224.save(taxa_224_path / filename_224)
                    
                    # Speichere in 224x224rgb (RGB mit orthogonalen Kamerawinkeln)
                    img_224_rgb = img_224.convert('RGB')
                    img_224_rgb.save(rgb_224_dir / rgb_filename)
                    
                    # Verarbeite und speichere in 256x256 (RGB)
                    img_256 = process_and_resize_image(img_path, (256, 256), to_rgb=True)
                    img_256.save(taxa_256_path / filename_256)
                    img_256.save(rgb_pair_256_path / rgb_filename)
                    
                    total_processed += 1
                    
                print(f"  Paar {i+1}/{len(pairs_to_process)} verarbeitet")
                
            except Exception as e:
                print(f"  Fehler beim Verarbeiten von Paar {i+1}: {e}")
                continue
        
        processed_count = len(pairs_to_process) * 2  # 2 Bilder pro Paar
        print(f"  {processed_count} Bilder für {taxa} generiert")
    
    print(f"\n✅ Verarbeitung abgeschlossen!")
    print(f"Insgesamt {total_processed} Bilder verarbeitet")
    print(f"Ausgabeverzeichnisse:")
    print(f"  - 224x224: {size_224_path}")
    print(f"  - 256x256: {size_256_path}")
    print(f"  - 224x224rgb: {rgb_224_path}")
    print(f"  - RGB 256x256: {rgb_256_path}")
    
    # Zeige Zusammenfassung pro Taxa
    print(f"\nZusammenfassung pro Taxa:")
    for taxa in taxa_pairs:
        count_224 = len(list((size_224_path / taxa).glob("*.png")))
        count_256 = len(list((size_256_path / taxa).glob("*.png")))
        count_rgb_224 = len(list(rgb_224_path.glob(f"{taxa}_*")))
        count_rgb_256 = len(list(rgb_256_path.glob(f"{taxa}__*")))
        print(f"  {taxa}: {count_224} Bilder (224x224), {count_256} Bilder (256x256)")
        print(f"    RGB Ordner: {count_rgb_224} (224x224rgb), {count_rgb_256} (256x256)")


def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(
        description="Generiere verarbeitete holographische Pollenbilder ohne Ripples"
    )
    parser.add_argument("--output-dir", "-o", default="processed_images",
                      help="Ausgabeverzeichnis für verarbeitete Bilder")
    parser.add_argument("--max-images", "-m", type=int, default=6,
                      help="Maximale Anzahl Bildpaare pro Taxa (Standard: 6)")
    
    args = parser.parse_args()
    
    print("Starte Generierung verarbeiteter Bilder...")
    print(f"Ausgabeverzeichnis: {args.output_dir}")
    print(f"Max. Bilder pro Taxa: {args.max_images}")
    print(f"Zielgrößen: 224x224 und 256x256 Pixel")
    print(f"Verarbeitung: Normalisierung + Ripple-Entfernung + Skalierung")
    
    try:
        generate_processed_images(args.output_dir, args.max_images)
        return 0
        
    except Exception as e:
        print(f"❌ Fehler beim Generieren der Bilder: {e}")
        return 1


if __name__ == "__main__":
    exit(main())