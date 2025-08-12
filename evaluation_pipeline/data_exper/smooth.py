import os
import sys
import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def smooth_mesh(mesh, method='laplacian', iterations=5, lambda_param=0.5, subdivision_levels=0, remesh=False):
    """
    Glättet ein Mesh mit verschiedenen Methoden
    
    Args:
        mesh: Open3D TriangleMesh object
        method: Glättungsmethode ('laplacian', 'taubin', 'simple')
        iterations: Anzahl der Glättungsiterationen
        lambda_param: Lambda-Parameter für Laplacian-Glättung
        subdivision_levels: Anzahl der Subdivision-Stufen (0 = keine)
        remesh: Ob das Mesh neu erstellt werden soll für bessere Topologie
    
    Returns:
        Geglättetes Mesh
    """
    # Deep copy erstellen
    import copy
    mesh_smoothed = copy.deepcopy(mesh)
    
    # Mesh-Bereinigung vor der Glättung
    mesh_smoothed.remove_duplicated_vertices()
    mesh_smoothed.remove_duplicated_triangles()
    mesh_smoothed.remove_degenerate_triangles()
    mesh_smoothed.remove_non_manifold_edges()
    
    # Remeshing für bessere Topologie (experimentell)
    if remesh:
        logger.info("Führe Remeshing durch für bessere Mesh-Topologie")
        try:
            # Berechne die durchschnittliche Kantenlänge
            import numpy as np
            edges = np.asarray(mesh_smoothed.get_edge_list())
            vertices = np.asarray(mesh_smoothed.vertices)
            edge_lengths = []
            for edge in edges:
                v1, v2 = edge
                length = np.linalg.norm(vertices[v1] - vertices[v2])
                edge_lengths.append(length)
            avg_edge_length = np.mean(edge_lengths)
            
            # Isotropes Remeshing
            mesh_smoothed = mesh_smoothed.simplify_quadric_decimation(target_number_of_triangles=len(mesh_smoothed.triangles))
            logger.info(f"Remeshing abgeschlossen: {len(mesh_smoothed.vertices)} vertices, {len(mesh_smoothed.triangles)} faces")
        except Exception as e:
            logger.warning(f"Remeshing fehlgeschlagen: {e}")
    
    # Optionale Subdivision für bessere Glättung
    if subdivision_levels > 0:
        logger.info(f"Führe {subdivision_levels} Subdivision-Stufen durch")
        for i in range(subdivision_levels):
            mesh_smoothed = mesh_smoothed.subdivide_loop(number_of_iterations=1)
            logger.info(f"Subdivision Stufe {i+1}: {len(mesh_smoothed.vertices)} vertices, {len(mesh_smoothed.triangles)} faces")
    
    # Mehrfache Glättung mit verschiedenen Methoden für bessere Ergebnisse
    if method == 'laplacian':
        # Laplacian-Glättung
        mesh_smoothed = mesh_smoothed.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=lambda_param
        )
    elif method == 'taubin':
        # Taubin-Glättung (bewahrt das Volumen besser)
        mesh_smoothed = mesh_smoothed.filter_smooth_taubin(
            number_of_iterations=iterations,
            lambda_filter=lambda_param,
            mu=-lambda_param - 0.01
        )
        # Zusätzliche leichte Laplacian-Glättung für noch glattere Oberflächen
        mesh_smoothed = mesh_smoothed.filter_smooth_laplacian(
            number_of_iterations=max(1, iterations // 4),
            lambda_filter=lambda_param * 0.3
        )
    elif method == 'simple':
        # Einfache Glättung
        mesh_smoothed = mesh_smoothed.filter_smooth_simple(
            number_of_iterations=iterations
        )
    else:
        logger.warning(f"Unbekannte Glättungsmethode: {method}. Verwende Laplacian.")
        mesh_smoothed = mesh_smoothed.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=lambda_param
        )
    
    return mesh_smoothed

def process_obj_file(input_path, output_path, smooth_method='laplacian', iterations=5, lambda_param=0.5, color=(0.7, 0.7, 0.7), subdivision_levels=0, remesh=False):
    """
    Verarbeitet eine einzelne OBJ-Datei
    
    Args:
        input_path: Pfad zur Eingabe-OBJ-Datei
        output_path: Pfad zur Ausgabe-OBJ-Datei
        smooth_method: Glättungsmethode
        iterations: Anzahl der Iterationen
        lambda_param: Lambda-Parameter
        color: RGB-Farbe als Tuple (r, g, b) mit Werten zwischen 0 und 1
        subdivision_levels: Anzahl der Subdivision-Stufen
        remesh: Ob Remeshing angewendet werden soll
    """
    try:
        # OBJ-Datei laden
        logger.info(f"Lade Mesh: {input_path}")
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        
        if len(mesh.vertices) == 0:
            logger.warning(f"Mesh ist leer oder konnte nicht geladen werden: {input_path}")
            return False
        
        # Mesh-Informationen ausgeben
        logger.info(f"Mesh hat {len(mesh.vertices)} Vertices und {len(mesh.triangles)} Faces")
        
        # Sicherstellen, dass Vertex-Normalen berechnet sind
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Mesh glätten
        logger.info(f"Glätte Mesh mit Methode: {smooth_method}, Iterationen: {iterations}")
        mesh_smoothed = smooth_mesh(mesh, method=smooth_method, iterations=iterations, lambda_param=lambda_param, subdivision_levels=subdivision_levels, remesh=remesh)
        
        # Normalen neu berechnen nach der Glättung
        mesh_smoothed.compute_vertex_normals()
        
        # Farbe zu allen Vertices hinzufügen
        mesh_color = np.array(color)  # Verwendet die angegebene Farbe
        vertex_colors = np.tile(mesh_color, (len(mesh_smoothed.vertices), 1))
        mesh_smoothed.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        logger.info(f"Farbe angewendet: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        
        # Ausgabeordner erstellen falls nicht vorhanden
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Geglättetes Mesh speichern
        logger.info(f"Speichere geglättetes Mesh: {output_path}")
        success = o3d.io.write_triangle_mesh(str(output_path), mesh_smoothed)
        
        if not success:
            logger.error(f"Fehler beim Speichern: {output_path}")
            return False
        
        logger.info(f"Erfolgreich verarbeitet: {input_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Fehler beim Verarbeiten von {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, smooth_method='laplacian', iterations=5, lambda_param=0.5, recursive=True, color=(0.7, 0.7, 0.7), subdivision_levels=0, remesh=False):
    """
    Verarbeitet alle OBJ-Dateien in einem Verzeichnis
    
    Args:
        input_dir: Eingabeverzeichnis
        output_dir: Ausgabeverzeichnis
        smooth_method: Glättungsmethode
        iterations: Anzahl der Iterationen
        lambda_param: Lambda-Parameter
        recursive: Rekursiv durch Unterordner
        color: RGB-Farbe als Tuple (r, g, b) mit Werten zwischen 0 und 1
        subdivision_levels: Anzahl der Subdivision-Stufen
        remesh: Ob Remeshing angewendet werden soll
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Eingabeverzeichnis existiert nicht: {input_dir}")
        return
    
    # Alle OBJ-Dateien finden
    if recursive:
        obj_files = list(input_path.rglob("*.obj"))
    else:
        obj_files = list(input_path.glob("*.obj"))
    
    if not obj_files:
        logger.warning(f"Keine OBJ-Dateien gefunden in: {input_dir}")
        return
    
    logger.info(f"Gefunden {len(obj_files)} OBJ-Dateien zum Verarbeiten")
    
    successful = 0
    failed = 0
    
    for obj_file in obj_files:
        # Relative Pfadstruktur beibehalten
        relative_path = obj_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        # Dateiname mit "_smoothed" erweitern
        output_file = output_file.with_stem(f"{output_file.stem}_smoothed")
        
        if process_obj_file(obj_file, output_file, smooth_method, iterations, lambda_param, color, subdivision_levels, remesh):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Verarbeitung abgeschlossen: {successful} erfolgreich, {failed} fehlgeschlagen")

def main():
    parser = argparse.ArgumentParser(description="Glättet OBJ-Dateien und speichert sie in einem neuen Ordner")
    parser.add_argument("input", help="Eingabeverzeichnis mit OBJ-Dateien")
    parser.add_argument("output", help="Ausgabeverzeichnis für geglättete OBJ-Dateien")
    parser.add_argument("--method", choices=['laplacian', 'taubin', 'simple'], 
                       default='laplacian', help="Glättungsmethode (default: laplacian)")
    parser.add_argument("--iterations", type=int, default=5, 
                       help="Anzahl der Glättungsiterationen (default: 5)")
    parser.add_argument("--lambda", dest='lambda_param', type=float, default=0.5, 
                       help="Lambda-Parameter für Glättung (default: 0.5)")
    parser.add_argument("--color", nargs=3, type=float, default=[0.7, 0.7, 0.7],
                       help="RGB-Farbe für das Mesh (3 Werte zwischen 0 und 1, default: 0.7 0.7 0.7 für Grau)")
    parser.add_argument("--subdivision", type=int, default=0,
                       help="Anzahl der Subdivision-Stufen für glattere Oberflächen (default: 0, empfohlen: 1-2)")
    parser.add_argument("--remesh", action='store_true',
                       help="Verwende Remeshing für bessere Mesh-Topologie (experimentell)")
    parser.add_argument("--no-recursive", action='store_true', 
                       help="Nicht rekursiv durch Unterordner")
    
    args = parser.parse_args()
    
    logger.info("=== OBJ Mesh Smoothing Tool ===")
    logger.info(f"Eingabe: {args.input}")
    logger.info(f"Ausgabe: {args.output}")
    logger.info(f"Methode: {args.method}")
    logger.info(f"Iterationen: {args.iterations}")
    logger.info(f"Lambda: {args.lambda_param}")
    logger.info(f"Farbe: RGB({args.color[0]:.2f}, {args.color[1]:.2f}, {args.color[2]:.2f})")
    logger.info(f"Subdivision-Stufen: {args.subdivision}")
    logger.info(f"Remeshing: {args.remesh}")
    logger.info(f"Rekursiv: {not args.no_recursive}")
    
    process_directory(
        args.input, 
        args.output, 
        args.method, 
        args.iterations, 
        args.lambda_param, 
        not args.no_recursive,
        tuple(args.color),
        args.subdivision,
        args.remesh
    )

if __name__ == "__main__":
    # Beispiel für direkte Verwendung ohne Kommandozeilenargumente
    if len(sys.argv) == 1:
        # Standard-Pfade für Ihr Workspace
        current_dir = Path(__file__).parent
        input_directory = current_dir
        output_directory = current_dir / "smoothed_obj_files"
        
        logger.info("Kein Kommandozeilenargument angegeben. Verwende Standard-Pfade:")
        logger.info(f"Eingabe: {input_directory}")
        logger.info(f"Ausgabe: {output_directory}")
        
        process_directory(
            input_directory, 
            output_directory, 
            smooth_method='taubin',  # Taubin ist oft besser für 3D-Modelle
            iterations=25,  # Mehr Iterationen für bessere Glättung
            lambda_param=0.9,  # Höherer Lambda-Wert für stärkere Glättung
            recursive=True,
            color=(0.7, 0.7, 0.7),  # Hellgrau
            subdivision_levels=2,  # Zwei Subdivision-Stufen für viel glattere Oberflächen
            remesh=False  # Kein Remeshing standardmäßig
        )
    else:
        main()
