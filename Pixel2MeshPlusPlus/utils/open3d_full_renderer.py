import open3d as o3d
import numpy as np
import os
from pathlib import Path
import copy

def create_mesh_with_wireframe(mesh_path):
    """
    Lädt ein Mesh und erstellt eine Kombination aus solidem Mesh und blauen Wireframe-Linien.
    """
    try:
        # Lade das Mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        if len(mesh.vertices) == 0:
            print(f"Warnung: {mesh_path} enthält keine Vertices")
            return None, None
        
        # Setze die Mesh-Farbe auf #1b669b (RGB: 27, 102, 155)
        mesh.paint_uniform_color([27/255, 102/255, 155/255])  # #1b669b

        # Erstelle Wireframe aus dem Mesh
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireframe.paint_uniform_color([1.0, 1.0, 1.0])  # Weiß
        
        # Berechne Normalen für bessere Beleuchtung
        mesh.compute_vertex_normals()
        
        return mesh, wireframe
        
    except Exception as e:
        print(f"Fehler beim Laden von {mesh_path}: {str(e)}")
        return None, None

def render_view(mesh, wireframe, view_name, output_path):
    """
    Rendert eine spezifische Ansicht des Meshes mit Wireframe.
    """
    try:
        # Erstelle Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=1024, visible=False)
        
        # Füge Geometrien hinzu
        vis.add_geometry(mesh)
        vis.add_geometry(wireframe)
        
        # Setup Render-Optionen
        render_option = vis.get_render_option()
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # Weißer Hintergrund
        render_option.mesh_show_back_face = True
        render_option.line_width = 10.0  # Noch dickere Wireframe-Linien
        
        # Einfache Kamera-Einstellung
        ctr = vis.get_view_control()
        
        # Berechne Mesh-Zentrum
        bbox = mesh.get_axis_aligned_bounding_box()
        mesh_center = bbox.get_center()
        
        # Setze Kamera basierend auf der Ansicht
        if view_name == "front":
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
        elif view_name == "top":
            ctr.set_front([0, -1, 0])
            ctr.set_up([0, 0, 1])
        elif view_name == "bottom":
            ctr.set_front([0, 1, 0])
            ctr.set_up([0, 0, -1])
        elif view_name == "side":
            ctr.set_front([-1, 0, 0])
            ctr.set_up([0, 1, 0])
        
        ctr.set_lookat(mesh_center)
        ctr.set_zoom(0.7)  # Näher zoom für bessere Sicht
        
        # Rendere das Bild
        vis.poll_events()
        vis.update_renderer()
        
        # Speichere als PNG
        vis.capture_screen_image(output_path, do_render=True)
        vis.destroy_window()
        
        print(f"  {view_name}-Ansicht gespeichert: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"  Fehler beim Rendern der {view_name}-Ansicht: {str(e)}")
        return False

def render_mesh_multi_views(obj_path, output_base_dir):
    """
    Rendert ein Mesh in 4 verschiedenen Ansichten.
    """
    # Lade Mesh und erstelle Wireframe
    mesh, wireframe = create_mesh_with_wireframe(obj_path)
    
    if mesh is None or wireframe is None:
        return False
    
    # Erstelle Ausgabe-Ordner für dieses Mesh
    mesh_name = Path(obj_path).stem
    mesh_output_dir = os.path.join(output_base_dir, mesh_name + "_views")
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    # Definiere die 4 Ansichten
    views = ["front", "top", "bottom", "side"]
    
    print(f"Rendere {mesh_name}...")
    success_count = 0
    
    # Rendere jede Ansicht
    for view_name in views:
        output_path = os.path.join(mesh_output_dir, f"{mesh_name}_{view_name}.png")
        
        success = render_view(
            copy.deepcopy(mesh), 
            copy.deepcopy(wireframe),
            view_name,
            output_path
        )
        
        if success:
            success_count += 1
    
    print(f"  {success_count}/4 Ansichten erfolgreich für {mesh_name}")
    return success_count > 0

def find_obj_files_recursive(root_dir):
    """
    Findet rekursiv alle .obj-Dateien in der Ordnerstruktur.
    """
    obj_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    
    return obj_files

def create_output_structure(obj_path, base_input_dir, base_output_dir):
    """
    Erstellt die entsprechende Ausgabe-Ordnerstruktur basierend auf der Eingabe-Struktur.
    """
    # Berechne relativen Pfad zur Basis
    rel_path = os.path.relpath(os.path.dirname(obj_path), base_input_dir)
    
    # Erstelle entsprechenden Ausgabe-Pfad
    output_dir = os.path.join(base_output_dir, rel_path)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def main():
    """
    Hauptfunktion zum rekursiven Rendern aller .obj-Dateien.
    """
    # Eingabe- und Ausgabe-Verzeichnisse
    input_dir = r"C:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\pixel2meshplusplus\output"
    output_dir = r"C:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\open3d_renderings"
    
    # Erstelle Basis-Ausgabe-Ordner
    os.makedirs(output_dir, exist_ok=True)
    
    # Finde alle .obj-Dateien rekursiv
    print("Suche nach .obj-Dateien...")
    obj_files = find_obj_files_recursive(input_dir)
    
    if not obj_files:
        print("Keine .obj-Dateien gefunden!")
        return
    
    print(f"Gefunden: {len(obj_files)} .obj-Dateien")
    print("=" * 60)
    
    # Verarbeite jede .obj-Datei
    successful_renders = 0
    
    for i, obj_path in enumerate(obj_files, 1):
        rel_path = os.path.relpath(obj_path, input_dir)
        print(f"[{i}/{len(obj_files)}] {rel_path}")
        
        # Erstelle entsprechende Ausgabe-Struktur
        mesh_output_dir = create_output_structure(obj_path, input_dir, output_dir)
        
        # Rendere das Mesh in 4 Ansichten
        success = render_mesh_multi_views(obj_path, mesh_output_dir)
        
        if success:
            successful_renders += 1
            
        # Progress-Update alle 50 Dateien
        if i % 50 == 0:
            print(f"\n>>> Fortschritt: {i}/{len(obj_files)} ({i/len(obj_files)*100:.1f}%) <<<")
            print(f">>> Erfolgreich: {successful_renders}/{i} <<<\n")
    
    print("=" * 60)
    print(f"🎉 FERTIG! 🎉")
    print(f"Erfolgreich gerendert: {successful_renders}/{len(obj_files)} Meshes")
    print(f"Ausgabe-Ordner: {output_dir}")
    print(f"Jedes Mesh hat 4 Ansichten: Front, Top, Bottom, Side")
    print(f"Mit blauen Wireframe-Linien und transparentem Hintergrund")

if __name__ == "__main__":
    main()
