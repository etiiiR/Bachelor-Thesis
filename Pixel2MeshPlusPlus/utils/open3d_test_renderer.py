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
        
        # Setze eine neutrale graue Farbe für das Mesh
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Hellgrau
        
        # Erstelle Wireframe aus dem Mesh
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireframe.paint_uniform_color([0.0, 0.3, 1.0])  # Blaue Linien
        
        # Berechne Normalen für bessere Beleuchtung
        mesh.compute_vertex_normals()
        
        return mesh, wireframe
        
    except Exception as e:
        print(f"Fehler beim Laden von {mesh_path}: {str(e)}")
        return None, None

def render_view_simple(mesh, wireframe, view_name, output_path, view_params):
    """
    Einfachere Render-Funktion für Tests.
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
        render_option.line_width = 3.0  # Dickere Wireframe-Linien
        
        # Einfache Kamera-Einstellung
        ctr = vis.get_view_control()
        
        # Berechne Mesh-Zentrum und Größe
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
        ctr.set_zoom(0.7)  # Näher zoom
        
        # Rendere das Bild
        vis.poll_events()
        vis.update_renderer()
        
        # Speichere als PNG
        vis.capture_screen_image(output_path, do_render=True)
        vis.destroy_window()
        
        print(f"  {view_name}-Ansicht gespeichert: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Fehler beim Rendern der {view_name}-Ansicht: {str(e)}")
        return False

def test_single_mesh():
    """
    Test-Funktion für ein einzelnes Mesh.
    """
    # Test mit einer Datei
    test_file = r"C:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\pixel2meshplusplus\output\1\1\pollen_17767_Common_knapweed_Centaurea_nigra_pollen_grain_00_predict_1.obj"
    output_dir = r"C:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\open3d_test"
    
    # Erstelle Ausgabe-Ordner
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Teste mit: {os.path.basename(test_file)}")
    
    # Lade Mesh und erstelle Wireframe
    mesh, wireframe = create_mesh_with_wireframe(test_file)
    
    if mesh is None or wireframe is None:
        print("Fehler beim Laden des Test-Meshes!")
        return
    
    print(f"Mesh geladen: {len(mesh.vertices)} Vertices, {len(mesh.triangles)} Faces")
    print(f"Wireframe erstellt: {len(wireframe.lines)} Linien")
    
    # Teste die 4 Ansichten
    views = ["front", "top", "bottom", "side"]
    
    for view_name in views:
        output_path = os.path.join(output_dir, f"test_{view_name}.png")
        success = render_view_simple(
            copy.deepcopy(mesh), 
            copy.deepcopy(wireframe),
            view_name,
            output_path,
            {}
        )
        
        if success:
            print(f"✓ {view_name} erfolgreich")
        else:
            print(f"✗ {view_name} fehlgeschlagen")

if __name__ == "__main__":
    test_single_mesh()
