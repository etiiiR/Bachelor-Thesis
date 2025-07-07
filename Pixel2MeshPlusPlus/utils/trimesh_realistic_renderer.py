import trimesh
import numpy as np
import os
from PIL import Image
import pyrender

def render_mesh_with_trimesh(obj_file, output_file):
    """
    Rendert ein .obj Mesh mit trimesh/pyrender für realistische Beleuchtung.
    """
    try:
        # Lade das Mesh mit trimesh
        mesh = trimesh.load(obj_file)
        
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print(f"Fehler: {obj_file} ist kein gültiges Mesh")
            return
        
        print(f"Verarbeite {obj_file} mit trimesh/pyrender...")
        
        # Sanfte Mesh-Reparatur falls nötig (weniger aggressiv)
        if not mesh.is_watertight:
            print(f"Mesh ist nicht wasserdicht, führe sanfte Reparatur durch...")
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            # Keine aggressive fill_holes - kann zu Artefakten führen
        
        # Konvertiere zu pyrender Mesh
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            # Verwende ursprüngliche Vertex-Farben
            vertex_colors = mesh.visual.vertex_colors
            print(f"Verwende ursprüngliche Vertex-Farben für {obj_file}")
        elif hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
            # Konvertiere Face-Farben zu Vertex-Farben
            vertex_colors = mesh.visual.face_colors
            print(f"Verwende ursprüngliche Face-Farben für {obj_file}")
        else:
            # Fallback: Standard-Farbe
            vertex_colors = np.full((len(mesh.vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
            print(f"Keine ursprünglichen Farben gefunden für {obj_file}, verwende Standard-Grau")
        
        # Stelle sicher, dass Vertex-Farben die richtige Form haben
        if len(vertex_colors) != len(mesh.vertices):
            # Falls Face-Farben vorliegen, interpoliere zu Vertex-Farben
            vertex_colors = np.full((len(mesh.vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
        
        # Erstelle pyrender Mesh mit solidem Material
        # Verwende die ursprünglichen Farben oder ein solides Grau
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            # Normalisiere Vertex-Farben zu [0,1] Bereich
            base_color = np.mean(vertex_colors[:, :3], axis=0) / 255.0
            # Dämpfe die Farbe leicht für solideres Aussehen
            base_color = base_color * 0.8 + 0.2  # Macht Farben weniger gesättigt
        else:
            base_color = [0.8, 0.8, 0.8]  # Helleres, solideres Grau
        
        # Material für solides, undurchsichtiges Rendering
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[base_color[0], base_color[1], base_color[2], 1.0],  # Vollständig undurchsichtig
            metallicFactor=0.0,  # Kein Metall-Effekt
            roughnessFactor=1.0,  # Komplett matt für maximale Diffusion
            alphaMode='OPAQUE',  # Explizit undurchsichtig
            doubleSided=True  # Beide Seiten der Faces rendern
        )
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # Erstelle Szene mit natürlicher Beleuchtung
        scene = pyrender.Scene(ambient_light=[0.6, 0.6, 0.6])  # Mehr Ambient Light für weichere Schatten
        scene.add(mesh_pyrender)
        
        # Einfache, natürliche Beleuchtung (nur ein Hauptlicht)
        # Hauptlicht von oben-vorne für natürliche Schatten
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        light1_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.7071, -0.7071, 1.0],  # Von oben-vorne
            [0.0, 0.7071, 0.7071, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(light1, pose=light1_pose)
        
        # Erstelle Kamera (Front-Ansicht)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        
        # Berechne optimale Kamera-Position
        bounds = mesh.bounds
        centroid = (bounds[0] + bounds[1]) / 2.0  # Berechne Zentrum manuell
        scale = np.linalg.norm(bounds[1] - bounds[0])
        
        # Kamera-Position für Front-Ansicht - näher zum Objekt
        camera_pose = np.array([
            [1.0, 0.0, 0.0, centroid[0]],
            [0.0, 1.0, 0.0, centroid[1]],
            [0.0, 0.0, 1.0, centroid[2] + scale * 1.2],  # Viel näher: 1.2 statt 2.5
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)
        
        # Renderer erstellen und rendern
        renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
        
        # Rendere nur realistische Version (ohne Wireframes)
        try:
            color, depth = renderer.render(scene)
            output_path = output_file.replace('.png', '_front_realistic.png')
            
            # Speichere als PNG
            if color is not None:
                img = Image.fromarray(color, 'RGB')
                img.save(output_path)
                print(f"Gespeichert: {output_path}")
            else:
                print(f"Fehler: Rendering ergab leeres Bild für {obj_file}")
        except Exception as render_error:
            print(f"Rendering-Fehler für {obj_file}: {render_error}")
            # Fallback: Einfaches Material versuchen
            try:
                simple_material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                    alphaMode='OPAQUE',
                    doubleSided=True
                )
                mesh_simple = pyrender.Mesh.from_trimesh(mesh, material=simple_material)
                scene_simple = pyrender.Scene(ambient_light=[0.6, 0.6, 0.6])
                scene_simple.add(mesh_simple)
                scene_simple.add(camera, pose=camera_pose)
                
                # Auch hier einfache Beleuchtung
                simple_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
                simple_light_pose = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.7071, -0.7071, 1.0],
                    [0.0, 0.7071, 0.7071, 1.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                scene_simple.add(simple_light, pose=simple_light_pose)
                
                color, depth = renderer.render(scene_simple)
                output_path = output_file.replace('.png', '_front_realistic.png')
                
                if color is not None:
                    img = Image.fromarray(color, 'RGB')
                    img.save(output_path)
                    print(f"Gespeichert (Fallback): {output_path}")
                else:
                    print(f"Auch Fallback-Rendering fehlgeschlagen für {obj_file}")
            except Exception as fallback_error:
                print(f"Fallback-Rendering fehlgeschlagen für {obj_file}: {fallback_error}")
        
        renderer.delete()
        
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {obj_file}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Hauptfunktion zum Rendern aller .obj Dateien"""
    
    # Erstelle neuen Ordner für die Renderings
    output_folder = "realistic_renderings"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Ordner '{output_folder}' erstellt.")
    
    # Finde alle .obj Dateien im aktuellen Verzeichnis
    obj_files = [f for f in os.listdir('.') if f.endswith('.obj')]
    
    if not obj_files:
        print("Keine .obj Dateien gefunden!")
        return
    
    print(f"Gefundene .obj Dateien: {obj_files}")
    
    for obj_file in obj_files:
        # Speichere in den neuen Ordner
        output_file = os.path.join(output_folder, obj_file.replace('.obj', '_trimesh_render.png'))
        print(f"\nRendere {obj_file} mit trimesh/pyrender...")
        render_mesh_with_trimesh(obj_file, output_file)
    
    print(f"\nFertig! {len(obj_files)} Meshes wurden in '{output_folder}' gerendert.")

if __name__ == "__main__":
    main()
