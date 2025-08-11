import vtk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import math
from vtk.util import numpy_support

def get_camera_positions(num_views=8, distance=1.75):
    """
    Generates a list of camera positions on a sphere looking at the origin.
    This uses a standard set of 8 views common in many 3D deep learning tasks.
    """
    positions = []
    # A standard set of 8 azimuth and elevation angles
    angles = [
        (45, 30), (-45, 30), (135, 30), (-135, 30),
        (45, -30), (-45, -30), (135, -30), (-135, -30)
    ]

    for i in range(num_views):
        azimuth, elevation = angles[i]
        
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)

        x = distance * math.cos(el_rad) * math.sin(az_rad)
        y = distance * math.sin(el_rad)
        z = distance * math.cos(el_rad) * math.cos(az_rad)
        
        positions.append(((x, y, z), (azimuth, elevation)))
        
    return positions

def render_multiview_data(mesh_path, output_dir, image_size=(224, 224), num_views=8):
    """
    Loads, normalizes, and renders a mesh from multiple views, saving the output
    in a format compatible with the Pixel2Mesh++ DataFetcher.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the mesh (from your working script)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    polydata = reader.GetOutput()

    # 2. Normalize the mesh to a unit box (from your working script)
    bounds = polydata.GetBounds()
    center = np.array([(bounds[0] + bounds[1]) / 2.0, 
                       (bounds[2] + bounds[3]) / 2.0, 
                       (bounds[4] + bounds[5]) / 2.0])
    
    # Calculate scale to fit into a [-0.5, 0.5] box
    scale = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    
    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.Scale(1.0 / scale, 1.0 / scale, 1.0 / scale)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    normalized_polydata = transform_filter.GetOutput()

    # 3. Set up the rendering pipeline (from your working script)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(normalized_polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1) # White background

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(*image_size)

    # Image capture filter
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.ReadFrontBufferOff()

    # --- Multi-view Rendering Loop ---
    camera_positions = get_camera_positions(num_views)
    all_camera_meta = []
    
    rendering_dir = os.path.join(output_dir, 'rendering')
    os.makedirs(rendering_dir, exist_ok=True)

    for i, (position, angles) in enumerate(camera_positions):
        camera = renderer.GetActiveCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)

        # *** THIS IS THE CRUCIAL FIX ***
        # Reset the camera to frame the normalized object correctly for each view.
        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()

        render_window.Render()
        
        # Save the rendered image
        window_to_image_filter.Modified()
        window_to_image_filter.Update()
        vtk_image = window_to_image_filter.GetOutput()
        
        dims = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        numpy_image = numpy_support.vtk_to_numpy(vtk_array)
        numpy_image = numpy_image.reshape((dims[1], dims[0], 3))
        
        image_path = os.path.join(rendering_dir, f'{i:02d}.png')
        plt.imsave(image_path, np.flipud(numpy_image)) # Flip because VTK origin is bottom-left
        
        # Store camera metadata for `rendering_metadata.txt`
        # Format: [azimuth, elevation, unused_index, distance, fov]
        camera_meta = [angles[0], angles[1], 0, np.linalg.norm(position), camera.GetViewAngle()]
        all_camera_meta.append(camera_meta)
    
    # Save camera metadata file
    metadata_path = os.path.join(rendering_dir, 'rendering_metadata.txt')
    np.savetxt(metadata_path, np.array(all_camera_meta), fmt='%f')

    # 4. Save the ground truth mesh data (vertices and faces)
    vertices = numpy_support.vtk_to_numpy(normalized_polydata.GetPoints().GetData())
    faces_vtk = numpy_support.vtk_to_numpy(normalized_polydata.GetPolys().GetData())
    faces = faces_vtk.reshape(-1, 4)[:, 1:] # Reshape and remove the '3' count per face

    dat_content = (vertices.astype(np.float32), faces.astype(np.int32))
    
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    output_dat_path = os.path.join(output_dir, f"{output_dir.split(os.sep)[-2]}_{base_name}_00.dat")
    
    with open(output_dat_path, 'wb') as f:
        pickle.dump(dat_content, f)

    print(f"✅ Correctly processed {mesh_path} and saved to {output_dir}")

# === Main Execution Block ===
if __name__ == "__main__":
    input_folder = r"interim"
    # The root folder where categories will be created
    output_root_folder = r"pixel2mesh_data"
    os.makedirs(output_root_folder, exist_ok=True)
    
    # We'll use a default category name, e.g., 'stl_files'
    category_name = "stl_files"
    category_folder = os.path.join(output_root_folder, category_name)
    os.makedirs(category_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".stl"):
            mesh_path = os.path.join(input_folder, filename)
            model_id = os.path.splitext(filename)[0]
            
            # Each model gets its own subdirectory inside the category folder
            output_model_dir = os.path.join(category_folder, model_id)

            try:
                print(f"🔄 Processing {filename}...")
                render_multiview_data(mesh_path, output_model_dir, num_views=8)
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")