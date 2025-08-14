import vtk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from vtk.util import numpy_support

def save_vtk_rgb_image(render_window, path='debug_rgb.png'):
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()
    print(f"🖼️ Saved VTK render to: {path}")


def render_depth_and_normal(mesh_path, output_path, image_size=(224, 224)):
    # Load mesh
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Normalize: center and scale
    bounds = polydata.GetBounds()
    center = [(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, (bounds[5]+bounds[4])/2]
    scale = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) / 2.0

    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.Scale(1.0/scale, 1.0/scale, 1.0/scale)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # Recompute normals
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputConnection(transform_filter.GetOutputPort())
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.Update()

    # Mapper & Actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Renderer setup
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(*image_size)

    # Camera setup
    camera = vtk.vtkCamera()
    camera.SetPosition(0, 0, 3)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 1, 0)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()

    render_window.Render()

    # Save RGB render for debug
    save_vtk_rgb_image(render_window, output_path.replace('.dat', '_vtk_rgb.png'))

    # Read depth buffer
    z_buffer = vtk.vtkFloatArray()
    render_window.GetZbufferData(0, 0, image_size[0]-1, image_size[1]-1, z_buffer)
    depth = np.frombuffer(z_buffer, dtype=np.float32).reshape(image_size[1], image_size[0])

    near, far = 0.1, 10.0
    depth = 2.0 * near * far / (far + near - (2.0 * depth - 1.0) * (far - near))

    # Get normals from screen render
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()

    image = window_to_image.GetOutput()
    dims = image.GetDimensions()
    normal_array = image.GetPointData().GetScalars()
    normal_np = numpy_support.vtk_to_numpy(normal_array).reshape((dims[1], dims[0], 3)).astype(np.float32)
    normal_np = (normal_np / 255.0) * 2.0 - 1.0  # map [0, 255] → [-1, 1]

    # Camera intrinsics
    fx = fy = image_size[0] / (2 * np.tan(np.radians(30)))  # 60° FOV
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Camera extrinsics
    R = np.eye(3, dtype=np.float32)
    t = np.array([[0.0], [0.0], [3.0]], dtype=np.float32)
    camera_dict = {'extrinsic': np.hstack((R, t)), 'intrinsic': intrinsic}

    # Save .dat
    data = {
        'depth': depth.astype(np.float32),
        'normal': normal_np,
        'camera': camera_dict
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Saved .dat file to: {output_path}")


def inspect_dat_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("📦 Keys in .dat file:", list(data.keys()))
    if 'depth' in data:
        print("🟦 Depth shape:", data['depth'].shape, "| dtype:", data['depth'].dtype)
        print("   Min/Max depth:", np.min(data['depth']), "/", np.max(data['depth']))
    if 'normal' in data:
        print("🟨 Normal shape:", data['normal'].shape, "| dtype:", data['normal'].dtype)
        print("   Example normal vector:", data['normal'][0, 0])
    if 'camera' in data:
        cam = data['camera']
        print("📷 Intrinsic:\n", cam['intrinsic'])
        print("📷 Extrinsic:\n", cam['extrinsic'])


def visualize_dat(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    depth = data['depth']
    normal = data['normal']

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(depth, cmap='viridis')
    axs[0].set_title('Depth Map')
    axs[0].axis('off')

    norm_vis = (normal + 1.0) / 2.0
    axs[1].imshow(norm_vis)
    axs[1].set_title('Normal Map')
    axs[1].axis('off')

    plt.tight_layout()
    out_path = file_path.replace('.dat', '_visualization.png')
    plt.savefig(out_path)
    print(f"🖼️ Saved visualization to: {out_path}")


# === Run this on your STL ===
mesh_file = r"interim/17767_Common_knapweed_Centaurea_nigra_pollen_grain.stl"
output_file = r"pixel2mesh/17767.dat"

render_depth_and_normal(mesh_file, output_file)
inspect_dat_file(output_file)
visualize_dat(output_file)