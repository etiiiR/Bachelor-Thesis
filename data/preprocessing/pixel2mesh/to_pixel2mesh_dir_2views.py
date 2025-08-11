# generate_renderings.py
# ───────────────────────────────────────────────────────────────────────────────
# For each .stl in "interim/<category>/<item_id>.stl", this script:
#   1) Creates "pixel2mesh/<category>/<item_id>/rendering/"
#   2) Writes rendering_metadata.txt with exactly 8 lines (tx ty tz fx fy)
#   3) Renders three RGBA‐PNGs: 00.png, 06.png, 07.png (224×224),
#      using your “working” normalization code so that the mesh is centered + scaled.
#
# The ORIGINAL DataFetcher (which does `for view in [0,6,7]`) will load these PNGs
# and metadata correctly—no need to modify DataFetcher.py at all.
# ───────────────────────────────────────────────────────────────────────────────

import os
import math
import vtk
import numpy as np
import cv2
from vtk.util import numpy_support

# ------------------------------------------------------------------------------
# 1) CAMERA_POSES:  8 world‐space camera centers (so rendering_metadata.txt has rows 0..7).
#    Rows 0, 6, 7 are “real” (we actually render them). Rows 1–5 are dummy, but must exist.
#    Each metadata row = "tx ty tz fx fy"
# ------------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)    # (width, height)
FOV_DEG    = 60.0          # horizontal FOV in degrees

W, H = IMAGE_SIZE
fx = fy = (W / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))

CAMERA_POSES = [
    ( 0.0,    0.0,   2.0),   # index 0  → front (+Z)
    ( 1.414,  1.414, 2.0),   # index 1 (dummy)
    (-1.414,  1.414, 2.0),   # index 2 (dummy)
    ( 1.414, -1.414, 2.0),   # index 3 (dummy)
    (-1.414, -1.414, 2.0),   # index 4 (dummy)
    ( 0.0,    0.0,  -2.0),   # index 5 (dummy)
    ( 2.0,    0.0,   0.0),   # index 6  → side (+X)
    ( 0.0,    2.0,   0.0)    # index 7  → top  (+Y)
]
assert len(CAMERA_POSES) == 8

# ------------------------------------------------------------------------------
# 2) render_png_view(…):
#    - Exactly the same normalization code you provided, plus a simple gray
#      material and a single light so that the mesh is visible on white.
#    - Outputs a 224×224 RGBA PNG (alpha=255 everywhere).
# ------------------------------------------------------------------------------
def render_png_view(mesh_path, out_png_path, cam_pos, image_size):
    """
    mesh_path:    path to .stl
    out_png_path: where to write a 224×224 RGBA PNG (mesh centered+normalized)
    cam_pos:      (x,y,z) camera position in world space
    image_size:   (W, H)
    """
    W, H = image_size

    # 2.1) Load & normalize mesh (unit sphere)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    polydata = reader.GetOutput()

    bounds = polydata.GetBounds()
    center = [ (bounds[1]+bounds[0]) / 2.0,
               (bounds[3]+bounds[2]) / 2.0,
               (bounds[5]+bounds[4]) / 2.0 ]
    scale = max(bounds[1]-bounds[0],
                bounds[3]-bounds[2],
                bounds[5]-bounds[4]) / 2.0

    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.Scale(1.0/scale, 1.0/scale, 1.0/scale)

    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetInputData(polydata)
    tf_filter.SetTransform(transform)
    tf_filter.Update()

    # 2.2) Recompute normals on normalized mesh
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputConnection(tf_filter.GetOutputPort())
    normals_filter.ComputePointNormalsOn()
    normals_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Give it a simple mid‐gray material so it’s visible on white
    prop = actor.GetProperty()
    prop.SetColor(0.7, 0.7, 0.7)
    prop.SetDiffuse(1.0)
    prop.SetSpecular(0.0)
    prop.SetAmbient(0.2)

    # 2.3) Offscreen RGBA renderer (white background)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.SetBackgroundAlpha(1.0)
    renderer.AddActor(actor)

    # Add a single directional light at the camera, pointing at origin
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(*cam_pos)
    light.SetFocalPoint(0.0, 0.0, 0.0)
    light.SetColor(1.0, 1.0, 1.0)
    light.SetIntensity(1.0)
    renderer.AddLight(light)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetAlphaBitPlanes(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(W, H)

    # 2.4) Camera setup
    camera = vtk.vtkCamera()
    camera.SetPosition(*cam_pos)
    camera.SetFocalPoint(0.0, 0.0, 0.0)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetClippingRange(0.1, 10.0)
    renderer.SetActiveCamera(camera)
    renderer.ResetCameraClippingRange()

    render_window.Render()

    # 2.5) Capture RGBA buffer
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetInputBufferTypeToRGBA()
    w2if.ReadFrontBufferOff()
    w2if.Update()

    vtk_img = w2if.GetOutput()
    dims = vtk_img.GetDimensions()  # (W, H, 1)
    arr = numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    rgba = arr.reshape((dims[1], dims[0], 4)).astype(np.uint8)  # (H, W, 4)

    # Flip vertically (VTK’s origin is bottom‐left; OpenCV’s is top‐left)
    rgba = np.flipud(rgba)

    # Force alpha=255 everywhere (so no transparency at all)
    rgba[:, :, 3] = 255

    # 2.6) Write RGBA PNG via OpenCV
    cv2.imwrite(out_png_path, rgba)
    print(f"  ✓ Saved PNG: {out_png_path}")


# ------------------------------------------------------------------------------
# 3) MAIN LOOP: for each STL, write metadata + render PNGs 00, 06, 07
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    input_root  = "interim"     # e.g. "interim/Chair/02958343.stl"
    output_root = "pixel2mesh"  # → "pixel2mesh/Chair/02958343/rendering/"

    for category in os.listdir(input_root):
        cat_path = os.path.join(input_root, category)
        if not os.path.isdir(cat_path):
            continue

        for filename in os.listdir(cat_path):
            if not filename.lower().endswith(".stl"):
                continue

            item_id  = os.path.splitext(filename)[0]
            mesh_path = os.path.join(cat_path, filename)

            render_dir = os.path.join(output_root, category, item_id, "rendering")
            os.makedirs(render_dir, exist_ok=True)

            # 3.1) Write rendering_metadata.txt (8 rows of "tx ty tz fx fy")
            meta_path = os.path.join(render_dir, "rendering_metadata.txt")
            with open(meta_path, "w") as mf:
                for (tx, ty, tz) in CAMERA_POSES:
                    mf.write(f"{tx:.6f} {ty:.6f} {tz:.6f} {fx:.6f} {fy:.6f}\n")
            print(f"\n✓ Wrote metadata: {meta_path}  (8 lines)")

            # 3.2) Render exactly three PNGs: 00.png (view 0), 06.png (view 6), 07.png (view 7)
            for view_idx in [0, 6, 7]:
                cam = CAMERA_POSES[view_idx]
                png_name = f"{view_idx:02d}.png"
                out_png  = os.path.join(render_dir, png_name)
                print(f"⟳ Rendering view {view_idx:02d} for {category}/{item_id}")
                try:
                    render_png_view(mesh_path, out_png, cam, IMAGE_SIZE)
                except Exception as e:
                    print(f"  ⚠️  Failed view {view_idx:02d} for {category}/{item_id}: {e}")

    print("\n🎉 Done! Point your ORIGINAL DataFetcher at:")
    print(f"    {os.path.abspath(output_root)}")
    print("  – Each object now has <cat>/<item_id>/rendering/00.png, 06.png, 07.png")
    print("  – and a rendering_metadata.txt (8 rows of “tx ty tz fx fy”).")