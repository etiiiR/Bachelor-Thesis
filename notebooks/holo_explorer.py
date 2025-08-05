# %%
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import morphology, filters
import pandas as pd

# %% [markdown]
# ## Holographic Ripple Filter
# The following class applies a PIL transform that
# 1) blurs to suppress noise
# 2) thresholds (Otsu or adaptive)
# 3) morphological-opens to knock out thin fringes
# 4) applies mask to original gray image, setting background to white
# 5) recenters & scales the object to fill the frame
# 6) finds the object bbox
# 7) scales it by up to `max_scale` (but no more than frame size)
# 8) pastes centered on white canvas

# %%
class RemoveRipples(object):
    def __init__(
        self,
        method: str = "otsu",
        blur_ksize: int = 5,
        adaptive_blocksize: int = 51,
        adaptive_C: int = 2,
        morph_ksize: int = 5,
        max_scale: float = 1.7,
    ):
        self.method = method.lower()
        self.blur_ksize = blur_ksize
        self.adaptive_blocksize = adaptive_blocksize
        self.adaptive_C = adaptive_C
        self.morph_ksize = morph_ksize
        self.max_scale = max_scale

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize)
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        orig = np.array(img.convert("L"), dtype=np.uint8)
        h, w = orig.shape

        # blur & threshold as before
        blur = cv2.GaussianBlur(orig, (self.blur_ksize,) * 2, 0)
        if self.method == "otsu":
            _, mask = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
        else:
            mask = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.adaptive_blocksize,
                self.adaptive_C,
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        # apply mask
        result = np.where(mask == 255, orig, 255).astype(np.uint8)

        # find bbox of object
        ys, xs = np.where(result < 255)
        if len(xs) == 0 or len(ys) == 0:
            return Image.new("L", (w, h), color=255)

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        crop = result[y1 : y2 + 1, x1 : x2 + 1]

        crop_h, crop_w = crop.shape
        max_by_frame = min(w / crop_w, h / crop_h)
        scale = min(self.max_scale, max_by_frame)

        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        # resize & paste
        crop_pil = Image.fromarray(crop, mode="L")
        resized = crop_pil.resize((new_w, new_h), resample=Image.LANCZOS)
        canvas = Image.new("L", (w, h), color=255)
        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2
        canvas.paste(resized, (x_off, y_off))

        return canvas

# %%
ripple_filter = RemoveRipples(
    method="otsu",
    blur_ksize=5,
    adaptive_blocksize=51,
    adaptive_C=2,
    morph_ksize=5,
    max_scale=1.7,
)

image_path = "../data/subset_poleno/g_carpinus_s_betulus/poleno-27_2023-04-06_18.06.28.175254_ev.computed_data.holography.image_pairs.0.0.rec_mag.png"

img = Image.open(image_path)
arr = np.array(img).astype(np.float32)
min_val, max_val = arr.min(), arr.max()
if max_val > min_val:
    arr = (arr - min_val) / (max_val - min_val) * 255.0
else:
    arr = np.zeros_like(arr)
gray_img = Image.fromarray(arr.astype(np.uint8), mode="L")

filtered_img = ripple_filter(gray_img)

orig_arr = np.array(gray_img)
filtered_arr = np.array(filtered_img)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(orig_arr, cmap="gray")
axes[0].set_title("Normalized Original")
axes[0].axis("off")

axes[1].imshow(filtered_arr, cmap="gray")
axes[1].set_title("After RemoveRipples")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# %%
class HolographicPollenDataset(Dataset):
    def __init__(self, transform=None, extensions=None):
        self.root_dir = os.path.join(os.getenv("DATA_DIR_PATH"), "subset_poleno")
        self.transform = transform
        self.extensions = extensions or [".png"]

        self.classes = sorted(
            [
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )

        raw_samples = []
        for taxa in self.classes:
            cls_dir = os.path.join(self.root_dir, taxa)
            for fname in os.listdir(cls_dir):
                if any(fname.lower().endswith(ext) for ext in self.extensions):
                    path = os.path.join(cls_dir, fname)
                    raw_samples.append((path, taxa))

        groups = {}
        for path, taxa in raw_samples:
            fname = os.path.basename(path)
            if "image_pairs" not in fname:
                continue
            base = fname.split("image_pairs")[0]
            groups.setdefault((base, taxa), []).append(path)

        # Build pairs list
        self.pairs = []  # (path0, path1, taxa_name)
        for (base, taxa), paths in groups.items():
            p0 = next((p for p in paths if ".0." in os.path.basename(p)), None)
            p1 = next((p for p in paths if ".1." in os.path.basename(p)), None)

            if p0 and p1:
                self.pairs.append((p0, p1, taxa))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path0, path1, taxa = self.pairs[idx]

        def load_and_normalize(p):
            img = Image.open(p)
            arr = np.array(img).astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            else:
                arr = np.zeros_like(arr)
            return Image.fromarray(arr.astype(np.uint8), mode="L")

        img0 = load_and_normalize(path0)
        img1 = load_and_normalize(path1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # Return images, taxa, and the paths for reference
        return {
            "images": (img0, img1),
            "taxa": taxa,
            "paths": (path0, path1),
            "image_pair_idx": idx,
        }


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        RemoveRipples(
            method="otsu",
            blur_ksize=5,
            adaptive_blocksize=51,
            adaptive_C=2,
            morph_ksize=5,
        ),
        transforms.ToTensor(),
    ]
)

dataset = HolographicPollenDataset(transform=transform)

# %%
# Test the dataset with new structure
sample = dataset[0]
print("Sample structure:")
print(f"Images shape: {sample['images'][0].shape}, {sample['images'][1].shape}")
print(f"Taxa: {sample['taxa']}")
print(f"Paths: {sample['paths'][0]}")
print(f"       {sample['paths'][1]}")
print(f"Image pair index: {sample['image_pair_idx']}")

# Check if the two images in the pair are equal
torch.all(
    sample["images"][0] == sample["images"][1]
)  # Check if the two images in the pair are equal

# %%
x = 200  # Changed from 2000 to a smaller number to avoid index errors

sample = dataset[x]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the images
axes[0].imshow(sample["images"][0].numpy().transpose(1, 2, 0), cmap="gray")
axes[0].set_title("Image 0")
axes[0].axis("off")

axes[1].imshow(sample["images"][1].numpy().transpose(1, 2, 0), cmap="gray")
axes[1].set_title("Image 1")
axes[1].axis("off")

# Show taxa and paths in the title
fig.suptitle(f'Taxa: {sample["taxa"]} | Index: {x}', fontsize=16)

# Print paths for reference
print(f"Image 0 path: {sample['paths'][0]}")
print(f"Image 1 path: {sample['paths'][1]}")

plt.tight_layout()
plt.show()

# %%
# Generate 256x256 images using the dataset with paths
from pathlib import Path


def generate_256x256_images(dataset, output_dir="generated_holo_256", max_samples=10):
    """Generate 256x256 images from the holographic dataset."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create 256x256 transform
    transform_256 = transforms.Compose(
        [
            RemoveRipples(
                method="otsu",
                blur_ksize=5,
                adaptive_blocksize=51,
                adaptive_C=2,
                morph_ksize=5,
                max_scale=1.7,
            ),
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
        ]
    )

    print(
        f"Generating {min(max_samples, len(dataset))} samples at 256x256 resolution..."
    )

    for i in range(min(max_samples, len(dataset))):
        # Get original sample data
        sample = dataset[i]
        taxa = sample["taxa"]
        paths = sample["paths"]

        # Create taxa directory
        taxa_dir = output_path / taxa
        taxa_dir.mkdir(exist_ok=True)

        # Process each image in the pair
        for img_idx, img_path in enumerate(paths):
            # Load and process the original image (without dataset transform)
            img = Image.open(img_path)
            arr = np.array(img).astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            else:
                arr = np.zeros_like(arr)

            # Convert to PIL and apply 256x256 transform
            pil_img = Image.fromarray(arr.astype(np.uint8), mode="L")
            transformed_img = transform_256(pil_img)

            # Convert tensor back to PIL for saving
            if transformed_img.dim() == 3 and transformed_img.size(0) == 1:
                transformed_img = transformed_img.squeeze(0)

            img_array = (transformed_img.numpy() * 255).astype(np.uint8)
            final_img = Image.fromarray(img_array, mode="L")

            # Create filename from original path
            original_name = Path(img_path).stem
            output_filename = f"{original_name}_256x256.png"
            output_file = taxa_dir / output_filename

            # Save the image
            final_img.save(output_file)
            print(f"Saved: {output_file}")

    print(f"\nGeneration complete! Images saved to: {output_path}")


# Test with a few samples
generate_256x256_images(dataset, max_samples=5)

# %%
import sys

# move out of the current directory
sys.path.append("../")

from data import HolographicPolenoDataModule

# %%
datamodule = HolographicPolenoDataModule(
    batch_size=3, num_workers=0, image_transforms=transform
)

# %%
datamodule.setup("test")

for batch in datamodule.test_dataloader():
    print(batch[0][0].shape, batch[1])

# %%
results = pd.read_csv("../eval/Holo/mesh_eval_results_holo.csv")
display(results)

# %%
filtered = results

aggregated = filtered.groupby("model", as_index=False).agg(
    mean_chamfer=("chamfer", "mean"),
    std_chamfer=("chamfer", "std"),
    mean_fscore_1=("fscore_1", "mean"),
    std_fscore_1=("fscore_1", "std"),
    mean_fscore_2_5=("fscore_2_5", "mean"),
    std_fscore_2_5=("fscore_2_5", "std"),
    mean_fscore_5=("fscore_5", "mean"),
    std_fscore_5=("fscore_5", "std"),
    mean_iou=("voxel_iou", "mean"),
    std_iou=("voxel_iou", "std"),
)

aggregated

# %%
# Enhanced Holographic Pollen 3D Reconstruction Visualization System
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
import warnings
import os
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")

# Configuration
MAPPING_CSV = (
    "C:/Users/super/Documents/Github/sequoia/Eval/Holo/holo_file_mapping_corrected.csv"
)
MESH_BASE_DIR = (
    "C:/Users/super/Documents/Github/sequoia/TestEvaluationPipeline/data_holo"
)
IMAGE_BASE_DIR = os.path.join(os.getenv("DATA_DIR_PATH", ""), "subset_poleno")

# Load mapping and results data
mapping_df = pd.read_csv(MAPPING_CSV)
try:
    results_df = pd.read_csv("../eval/Holo/mesh_eval_results_holo.csv")
except:
    results_df = None


class HolographicReconstructionVisualizer:
    """Enhanced visualization system with metrics integration and improved mesh rendering."""

    def __init__(
        self, mapping_df, mesh_base_dir, image_base_dir, dataset=None, results_df=None
    ):
        self.mapping_df = mapping_df
        self.mesh_base_dir = mesh_base_dir
        self.image_base_dir = image_base_dir
        self.dataset = dataset
        self.results_df = results_df
        self.mesh_types = [
            "gt_file",
            "pix2vox_file",
            "pixel2mesh_file",
            "pixelnerf_file",
        ]
        self.mesh_labels = [
            "Ground Truth\n(Visual Hull)",
            "Pix2Vox",
            "Pixel2Mesh",
            "PixelNeRF",
        ]
        self.model_names = ["vh_2img_holo", "pix2vox", "pixel2mesh", "pixelnerf"]
        self.mesh_colors = ["lightblue", "lightcoral", "lightgreen", "plum"]
        self.overlay_colors = ["lightblue", "lightcoral", "lightgreen", "plum"]
        # Maximum transparency for better overlap visualization
        self.alphas = [0.15, 0.15, 0.15, 0.15]
        self.edge_colors = ["darkblue", "darkred", "darkgreen", "indigo"]

    def load_and_normalize_image(self, image_path):
        """Load and normalize an image for display."""
        if not os.path.exists(image_path):
            return None

        img = Image.open(image_path)
        arr = np.array(img).astype(np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        else:
            arr = np.zeros_like(arr)
        return arr.astype(np.uint8)

    def find_image_pair_from_dataset(self, timestamp):
        """Find image pair from dataset using timestamp."""
        if self.dataset is None:
            return None, None

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            paths = sample["paths"]

            if any(timestamp in path for path in paths):
                img0 = sample["images"][0]
                img1 = sample["images"][1]

                if img0.dim() == 3 and img0.size(0) == 1:
                    img0 = img0.squeeze(0)
                if img1.dim() == 3 and img1.size(0) == 1:
                    img1 = img1.squeeze(0)

                img0_arr = (img0.numpy() * 255).astype(np.uint8)
                img1_arr = (img1.numpy() * 255).astype(np.uint8)

                return img0_arr, img1_arr

        return None, None

    def find_image_path_for_timestamp(self, timestamp):
        """Find the corresponding image file for a given timestamp."""
        for taxa_dir in Path(self.image_base_dir).iterdir():
            if taxa_dir.is_dir():
                for img_file in taxa_dir.iterdir():
                    if (
                        img_file.is_file()
                        and timestamp in img_file.name
                        and "image_pairs.0.0.rec_mag.png" in img_file.name
                    ):
                        return str(img_file)
        return None

    def find_mesh_file(self, filename, mesh_type=None):
        """Find a mesh file with special handling for ground truth."""
        if not os.path.exists(self.mesh_base_dir):
            return None

        if mesh_type == "gt_file":
            gt_dir = os.path.join(self.mesh_base_dir, "vh_2img_holo_test")
            if os.path.exists(gt_dir):
                gt_path = os.path.join(gt_dir, filename)
                if os.path.exists(gt_path):
                    return gt_path

        for root, dirs, files in os.walk(self.mesh_base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def load_mesh_safe(self, mesh_path, verbose=False):
        """Safely load a mesh file."""
        try:
            if mesh_path and os.path.exists(mesh_path):
                return trimesh.load(mesh_path)
        except Exception:
            pass
        return None

    def get_metrics_for_timestamp(self, timestamp, model_name):
        """Get evaluation metrics for a specific timestamp and model."""
        if self.results_df is None:
            return None

        # Find matching row in results
        mask = (self.results_df["timestamp"] == timestamp) & (
            self.results_df["model"] == model_name
        )
        matching_rows = self.results_df[mask]

        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            return {
                "chamfer": row.get("chamfer", None),
                "fscore_1": row.get("fscore_1", None),
                "fscore_2_5": row.get("fscore_2_5", None),
                "voxel_iou": row.get("voxel_iou", None),
            }
        return None

    def plot_mesh_with_wireframe(self, ax, mesh, color, alpha, edge_color, label):
        """Plot mesh with both surface and wireframe for better visibility."""
        vertices = mesh.vertices
        faces = mesh.faces

        # Plot surface with transparency
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            alpha=alpha,
            color=color,
            shade=True,
        )

        # Add wireframe for structure visibility
        # Subsample faces for wireframe to avoid clutter
        step = max(1, len(faces) // 200)  # Show max 200 triangles as wireframe
        wireframe_faces = faces[::step]

        for face in wireframe_faces:
            triangle = vertices[face]
            # Create triangle edges
            edges = [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ]
            for edge in edges:
                ax.plot3D(*zip(*edge), color=edge_color, alpha=0.1, linewidth=0.1)

    def create_enhanced_legend(self, ax, colors, labels, alphas, metrics_list):
        """Create an enhanced legend with colors and metrics."""
        legend_elements = []

        for i, (color, label, alpha) in enumerate(zip(colors, labels, alphas)):
            # Base label
            clean_label = label.replace("\n", " ")

            # Add metrics if available
            if i < len(metrics_list) and metrics_list[i]:
                metrics = metrics_list[i]
                metrics_str = []
                if metrics.get("chamfer") is not None:
                    metrics_str.append(f"CD: {metrics['chamfer']:.3f}")
                if metrics.get("fscore_1") is not None:
                    metrics_str.append(f"F1: {metrics['fscore_1']:.3f}")
                if metrics.get("voxel_iou") is not None:
                    metrics_str.append(f"IoU: {metrics['voxel_iou']:.3f}")

                if metrics_str:
                    clean_label += f"\n{' | '.join(metrics_str)}"

            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor=color, alpha=alpha, label=clean_label)
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=8,
            framealpha=0.95,
        )

    def create_color_legend(self, ax, colors, labels, alphas):
        """Create a colored legend for the overlay plot."""
        legend_elements = []
        for color, label, alpha in zip(colors, labels, alphas):
            legend_elements.append(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=color,
                    alpha=alpha,
                    label=label.replace("\n", " "),
                )
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=9,
            framealpha=0.9,
        )

    def plot_single_comparison(self, timestamp, verbose=False):
        """Create a clean single-row comparison plot with NO overlay."""
        row = self.mapping_df[self.mapping_df["timestamp"] == timestamp]
        if row.empty:
            return None

        row = row.iloc[0]

        # Get image pair from dataset
        img0_arr, img1_arr = self.find_image_pair_from_dataset(timestamp)

        if img0_arr is None or img1_arr is None:
            # Fallback to original method
            image_path = self.find_image_path_for_timestamp(timestamp)
            if not image_path:
                return None

            image_arr = self.load_and_normalize_image(image_path)
            if image_arr is None:
                return None
            img0_arr = img1_arr = image_arr

        # Create figure with SINGLE ROW layout (1x6) - NO OVERLAY
        fig, axes = plt.subplots(1, 6, figsize=(30, 8))
        fig.patch.set_facecolor('none')  # Make figure background transparent
        fig.patch.set_alpha(0.0)

        # Plot input images with LARGE text
        axes[0].imshow(img0_arr, cmap="gray")
        axes[0].set_title("Input Image 0", fontsize=30, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(img1_arr, cmap="gray")
        axes[1].set_title("Input Image 1", fontsize=30, fontweight="bold")
        axes[1].axis("off")
        
        # Load and display 4 meshes CLEANLY in single row
        for i, (mesh_type, label, color, model_name) in enumerate(
            zip(self.mesh_types, self.mesh_labels, self.mesh_colors, self.model_names)
        ):
            mesh = None

            if pd.notna(row[mesh_type]):
                mesh_filename = row[mesh_type]
                mesh_path = self.find_mesh_file(mesh_filename, mesh_type)
                if mesh_path:
                    mesh = self.load_mesh_safe(mesh_path, verbose)

            # Create 3D subplot 
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(1, 6, i + 3, projection="3d")
            
            # Entferne Hintergrund und Achsen
            ax.set_facecolor('none')
            ax.grid(False)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Versuche, alle Achsen zu verstecken
            try:
                ax.w_xaxis.line.set_color((0,0,0,0))
                ax.w_yaxis.line.set_color((0,0,0,0))
                ax.w_zaxis.line.set_color((0,0,0,0))
            except Exception:
                pass
            # Keine z-Achse Methoden, da nicht immer verfügbar

            if mesh is not None:
                try:
                    vertices = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.faces)
                    if hasattr(ax, 'plot_trisurf'):
                        ax.plot_trisurf(
                            vertices[:, 0], vertices[:, 1], vertices[:, 2],
                            triangles=faces, alpha=1.0, color=color, shade=True,
                            linewidth=0, edgecolor='none'
                        )
                        max_range = (
                            np.array([
                                vertices[:, 0].max() - vertices[:, 0].min(),
                                vertices[:, 1].max() - vertices[:, 1].min(),
                                vertices[:, 2].max() - vertices[:, 2].min(),
                            ]).max() / 2.0
                        )
                        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
                        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
                        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
                        if hasattr(ax, 'set_xlim'):
                            ax.set_xlim(mid_x - max_range, mid_x + max_range)
                        if hasattr(ax, 'set_ylim'):
                            ax.set_ylim(mid_y - max_range, mid_y + max_range)
                        if hasattr(ax, 'set_zlim'):
                            ax.set_zlim(mid_z - max_range, mid_z + max_range)
                    clean_label = label.replace("\n", " ")
                    ax.set_title(clean_label, fontsize=30, fontweight="bold", color="black")
                except Exception as e:
                    ax.text(0.5, 0.5, str(e), horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontsize=20, color="red")
                    clean_label = label.replace("\n", " ")
                    ax.set_title(f"{clean_label} - Error", fontsize=30, color="red")
            else:
                ax.text(0.5, 0.5, "Not Available", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontsize=30, color="red")
                clean_label = label.replace("\n", " ")
                ax.set_title(f"{clean_label} - Missing", fontsize=30, color="red")

        # Extract taxa information
        taxa = "Unknown"
        if self.dataset:
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                if any(timestamp in path for path in sample["paths"]):
                    taxa = sample["taxa"]
                    break
        else:
            image_path = self.find_image_path_for_timestamp(timestamp)
            if image_path:
                taxa = Path(image_path).parent.name

        plt.suptitle(f"{taxa} - {timestamp}", fontsize=30, fontweight="bold")
        plt.tight_layout()
        plt.show()

        return fig

    def process_all_reconstructions(self):
        """Process all reconstructions and create visualizations."""
        successful_plots = 0
        failed_plots = 0

        for i, (_, row) in enumerate(self.mapping_df.iterrows()):
            timestamp = row["timestamp"]

            try:
                fig = self.plot_single_comparison(timestamp, verbose=False)
                if fig is not None:
                    successful_plots += 1
                else:
                    failed_plots += 1
            except Exception:
                failed_plots += 1

        return {
            "successful": successful_plots,
            "failed": failed_plots,
            "total": len(self.mapping_df),
        }

    def create_overview_grid(self):
        """Create a summary grid of all reconstructions."""
        samples_to_show = len(self.mapping_df)
        rows = int(np.ceil(np.sqrt(samples_to_show)))
        cols = int(np.ceil(samples_to_show / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(samples_to_show):
            row_idx = i // cols
            col_idx = i % cols
            ax = axes[row_idx, col_idx]

            timestamp = self.mapping_df.iloc[i]["timestamp"]

            # Try to get image from dataset first
            img0_arr, img1_arr = self.find_image_pair_from_dataset(timestamp)

            if img0_arr is not None:
                ax.imshow(img0_arr, cmap="gray")

                taxa = "Unknown"
                if self.dataset:
                    for j in range(len(self.dataset)):
                        sample = self.dataset[j]
                        if any(timestamp in path for path in sample["paths"]):
                            taxa = sample["taxa"].replace("g_", "").replace("_s_", " ")
                            break

                ax.set_title(f"{taxa}\n{timestamp[:10]}", fontsize=6)
            else:
                # Fallback method
                image_path = self.find_image_path_for_timestamp(timestamp)
                if image_path:
                    image_arr = self.load_and_normalize_image(image_path)
                    if image_arr is not None:
                        ax.imshow(image_arr, cmap="gray")
                        taxa = (
                            Path(image_path)
                            .parent.name.replace("g_", "")
                            .replace("_s_", " ")
                        )
                        ax.set_title(f"{taxa}\n{timestamp[:10]}", fontsize=6)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "Error",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"Error\n{timestamp[:10]}", fontsize=6)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Not Found",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Missing\n{timestamp[:10]}", fontsize=6)

            ax.axis("off")

        # Hide unused subplots
        for i in range(samples_to_show, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            axes[row_idx, col_idx].axis("off")

        plt.suptitle(
            f"Overview: {samples_to_show} Holographic Pollen Reconstructions",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def render_mesh_to_image(self, mesh, color, size=512):
        """
        Render a 3D mesh directly to a 2D image without using matplotlib 3D axes.
        This completely avoids any axis-related issues.
        """
        try:
            # Use trimesh's built-in scene rendering
            scene = trimesh.Scene([mesh])
            
            # Set mesh color
            if hasattr(mesh.visual, 'face_colors'):
                # Convert color name to RGB if needed
                if isinstance(color, str):
                    import matplotlib.colors as mcolors
                    rgb = mcolors.to_rgb(color)
                    rgba = [int(c * 255) for c in rgb] + [255]
                else:
                    rgba = color
                mesh.visual.face_colors = rgba
            
            # Render to image with white background
            try:
                # Try using pyrender if available
                image = scene.save_image(resolution=[size, size], background=[255, 255, 255, 255])
                return np.array(image)
            except:
                # Fallback: create simple 2D projection
                return self._create_simple_projection(mesh, color, size)
                
        except Exception:
            # Ultimate fallback
            return self._create_simple_projection(mesh, color, size)
    
    def _create_simple_projection(self, mesh, color, size):
        """Create a simple 2D projection of the mesh."""
        try:
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Create simple projection (front view)
            x = vertices[:, 0]
            y = vertices[:, 1]
            
            # Create image using matplotlib but without 3D axes
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='white')
            ax.set_facecolor('white')
            ax.axis('off')
            
            # Draw filled triangles
            ax.tripcolor(x, y, faces, facecolors=color, alpha=1.0, edgecolors='none')
            
            # Set aspect and limits
            ax.set_aspect('equal')
            range_x = x.max() - x.min()
            range_y = y.max() - y.min()
            max_range = max(range_x, range_y)
            center_x = (x.max() + x.min()) / 2
            center_y = (y.max() + y.min()) / 2
            
            ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
            ax.set_ylim(center_y - max_range/2, center_y + max_range/2)
            
            # Remove margins completely
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Convert to numpy array
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            
            plt.close(fig)
            
            return image
            
        except Exception:
            # Final fallback - create a solid color square
            image = np.ones((size, size, 3), dtype=np.uint8) * 255
            return image

# Initialize the enhanced visualizer with metrics support
visualizer = HolographicReconstructionVisualizer(
    mapping_df, MESH_BASE_DIR, IMAGE_BASE_DIR, dataset=dataset, results_df=results_df
)

# %%
# Production-Ready Functions


def show_overview():
    """Show overview grid of all reconstructions using DataLoader images."""
    visualizer.create_overview_grid()


def show_specific(indices):
    """Show specific reconstructions by index with both holographic images."""
    if not isinstance(indices, list):
        indices = [indices]

    for idx in indices:
        if 0 <= idx < len(mapping_df):
            timestamp = mapping_df.iloc[idx]["timestamp"]
            visualizer.plot_single_comparison(timestamp, verbose=False)
        else:
            print(f"Index {idx} out of range (0-{len(mapping_df)-1})")


def show_random(n=3):
    """Show n random reconstructions with both holographic images."""
    import random

    indices = random.sample(range(len(mapping_df)), min(n, len(mapping_df)))
    show_specific(indices)


def test_dataloader_integration():
    """Test if DataLoader integration is working properly."""
    if len(mapping_df) > 0:
        timestamp = mapping_df.iloc[0]["timestamp"]
        img0, img1 = visualizer.find_image_pair_from_dataset(timestamp)
        if img0 is not None and img1 is not None:
            print("DataLoader integration working")
            return True
        else:
            print("DataLoader integration failed")
            return False
    else:
        print("No mapping data available")
        return False


def test_metrics_integration():
    """Test if evaluation metrics are available and integrated."""
    if results_df is not None:
        print("Metrics integration available")
        print(f"Available metrics: {list(results_df.columns)}")
        return True
    else:
        print("No evaluation metrics found")
        return False


# Enhanced Usage Guide
print("=" * 70)
print("ENHANCED HOLOGRAPHIC RECONSTRUCTION VISUALIZER")
print("=" * 70)
print("Functions:")
print("- show_overview() - Grid overview of all samples")
print("- show_specific([0,1,2]) - Show specific indices with metrics")
print("- show_random(3) - Show random samples with metrics")
print("- test_dataloader_integration() - Test DataLoader")
print("- test_metrics_integration() - Test metrics availability")
print("=" * 70)

# %%
def set_axes_equal_and_zoom(ax, mesh, zoom=0.8):
    """Set equal aspect ratio and zoom for better mesh visualization"""
    xyz = mesh.vertices
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2
    half = (maxs - mins).max() * zoom / 2

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def create_comprehensive_augmentation_plot():
    """Create a comprehensive thesis-ready plot showing all augmentation strategies using optimized plotting logic"""

    # Calculate grid dimensions
    n_meshes = len(sample_meshes)
    cols = 4  # 4 columns for better layout
    rows = (n_meshes + cols - 1) // cols  # Calculate needed rows

    # Sort meshes to put original first
    sorted_meshes = sorted(
        sample_meshes.items(), key=lambda x: (x[0] != "original", x[0])
    )

    # Create figure with appropriate size
    fig, axes = plt.subplots(
        rows,
        cols,
        subplot_kw={"projection": "3d"},
        figsize=(20, 6 * rows),
        squeeze=False,
    )

    for idx, (strategy, mesh) in enumerate(sorted_meshes):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Remove all axes and background
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for spine in (ax.xaxis, ax.yaxis, ax.zaxis):
            spine.pane.set_edgecolor("none")
        ax.set_axis_off()

        # Get color for this strategy
        color = augmentation_colors.get(strategy, "#666666")

        # Plot the mesh
        try:
            ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                mesh.vertices[:, 2],
                triangles=mesh.faces,
                color=color,
            )
        except Exception as e:
            print(f"Error plotting {strategy}: {e}")
            # Fallback to scatter plot
            ax.scatter(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                mesh.vertices[:, 2],
                c=color,
                alpha=0.6,
                s=1,
            )

        # Set title
        ax.set_title(
            f'{strategy.replace("_", " ").title()}',
            fontsize=30,  # Font size 30
            fontweight="normal",
        )

        # Set equal aspect ratio and zoom
        set_axes_equal_and_zoom(ax, mesh, zoom=0.8)

        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)

    # Hide unused subplots
    for idx in range(len(sorted_meshes), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


# Create the comprehensive plot
create_comprehensive_augmentation_plot()

# %%
