import os
import sys
import pandas as pd
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from dotenv import load_dotenv
from mesh_utils import MeshUtils

# --- Load environment variables from .env at the repo root ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(f"Loading environment variables from {repo_root}/.env")
dotenv_path = os.path.join(repo_root, ".env")
env_loaded = load_dotenv(dotenv_path)

def print_example_env():
    print("\nExample .env file (place at the root of your repository):\n")
    print("""DATA_DIR_PATH = 'C:/Users/example/Documents/GitHub/sequoia/data'
PIXEL_NERF_CONF_DIR = 'C:/Users/example/Documents/GitHub/sequoia/Pixel_Nerf/expconf.conf'

CHECKPOINTS_ROOT = 'C:/Users/example/Documents/Github/sequoia/Pixel_Nerf/checkpoints'
CONF_MAP_PATH = 'C:/Users/example/Documents/Github/sequoia/Eval/MeshGenerator/src/checkpoint_conf_map.json'
DATADIR = 'C:/Users/example/Documents/Github/shapenet_renderer/128_views/pollen_augmented'
OUTPUT_ROOT = 'C:/Users/example/Documents/Github/sequoia/Pixel_Nerf/eval/reconstructed'
PIXELNERF_SCRIPT = 'C:/Users/example/Documents/Github/sequoia/Eval/MeshGenerator/src/pixelnerf.py'

PRED_ROOT = 'C:/Users/example/Documents/Github/sequoia/TestEvaluationPipeline/data'
GT_ROOT = 'C:/Users/example/Documents/Github/sequoia/data/processed/interim'
GT_ROOT_AUG = 'C:/Users/example/Documents/Github/sequoia/data/processed/augmented'
CSV_SAVE_PATH = 'C:/Users/example/Documents/Github/sequoia/Eval/TestEvaluation/mesh_eval_results.csv'
""")

# Read from .env or use default fallback paths for holo data
def get_env_var_or_default(key, default_value):
    val = os.getenv(key)
    if val is None:
        return default_value
    return val.strip("'")

# Holo-specific paths
HOLO_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data_holo")
GT_HOLO_ROOT = os.path.join(HOLO_DATA_ROOT, "vh_2img_holo_test")
CSV_SAVE_PATH_HOLO = os.path.join(os.path.dirname(__file__), "test", "mesh_eval_results_holo.csv")

GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "test", "plots_holo")
os.makedirs(PLOTS_DIR, exist_ok=True)
COMPARE_DIR = os.path.join(os.path.dirname(__file__), "test", "compare_holo")
os.makedirs(COMPARE_DIR, exist_ok=True)

class HoloMeshEvaluator:
    def __init__(self, holo_data_root, gt_holo_root, csv_save_path, plot_plots=True, plot_compare=True, debug=True, voxel_iou_on=False):
        self.holo_data_root = holo_data_root
        self.gt_holo_root = gt_holo_root
        self.csv_save_path = csv_save_path
        self.results = []
        self.done_set = set()
        self.plot_plots = plot_plots
        self.plot_compare = plot_compare
        self.debug = debug
        self.voxel_iou_on = voxel_iou_on
        
        # Prediction model directories
        self.pred_models = {
            "pix2vox_aug_holo_test": os.path.join(holo_data_root, "pix2vox_aug_holo_test"),
            "Pixel2MeshPlusPlus": os.path.join(holo_data_root, "Pixel2MeshPlusPlus"),
            "PixelNerf_aug_holo": os.path.join(holo_data_root, "PixelNerf_aug_holo")
        }
        
        self._load_existing_results()
        self._build_timestamp_mapping()

    def _load_existing_results(self):
        if os.path.exists(self.csv_save_path):
            df = pd.read_csv(self.csv_save_path)
            self.done_set = set(df['pred_path'].tolist())
            self.results = df.to_dict('records')
        else:
            self.done_set = set()
            self.results = []

    def _extract_timestamp(self, filename):
        """Extract timestamp from filename using regex"""
        # Pattern to match the timestamp format: YYYY-MM-DD_HH.MM.SS.microseconds
        pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}\.\d+)'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None

    def _build_timestamp_mapping(self):
        """Build a mapping of timestamps to ground truth files - keeping ALL files"""
        self.timestamp_to_gt_files = {}  # timestamp -> list of GT files
        
        # Get all GT files and extract their timestamps
        gt_files = [
            f for f in os.listdir(self.gt_holo_root)
            if f.endswith(('.stl', '.obj'))
        ]
        
        for gt_file in gt_files:
            timestamp = self._extract_timestamp(gt_file)
            if timestamp:
                gt_path = os.path.join(self.gt_holo_root, gt_file)
                if timestamp not in self.timestamp_to_gt_files:
                    self.timestamp_to_gt_files[timestamp] = []
                self.timestamp_to_gt_files[timestamp].append((gt_file, gt_path))
        
        # For backward compatibility, create the old mapping too (takes first file per timestamp)
        self.timestamp_to_gt = {}
        for timestamp, files in self.timestamp_to_gt_files.items():
            self.timestamp_to_gt[timestamp] = files[0][1]  # Take first file path
        
        total_gt_files = sum(len(files) for files in self.timestamp_to_gt_files.values())
        print(f"Found {len(self.timestamp_to_gt_files)} unique timestamps covering {total_gt_files} GT files")

    def get_gt_path_by_timestamp(self, pred_path):
        """Find corresponding GT file using timestamp"""
        pred_filename = os.path.basename(pred_path)
        timestamp = self._extract_timestamp(pred_filename)
        
        if timestamp and timestamp in self.timestamp_to_gt_files:
            gt_files = self.timestamp_to_gt_files[timestamp]
            if len(gt_files) == 1:
                return gt_files[0][1]  # Return path of single file
            else:
                # If multiple GT files exist for this timestamp, try to find the best match
                # For now, prefer files without _1 suffix, or return the first one
                for gt_file, gt_path in gt_files:
                    if "_1" not in gt_file:
                        return gt_path
                return gt_files[0][1]  # Fallback to first file
        else:
            print(f"Warning: No GT found for timestamp {timestamp} in file {pred_filename}")
            return None

    def get_all_gt_paths_by_timestamp(self, timestamp):
        """Get all GT file paths for a given timestamp"""
        if timestamp in self.timestamp_to_gt_files:
            return [(gt_file, gt_path) for gt_file, gt_path in self.timestamp_to_gt_files[timestamp]]
        return []

    def collect_mesh_files(self, model_dir):
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(model_dir)
            for f in files if f.endswith((".stl", ".obj"))
        ]

    def evaluate_mesh(self, pred_mesh_path, gt_mesh_path, fscore_thresh=0.01):
        thresholds = [0.01, 0.025, 0.05]
        
        metric_names = [
            "Load meshes",
            "ICP alignment",
            "Normalize & convex hull",
            "Sample surface points",
            "Chamfer distance",
            "Hausdorff distance",
            "F-score (1%/2.5%/5%)",
            "Volume difference",
            "Surface area difference",
            "Edge length stats",
            "Voxel IoU",
            "Euler characteristic",
            "Normal consistency"
        ]
        results = None
        with tqdm(metric_names, desc="Metrics", leave=False) as metric_bar:
            # 1. Load meshes
            mesh_pred = trimesh.load(pred_mesh_path, process=False)
            mesh_gt = trimesh.load(gt_mesh_path, process=False)
            metric_bar.update(1)

            # 2. ICP alignment
            mesh_pred_aligned, pts_pred_aligned = MeshUtils.align_icp(mesh_pred, mesh_gt, n_points=5000)
            metric_bar.update(1)

            # 3. Normalize & convex hull
            mesh_pred_hull = mesh_pred_aligned
            mesh_gt_hull = MeshUtils.normalize_mesh(mesh_gt.copy())
            metric_bar.update(1)

            # 4. Sample surface points
            pts_pred = pts_pred_aligned
            face_idx_pred = np.random.randint(0, len(mesh_pred.faces), size=len(pts_pred))
            pts_gt, face_idx_gt = trimesh.sample.sample_surface(mesh_gt_hull, len(pts_pred))
            metric_bar.update(1)

            # 5. Chamfer distance
            chamfer = MeshUtils.chamfer_distance(pts_pred, pts_gt)
            metric_bar.update(1)

            # 6. Hausdorff distance
            hausdorff = MeshUtils.hausdorff_distance(pts_pred, pts_gt)
            metric_bar.update(1)

            # 7. F-scores
            # Pair-wise distances only once
            d1 = np.min(np.linalg.norm(pts_pred[:, None] - pts_gt[None], axis=-1), axis=1)
            d2 = np.min(np.linalg.norm(pts_gt[:, None] - pts_pred[None], axis=-1), axis=1)
            fscore_1, fscore_2_5, fscore_5 = MeshUtils.fscore_multi(d1, d2, thresholds)
            metric_bar.update(1)

            # 8. Volume difference
            vol_diff = MeshUtils.volume_difference(mesh_pred, mesh_gt)
            metric_bar.update(1)

            # 9. Surface area difference
            area_diff = MeshUtils.surface_area_difference(mesh_pred, mesh_gt)
            metric_bar.update(1)

            # 10. Edge length stats
            edge_mean_pred, edge_std_pred = MeshUtils.edge_length_stats(mesh_pred)
            edge_mean_gt, edge_std_gt = MeshUtils.edge_length_stats(mesh_gt)
            metric_bar.update(1)

            # 11. Voxel IoU (32x32x32 grid) - use aligned and normalized meshes!
            bbox = np.vstack([mesh_pred_hull.bounds, mesh_gt_hull.bounds])
            bbox_min = bbox.min(axis=0)
            bbox_max = bbox.max(axis=0)
            max_dim = np.max(bbox_max - bbox_min)
            pitch = max_dim / 32.0 if max_dim > 0 else 1.0

            # Shift both meshes to the same origin for voxelization
            mesh_pred_vox = mesh_pred_hull.copy()
            mesh_gt_vox = mesh_gt_hull.copy()
            mesh_pred_vox.apply_translation(-bbox_min)
            mesh_gt_vox.apply_translation(-bbox_min)

            voxel_iou = MeshUtils.voxel_iou(mesh_pred_vox, mesh_gt_vox, pitch=pitch)
            metric_bar.update(1)

            # 12. Euler characteristic
            euler_pred = MeshUtils.euler_characteristic(mesh_pred)
            euler_gt = MeshUtils.euler_characteristic(mesh_gt)
            metric_bar.update(1)

            # 13. Normal consistency
            normals_pred = mesh_pred.face_normals[face_idx_pred]
            normals_gt = mesh_gt.face_normals[face_idx_gt]
            normal_consistency = MeshUtils.normal_consistency(pts_pred, pts_gt, normals_pred, normals_gt)
            metric_bar.update(1)

            results = (
                chamfer, 
                hausdorff, 
                fscore_1,
                fscore_2_5,
                fscore_5,
                mesh_pred_hull, 
                mesh_gt_hull,
                vol_diff, 
                area_diff,
                edge_mean_pred, 
                edge_std_pred, 
                edge_mean_gt, 
                edge_std_gt,
                voxel_iou, 
                euler_pred, 
                euler_gt, 
                normal_consistency,
                None
            )
            
        return results

    def plot_meshes(self, mesh_pred, mesh_gt, model_name, fname):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Sample points from both meshes
        pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, 5000)
        pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, 5000)

        # Define views: (azimuth, elevation)
        views = {
            "front": (0, 0),
            "back": (180, 0),
            "left": (90, 0),
            "right": (-90, 0),
            "top": (0, 90),
            "bottom": (0, -90),
            "isometric": (45, 35)
        }

        fig, axs = plt.subplots(1, len(views), figsize=(4 * len(views), 4), subplot_kw={'projection': '3d'})

        if len(views) == 1:
            axs = [axs]

        for ax, (name, (azim, elev)) in zip(axs, views.items()):
            ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], c='orange', s=1, label='GT')
            ax.scatter(pts_pred[:, 0], pts_pred[:, 1], pts_pred[:, 2], c='deepskyblue', s=1, label='Prediction')
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(name)
            ax.set_axis_off()

        # Add legend only once (shared legend)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"{model_name}_{os.path.splitext(fname)[0]}_multi_view_pc.png")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        return plot_path

    def plot_meshes_compare(self, mesh_pred, mesh_gt, model_name, fname):
        # Plot the original (untouched) meshes as surfaces, not point clouds
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # Plot predicted mesh (original, not convex hull)
        ax1.set_title('Prediction (Mesh)')
        ax1.plot_trisurf(
            mesh_pred.vertices[:, 0], mesh_pred.vertices[:, 1], mesh_pred.vertices[:, 2],
            triangles=mesh_pred.faces, color='deepskyblue', edgecolor='none', alpha=0.9
        )
        ax1.set_axis_off()

        # Plot GT mesh (original, not convex hull)
        ax2.set_title('Ground Truth (Mesh)')
        ax2.plot_trisurf(
            mesh_gt.vertices[:, 0], mesh_gt.vertices[:, 1], mesh_gt.vertices[:, 2],
            triangles=mesh_gt.faces, color='orange', edgecolor='none', alpha=0.9
        )
        ax2.set_axis_off()

        plt.tight_layout()
        compare_path = os.path.join(COMPARE_DIR, f"{model_name}_{os.path.splitext(fname)[0]}_compare.png")
        plt.savefig(compare_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        return compare_path

    def create_mapping_csv(self):
        """Create a CSV file showing the mapping between all files"""
        mapping_data = []
        
        # Debug: Print all GT files (not just unique timestamps)
        total_gt_files = sum(len(files) for files in self.timestamp_to_gt_files.values())
        print(f"\n=== DEBUG: Found {len(self.timestamp_to_gt_files)} unique timestamps covering {total_gt_files} GT files ===")
        
        for timestamp in sorted(self.timestamp_to_gt_files.keys()):
            gt_files = self.timestamp_to_gt_files[timestamp]
            print(f"Timestamp {timestamp}: {len(gt_files)} files")
            for gt_file, gt_path in gt_files:
                print(f"  -> {gt_file}")
        
        # Debug: Print all prediction files per model
        for model_name, model_dir in self.pred_models.items():
            if os.path.exists(model_dir):
                files = [f for f in os.listdir(model_dir) if f.endswith(('.stl', '.obj'))]
                print(f"\n=== {model_name} ({len(files)} files) ===")
                pred_timestamps = []
                for file in files:
                    timestamp = self._extract_timestamp(file)
                    if timestamp:
                        pred_timestamps.append(timestamp)
                        print(f"  {timestamp} -> {file}")
                    else:
                        print(f"  NO_TIMESTAMP -> {file}")
                print(f"Unique timestamps in {model_name}: {len(set(pred_timestamps))}")
            else:
                print(f"\n=== {model_name} (directory not found) ===")
        
        # Create mapping for ALL GT files (not just unique timestamps)
        for timestamp, gt_files in self.timestamp_to_gt_files.items():
            for gt_file, gt_path in gt_files:
                mapping_entry = {
                    "timestamp": timestamp,
                    "gt_file": gt_file,
                    "pix2vox_file": "",
                    "pixel2mesh_file": "",
                    "pixelnerf_file": ""
                }
                
                # Find corresponding prediction files
                for model_name, model_dir in self.pred_models.items():
                    if os.path.exists(model_dir):
                        files = [f for f in os.listdir(model_dir) if f.endswith(('.stl', '.obj'))]
                        # Find files that match this specific GT file
                        matched_files = []
                        for file in files:
                            if timestamp in file:
                                matched_files.append(file)
                        
                        # For duplicates, try to match variants (e.g., _1 suffix)
                        best_match = ""
                        if matched_files:
                            if len(matched_files) == 1:
                                best_match = matched_files[0]
                            else:
                                # If GT file has _1 suffix, prefer prediction file with similar pattern
                                if "_1" in gt_file:
                                    # Look for prediction files that might correspond to the _1 variant
                                    for pred_file in matched_files:
                                        if pred_file not in [entry["pix2vox_file"] for entry in mapping_data if entry["timestamp"] == timestamp]:
                                            best_match = pred_file
                                            break
                                    if not best_match:
                                        best_match = matched_files[-1]  # Take last available
                                else:
                                    # For normal GT files, prefer the first prediction file
                                    for pred_file in matched_files:
                                        if pred_file not in [entry["pix2vox_file"] for entry in mapping_data if entry["timestamp"] == timestamp]:
                                            best_match = pred_file
                                            break
                                    if not best_match:
                                        best_match = matched_files[0]  # Take first available
                        
                        if best_match:
                            if model_name == "pix2vox_aug_holo_test":
                                mapping_entry["pix2vox_file"] = best_match
                            elif model_name == "Pixel2MeshPlusPlus":
                                mapping_entry["pixel2mesh_file"] = best_match
                            elif model_name == "PixelNerf_aug_holo":
                                mapping_entry["pixelnerf_file"] = best_match
                
                mapping_data.append(mapping_entry)
        
        # Debug: Check for incomplete mappings
        print(f"\n=== MAPPING RESULTS ===")
        complete_mappings = 0
        for entry in mapping_data:
            has_all = entry["pix2vox_file"] and entry["pixel2mesh_file"] and entry["pixelnerf_file"]
            if not has_all:
                missing = []
                if not entry["pix2vox_file"]: missing.append("pix2vox")
                if not entry["pixel2mesh_file"]: missing.append("pixel2mesh") 
                if not entry["pixelnerf_file"]: missing.append("pixelnerf")
                print(f"GT file {entry['gt_file']}: Missing {', '.join(missing)}")
            else:
                complete_mappings += 1
        
        print(f"\nComplete mappings (all 3 models): {complete_mappings}/{len(mapping_data)}")
        print(f"Total GT files mapped: {len(mapping_data)}")
        
        # Save mapping CSV
        mapping_df = pd.DataFrame(mapping_data)
        mapping_csv_path = os.path.join(os.path.dirname(self.csv_save_path), "holo_file_mapping.csv")
        mapping_df.to_csv(mapping_csv_path, index=False)
        print(f"\nFile mapping saved to: {mapping_csv_path}")
        return mapping_df

    def run(self):
        print(f"Saving CSV results to: {self.csv_save_path}")
        
        # First, create the mapping CSV
        mapping_df = self.create_mapping_csv()
        
        # Process each prediction model
        for model_name, model_dir in self.pred_models.items():
            if not os.path.exists(model_dir):
                print(f"Warning: Model directory {model_dir} does not exist")
                continue
                
            print(f"\nProcessing model: {model_name}")
            mesh_files = self.collect_mesh_files(model_dir)
            
            with tqdm(mesh_files, desc=f"{BLUE}Meshes in {model_name}{RESET}", leave=False, bar_format="{l_bar}%s{bar}%s{r_bar}" % (BLUE, RESET)) as mesh_bar:
                for pred_mesh_path in mesh_bar:
                    if pred_mesh_path in self.done_set:
                        continue
                        
                    fname = os.path.basename(pred_mesh_path)
                    gt_mesh_path = self.get_gt_path_by_timestamp(pred_mesh_path)
                    
                    if gt_mesh_path is None or not os.path.exists(gt_mesh_path):
                        result = {
                            "model": model_name,
                            "pred_path": pred_mesh_path,
                            "gt_path": gt_mesh_path or "NOT_FOUND",
                            "timestamp": self._extract_timestamp(fname),
                            "chamfer": "GT_NOT_FOUND",
                            "hausdorff": "GT_NOT_FOUND",
                            "fscore_1": "GT_NOT_FOUND",
                            "fscore_2_5": "GT_NOT_FOUND",
                            "fscore_5": "GT_NOT_FOUND",
                            "plot_path": "",
                            "compare_path": "",
                            "vol_diff": np.nan,
                            "area_diff": np.nan,
                            "edge_mean_pred": np.nan,
                            "edge_std_pred": np.nan,
                            "edge_mean_gt": np.nan,
                            "edge_std_gt": np.nan,
                            "voxel_iou": np.nan,
                            "euler_pred": np.nan,
                            "euler_gt": np.nan,
                            "normal_consistency": np.nan,
                            "voxel_plot_path": ""
                        }
                    else:
                        try:
                            (
                                chamfer, hausdorff, f1, f2_5, f5, mesh_pred_hull, mesh_gt_hull,
                                vol_diff, area_diff,
                                edge_mean_pred, edge_std_pred, edge_mean_gt, edge_std_gt,
                                voxel_iou, euler_pred, euler_gt, normal_consistency, voxel_plot_path
                            ) = self.evaluate_mesh(pred_mesh_path, gt_mesh_path)
                            
                            mesh_pred_orig = trimesh.load(pred_mesh_path, process=False)
                            mesh_gt_orig = trimesh.load(gt_mesh_path, process=False)
                            
                            plot_path = ""
                            compare_path = ""
                            
                            if self.plot_plots and mesh_pred_hull is not None and mesh_gt_hull is not None:
                                try:
                                    plot_path = self.plot_meshes(mesh_pred_hull, mesh_gt_hull, model_name, fname)
                                except Exception as e:
                                    plot_path = f"ERROR: {e}"
                                    
                            if self.plot_compare and mesh_pred_orig is not None and mesh_gt_orig is not None:
                                try:
                                    compare_path = self.plot_meshes_compare(mesh_pred_orig, mesh_gt_orig, model_name, fname)
                                except Exception as e:
                                    compare_path = f"ERROR: {e}"
                            
                            result = {
                                "model": model_name,
                                "pred_path": pred_mesh_path,
                                "gt_path": gt_mesh_path,
                                "timestamp": self._extract_timestamp(fname),
                                "chamfer": chamfer,
                                "hausdorff": hausdorff,
                                "fscore_1": f1,
                                "fscore_2_5": f2_5,
                                "fscore_5": f5,
                                "plot_path": plot_path,
                                "compare_path": compare_path,
                                "vol_diff": vol_diff,
                                "area_diff": area_diff,
                                "edge_mean_pred": edge_mean_pred,
                                "edge_std_pred": edge_std_pred,
                                "edge_mean_gt": edge_mean_gt,
                                "edge_std_gt": edge_std_gt,
                                "voxel_iou": voxel_iou,
                                "euler_pred": euler_pred,
                                "euler_gt": euler_gt,
                                "normal_consistency": normal_consistency,
                                "voxel_plot_path": voxel_plot_path
                            }
                        except Exception as e:
                            print(f"Error evaluating {fname}: {e}")
                            result = {
                                "model": model_name,
                                "pred_path": pred_mesh_path,
                                "gt_path": gt_mesh_path,
                                "timestamp": self._extract_timestamp(fname),
                                "chamfer": f"ERROR: {e}",
                                "hausdorff": f"ERROR: {e}",
                                "fscore_1": f"ERROR: {e}",
                                "fscore_2_5": f"ERROR: {e}",
                                "fscore_5": f"ERROR: {e}",
                                "plot_path": "",
                                "compare_path": "",
                                "vol_diff": np.nan,
                                "area_diff": np.nan,
                                "edge_mean_pred": np.nan,
                                "edge_std_pred": np.nan,
                                "edge_mean_gt": np.nan,
                                "edge_std_gt": np.nan,
                                "voxel_iou": np.nan,
                                "euler_pred": np.nan,
                                "euler_gt": np.nan,
                                "normal_consistency": np.nan,
                                "voxel_plot_path": ""
                            }
                    
                    # Print progress information
                    if isinstance(result["chamfer"], (int, float)):
                        tqdm.write(f"{model_name}/{fname}: chamfer={result['chamfer']:.4g}, "
                                 f"haus={result['hausdorff']:.4g}, F1={result['fscore_1']:.3f}, F2.5={result['fscore_2_5']:.3f}, F5={result['fscore_5']:.3f}")
                    else:
                        tqdm.write(f"{model_name}/{fname}: {result['chamfer']}")
                    
                    self.results.append(result)
                    self.done_set.add(pred_mesh_path)
                    self.save_results(incremental=True)
        
        self.save_results(incremental=False)

    def save_results(self, incremental=False):
        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_save_path, index=False)
        if not incremental:
            print("\nSummary Table:")
            print(df)
            print(f"\nResults saved to: {self.csv_save_path}")

if __name__ == "__main__":
    evaluator = HoloMeshEvaluator(
        holo_data_root=HOLO_DATA_ROOT,
        gt_holo_root=GT_HOLO_ROOT,
        csv_save_path=CSV_SAVE_PATH_HOLO,
        plot_plots=True,       # Set to False to disable plots in 'plots_holo'
        plot_compare=False,    # Set to False to disable plots in 'compare_holo'
        debug=False,           # Set to False to disable debug timing output
        voxel_iou_on=True     # Set to False to disable voxel IoU calculation
    )
    
    print("Starting Holography Mesh Evaluation...")
    evaluator.run()
