# =============================================================================
# DATASET MODULE - Enhanced holographic pollen dataset with multi-mesh support
# =============================================================================

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import MESH_DIRECTORIES, MESH_EXTENSIONS


class HolographicPollenDataset(Dataset):
    """
    Original holographic pollen dataset with metadata extraction and mesh mapping.
    """
    
    def __init__(self, transform=None, extensions=None, mesh_dir=None):
        data_dir = os.getenv("DATA_DIR_PATH")
        if data_dir is None:
            # Fallback to relative path if environment variable is not set
            data_dir = "../data"
        self.root_dir = os.path.join(data_dir, "subset_poleno")
        self.transform = transform
        self.extensions = extensions or [".png"]
        self.mesh_dir = mesh_dir

        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        # Load and group image pairs
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

        # Build pairs list with metadata
        self.pairs = []
        for (base, taxa), paths in groups.items():
            p0 = next((p for p in paths if ".0." in os.path.basename(p)), None)
            p1 = next((p for p in paths if ".1." in os.path.basename(p)), None)

            if p0 and p1:
                metadata = self._extract_metadata(os.path.basename(p0))
                self.pairs.append((p0, p1, taxa, metadata))

    def _extract_metadata(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename."""
        metadata = {
            "filename": filename,
            "datetime_str": None,
            "datetime_obj": None,
            "pollen_id": None,
            "mesh_identifier": None,
        }

        # Extract pollen ID
        pollen_match = re.search(r"poleno-(\d+)", filename)
        if pollen_match:
            metadata["pollen_id"] = f"poleno-{pollen_match.group(1)}"

        # Extract datetime string
        datetime_pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}\.\d+)"
        datetime_match = re.search(datetime_pattern, filename)
        if datetime_match:
            datetime_str = datetime_match.group(1)
            metadata["datetime_str"] = datetime_str

            try:
                date_part, time_part = datetime_str.split("_")
                time_clean = time_part.replace(".", ":")

                if time_clean.count(":") > 2:
                    time_parts = time_clean.split(":")
                    time_clean = ":".join(time_parts[:3])
                    microseconds = int(time_parts[3])
                else:
                    microseconds = 0

                dt_str = f"{date_part} {time_clean}"
                dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                if microseconds:
                    dt_obj = dt_obj.replace(microsecond=microseconds)

                metadata["datetime_obj"] = dt_obj

            except ValueError as e:
                print(f"Warning: Could not parse datetime from {datetime_str}: {e}")

        # Create mesh identifier
        if metadata["pollen_id"] and metadata["datetime_str"]:
            metadata["mesh_identifier"] = f"{metadata['pollen_id']}_{metadata['datetime_str']}"

        return metadata

    def get_mesh_path(self, idx: int) -> Optional[str]:
        """Get corresponding mesh file path for a given index."""
        if not self.mesh_dir or not os.path.exists(self.mesh_dir):
            return None

        # Check for any mesh files in the directory (STL, OBJ, PLY)
        try:
            mesh_files = [f for f in os.listdir(self.mesh_dir) 
                         if f.lower().endswith(('.stl', '.obj', '.ply'))]
            
            if mesh_files:
                # For demonstration, return the first available mesh file
                # In a real scenario, you would match based on specific criteria
                return os.path.join(self.mesh_dir, mesh_files[0])
        except:
            pass

        # Original logic for structured directories
        _, _, taxa, metadata = self.pairs[idx]
        datetime_str = metadata.get("datetime_str")
        if not datetime_str:
            return None

        try:
            for filename in os.listdir(self.mesh_dir):
                if filename.lower().endswith(".obj") and datetime_str in filename:
                    mesh_path = os.path.join(self.mesh_dir, filename)
                    if os.path.exists(mesh_path):
                        return mesh_path
        except:
            pass

        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], str, Dict[str, Any]]:
        path0, path1, taxa, metadata = self.pairs[idx]

        def load_and_normalize(p):
            # Use the exact same loading logic as in the working example
            img = Image.open(p)
            arr = np.array(img).astype(np.float32)
            min_val, max_val = arr.min(), arr.max()
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val) * 255.0
            else:
                arr = np.zeros_like(arr)
            return Image.fromarray(arr.astype(np.uint8), mode="L")

        img0 = load_and_normalize(path0)
        img1 = load_and_normalize(path1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (img0, img1), taxa, metadata


class EnhancedHolographicPollenDataset(HolographicPollenDataset):
    """
    Enhanced dataset class supporting multiple mesh reconstruction methods.
    """
    
    def __init__(self, mesh_directories: Optional[Dict[str, str]] = None, 
                 transform=None, extensions=None):
        # Initialize parent class
        super().__init__(transform=transform, extensions=extensions, mesh_dir=None)
        
        self.mesh_directories = mesh_directories or MESH_DIRECTORIES
        
        # Analyze mesh directories
        self.mesh_analysis = {}
        for method, mesh_dir in self.mesh_directories.items():
            if os.path.exists(mesh_dir):
                self.mesh_analysis[method] = self._analyze_mesh_directory(mesh_dir, method)
                print(f"  {method}: {self.mesh_analysis[method]['count']} mesh files")
            else:
                print(f"  Warning: {method} directory not found: {mesh_dir}")
                self.mesh_analysis[method] = {'files': [], 'count': 0, 'extension': None}

    def _analyze_mesh_directory(self, mesh_dir: str, method: Optional[str] = None) -> Dict[str, Any]:
        """Analyze mesh directory structure with correct file extensions."""
        mesh_files = []
        extension = '.ply'  # Default extension
        
        if os.path.exists(mesh_dir):
            # Determine the correct extension for this method
            if method and method in MESH_EXTENSIONS:
                extension = MESH_EXTENSIONS[method]
            else:
                # Try both common extensions
                extension = '.obj' if any(f.endswith('.obj') for f in os.listdir(mesh_dir)) else '.ply'
            
            for file in os.listdir(mesh_dir):
                if file.endswith(extension):
                    mesh_files.append(file)
        
        return {
            'files': sorted(mesh_files),
            'count': len(mesh_files),
            'extension': extension if mesh_files else None
        }

    def get_mesh_paths(self, idx: int) -> Dict[str, Optional[str]]:
        """Get corresponding mesh paths for all reconstruction methods."""
        if idx >= len(self.pairs):
            return {}

        _, _, taxa, metadata = self.pairs[idx]
        datetime_str = metadata.get('datetime_str')
        
        mesh_paths = {}
        
        for method, mesh_dir in self.mesh_directories.items():
            if not os.path.exists(mesh_dir):
                mesh_paths[method] = None
                continue

            # For methods with same naming as PixelNerf (pixel2meshpp)
            if method in ['pixelnerf', 'pixel2meshpp']:
                if datetime_str:
                    matching_meshes = []
                    for mesh_file in self.mesh_analysis[method]['files']:
                        if datetime_str in mesh_file:
                            matching_meshes.append(mesh_file)
                    
                    if matching_meshes:
                        mesh_paths[method] = os.path.join(mesh_dir, matching_meshes[0])
                    else:
                        mesh_paths[method] = None
                else:
                    mesh_paths[method] = None
            
            # For methods with different naming (pix2vox, visual_hull)
            else:
                if datetime_str:
                    for mesh_file in self.mesh_analysis[method]['files']:
                        # Try datetime matching first
                        if datetime_str in mesh_file:
                            mesh_paths[method] = os.path.join(mesh_dir, mesh_file)
                            break
                        # Try partial datetime matching (just date)
                        elif datetime_str[:10] in mesh_file:
                            mesh_paths[method] = os.path.join(mesh_dir, mesh_file)
                            break
                    else:
                        mesh_paths[method] = None
                else:
                    mesh_paths[method] = None
        
        return mesh_paths

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], str, Dict[str, Any], Dict[str, Optional[str]]]:
        # Get original data
        (img0, img1), taxa, metadata = super().__getitem__(idx)
        
        # Add mesh paths
        mesh_paths = self.get_mesh_paths(idx)
        
        return (img0, img1), taxa, metadata, mesh_paths


def create_dataset(mesh_directories: Optional[Dict[str, str]] = None, 
                  transform=None) -> EnhancedHolographicPollenDataset:
    """
    Factory function to create enhanced dataset.
    
    Args:
        mesh_directories: Dictionary of mesh directories for different methods
        transform: Optional transform pipeline
        
    Returns:
        EnhancedHolographicPollenDataset instance
    """
    return EnhancedHolographicPollenDataset(
        mesh_directories=mesh_directories,
        transform=transform
    )
