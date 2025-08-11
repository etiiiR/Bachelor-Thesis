#!/usr/bin/env python3
"""
Script to generate and save all images from the HolographicPollenDataset.

This script loads the dataset, processes all image pairs, and saves them
to an output directory with optional preprocessing transformations.

Usage:
    python generate_holo_images.py --output-dir ./generated_images
    python generate_holo_images.py --output-dir ./test_images --max-pairs 10 --create-pairs
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import json
import cv2
import numpy as np
from PIL import Image

# Load environment variables
load_dotenv()


class HolographicPollenDataset:
    """Simplified version of the HolographicPollenDataset for image generation."""
    
    def __init__(self, transform=None, extensions=None):
        data_dir = os.getenv("DATA_DIR_PATH")
        if data_dir is None:
            raise ValueError("DATA_DIR_PATH environment variable is not set")
        self.root_dir = os.path.join(data_dir, "subset_poleno")
        self.transform = transform
        self.extensions = extensions or [".png"]

        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

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
            if 'image_pairs' not in fname:
                continue
            base = fname.split('image_pairs')[0]
            groups.setdefault((base, taxa), []).append(path)

        # Build pairs list
        self.pairs = []  # (path0, path1, taxa_name)
        for (base, taxa), paths in groups.items():
            p0 = next((p for p in paths if '.0.' in os.path.basename(p)), None)
            p1 = next((p for p in paths if '.1.' in os.path.basename(p)), None)
            
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
            return Image.fromarray(arr.astype(np.uint8), mode='L')

        img0 = load_and_normalize(path0)
        img1 = load_and_normalize(path1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (img0, img1), taxa


class RemoveRipples:
    """Image preprocessing class to remove ripples and enhance pollen grain visibility."""
    
    def __init__(self,
                 method: str = 'otsu',
                 blur_ksize: int = 5,
                 adaptive_blocksize: int = 51,
                 adaptive_C: int = 2,
                 morph_ksize: int = 5,
                 max_scale: float = 1.7):
        self.method = method.lower()
        self.blur_ksize = blur_ksize
        self.adaptive_blocksize = adaptive_blocksize
        self.adaptive_C = adaptive_C
        self.morph_ksize = morph_ksize
        self.max_scale = max_scale

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_ksize, self.morph_ksize)
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        orig = np.array(img.convert('L'), dtype=np.uint8)
        h, w = orig.shape

        # blur & threshold
        blur = cv2.GaussianBlur(orig, (self.blur_ksize,)*2, 0)
        if self.method == 'otsu':
            _, mask = cv2.threshold(
                blur, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
        else:
            mask = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.adaptive_blocksize,
                self.adaptive_C
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        # apply mask
        result = np.where(mask == 255, orig, 255).astype(np.uint8)

        # find bbox of object
        ys, xs = np.where(result < 255)
        if len(xs) == 0 or len(ys) == 0:
            return Image.new('L', (w, h), color=255)

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        crop = result[y1:y2+1, x1:x2+1]

        crop_h, crop_w = crop.shape
        max_by_frame = min(w / crop_w, h / crop_h)
        scale = min(self.max_scale, max_by_frame)

        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        # resize & paste
        crop_pil = Image.fromarray(crop, mode='L')
        resized = crop_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        canvas = Image.new('L', (w, h), color=255)
        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2
        canvas.paste(resized, (x_off, y_off))

        return canvas


def create_output_structure(output_dir: Path, taxa_list: list):
    """Create output directory structure for organized image storage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each taxa
    for taxa in taxa_list:
        taxa_dir = output_dir / taxa
        taxa_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different image types
        (taxa_dir / "original").mkdir(exist_ok=True)
        (taxa_dir / "processed").mkdir(exist_ok=True)
        (taxa_dir / "pairs").mkdir(exist_ok=True)


def save_image_pair(img0, img1, taxa, base_name, output_dir: Path, save_processed=True):
    """Save an image pair to the output directory."""
    taxa_dir = output_dir / taxa
    
    # Save original images
    img0_path = taxa_dir / "original" / f"{base_name}_image0.png"
    img1_path = taxa_dir / "original" / f"{base_name}_image1.png"
    
    img0.save(img0_path)
    img1.save(img1_path)
    
    # Save processed images if requested
    if save_processed:
        ripple_remover = RemoveRipples()
        
        processed_img0 = ripple_remover(img0)
        processed_img1 = ripple_remover(img1)
        
        proc_img0_path = taxa_dir / "processed" / f"{base_name}_image0_processed.png"
        proc_img1_path = taxa_dir / "processed" / f"{base_name}_image1_processed.png"
        
        processed_img0.save(proc_img0_path)
        processed_img1.save(proc_img1_path)
    
    return img0_path, img1_path


def create_image_pairs_visualization(img0, img1, taxa, base_name, output_dir: Path):
    """Create side-by-side visualization of image pairs."""
    from PIL import Image
    import numpy as np
    
    # Convert to arrays
    arr0 = np.array(img0)
    arr1 = np.array(img1)
    
    # Ensure same height
    h0, w0 = arr0.shape
    h1, w1 = arr1.shape
    
    max_height = max(h0, h1)
    
    # Pad arrays to same height if needed
    if h0 < max_height:
        pad_h = max_height - h0
        arr0 = np.pad(arr0, ((0, pad_h), (0, 0)), mode='constant', constant_values=255)
    if h1 < max_height:
        pad_h = max_height - h1
        arr1 = np.pad(arr1, ((0, pad_h), (0, 0)), mode='constant', constant_values=255)
    
    # Concatenate horizontally
    combined = np.hstack([arr0, arr1])
    
    # Save combined image
    combined_img = Image.fromarray(combined, mode='L')
    pair_path = output_dir / taxa / "pairs" / f"{base_name}_pair.png"
    combined_img.save(pair_path)
    
    return pair_path


def generate_dataset_summary(dataset, output_dir: Path):
    """Generate a summary JSON file with dataset statistics."""
    summary = {
        "total_pairs": len(dataset.pairs),
        "total_taxa": len(dataset.classes),
        "taxa_list": dataset.classes,
        "pairs_per_taxa": {},
        "sample_files": {}
    }
    
    # Count pairs per taxa
    for _, _, taxa in dataset.pairs:
        summary["pairs_per_taxa"][taxa] = summary["pairs_per_taxa"].get(taxa, 0) + 1
    
    # Sample file paths for each taxa
    for taxa in dataset.classes:
        taxa_pairs = [(p0, p1) for p0, p1, t in dataset.pairs if t == taxa]
        if taxa_pairs:
            summary["sample_files"][taxa] = {
                "sample_pair": taxa_pairs[0],
                "total_pairs": len(taxa_pairs)
            }
    
    # Save summary
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate all images from HolographicPollenDataset")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./generated_holo_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--max-pairs", 
        type=int, 
        default=None,
        help="Maximum number of pairs to process (for testing)"
    )
    parser.add_argument(
        "--skip-processed", 
        action="store_true",
        help="Skip generation of processed images (faster)"
    )
    parser.add_argument(
        "--create-pairs", 
        action="store_true",
        help="Create side-by-side pair visualizations"
    )
    parser.add_argument(
        "--filter-taxa", 
        nargs="*",
        help="Only process specific taxa (space-separated list)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    
    print(f"Loading HolographicPollenDataset...")
    
    # Initialize dataset
    dataset = HolographicPollenDataset()
    
    print(f"Found {len(dataset.pairs)} image pairs across {len(dataset.classes)} taxa")
    
    # Filter by taxa if requested
    if args.filter_taxa:
        original_pairs = dataset.pairs
        dataset.pairs = [
            (p0, p1, taxa) for p0, p1, taxa in dataset.pairs 
            if taxa in args.filter_taxa
        ]
        print(f"Filtered to {len(dataset.pairs)} pairs for taxa: {args.filter_taxa}")
    
    # Limit pairs if requested
    if args.max_pairs:
        dataset.pairs = dataset.pairs[:args.max_pairs]
        print(f"Limited to {len(dataset.pairs)} pairs")
    
    # Create output structure
    print(f"Creating output directory structure at: {output_dir}")
    create_output_structure(output_dir, dataset.classes)
    
    # Generate dataset summary
    print("Generating dataset summary...")
    summary = generate_dataset_summary(dataset, output_dir)
    
    # Process all image pairs
    print("Processing image pairs...")
    
    failed_pairs = []
    
    for idx in tqdm(range(len(dataset.pairs)), desc="Generating images"):
        try:
            # Get image pair and taxa
            (img0, img1), taxa = dataset[idx]
            
            # Extract base name from file path
            path0, path1, _ = dataset.pairs[idx]
            base_name = os.path.splitext(os.path.basename(path0))[0]
            
            # Save images
            img0_path, img1_path = save_image_pair(
                img0, img1, taxa, base_name, output_dir, 
                save_processed=not args.skip_processed
            )
            
            # Create pair visualization if requested
            if args.create_pairs:
                pair_path = create_image_pairs_visualization(
                    img0, img1, taxa, base_name, output_dir
                )
            
        except Exception as e:
            print(f"Failed to process pair {idx}: {e}")
            failed_pairs.append((idx, str(e)))
            continue
    
    # Report results
    print(f"\n=== Generation Complete ===")
    print(f"Successfully processed: {len(dataset.pairs) - len(failed_pairs)} pairs")
    print(f"Failed: {len(failed_pairs)} pairs")
    
    if failed_pairs:
        print(f"\nFailed pairs:")
        for idx, error in failed_pairs[:10]:  # Show first 10 failures
            print(f"  Pair {idx}: {error}")
        if len(failed_pairs) > 10:
            print(f"  ... and {len(failed_pairs) - 10} more")
    
    print(f"\nOutput saved to: {output_dir.absolute()}")
    print(f"Dataset summary: {output_dir / 'dataset_summary.json'}")
    
    # Print directory structure
    print(f"\nDirectory structure:")
    for taxa in sorted(summary["pairs_per_taxa"].keys()):
        count = summary["pairs_per_taxa"][taxa]
        print(f"  {taxa}/ ({count} pairs)")
        print(f"    ├── original/")
        if not args.skip_processed:
            print(f"    ├── processed/")
        if args.create_pairs:
            print(f"    └── pairs/")


if __name__ == "__main__":
    main()
