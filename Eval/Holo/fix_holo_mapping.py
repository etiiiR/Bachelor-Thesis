import json
import os
import pandas as pd

# Load poleno selection
with open(r"C:\Users\super\Documents\Github\sequoia\data\poleno_selection.json", 'r') as f:
    poleno_selection = json.load(f)

# Define base paths
data_holo_path = r"C:\Users\super\Documents\Github\sequoia\TestEvaluationPipeline\data_holo"
pix2vox_path = os.path.join(data_holo_path, "pix2vox_aug_holo_test")
pixel2mesh_path = os.path.join(data_holo_path, "Pixel2MeshPlusPlus")
pixelnerf_path = os.path.join(data_holo_path, "PixelNerf_aug_holo")
gt_path = os.path.join(data_holo_path, "vh_2img_holo_test")

# Get actual files
pix2vox_files = os.listdir(pix2vox_path) if os.path.exists(pix2vox_path) else []
pixel2mesh_files = os.listdir(pixel2mesh_path) if os.path.exists(pixel2mesh_path) else []
pixelnerf_files = os.listdir(pixelnerf_path) if os.path.exists(pixelnerf_path) else []
gt_files = os.listdir(gt_path) if os.path.exists(gt_path) else []

print("Sample actual files:")
print(f"Pix2Vox: {pix2vox_files[:3]}")
print(f"Pixel2Mesh: {pixel2mesh_files[:3]}")
print(f"PixelNerf: {pixelnerf_files[:3]}")

# Create correct mapping based on actual file names
correct_mapping = []

# Process each species and sample
for species, samples in poleno_selection.items():
    for i, sample in enumerate(samples, 1):
        # Extract timestamp from sample name (remove the prefix part)
        # sample format: "poleno-27_2023-04-06_18.06.28.175254_ev.computed_data.holography.image_pairs.0.0.rec_mag"
        parts = sample.split('_')
        timestamp = f"{parts[1]}_{parts[2]}"
        
        # GT file (STL from the original sample name)
        gt_file = f"{sample}.stl"
        
        # Pix2Vox file (same as GT, but check if _1 variant exists)
        pix2vox_candidate1 = f"{sample}.stl"
        pix2vox_candidate2 = f"{sample}_1.stl"
        if pix2vox_candidate1 in pix2vox_files:
            pix2vox_file = pix2vox_candidate1
        elif pix2vox_candidate2 in pix2vox_files:
            pix2vox_file = pix2vox_candidate2
        else:
            pix2vox_file = pix2vox_candidate1  # Default
        
        # Pixel2Mesh file - find matching file
        device_number = parts[0].split('-')[1]  # Extract device number (e.g., "27" from "poleno-27")
        pixel2mesh_pattern = f"{species}_{device_number}_{timestamp}.obj"
        pixel2mesh_file = pixel2mesh_pattern
        
        # PixelNerf file - find matching file  
        pixelnerf_pattern = f"{species}__{i:02d}_{device_number}_{timestamp}.obj"
        pixelnerf_file = pixelnerf_pattern
        
        correct_mapping.append({
            'timestamp': timestamp,
            'gt_file': gt_file,
            'pix2vox_file': pix2vox_file,
            'pixel2mesh_file': pixel2mesh_file,
            'pixelnerf_file': pixelnerf_file
        })

# Create DataFrame and save
df = pd.DataFrame(correct_mapping)
df.to_csv('holo_file_mapping_corrected.csv', index=False)

print(f"\nCorrected mapping saved to holo_file_mapping_corrected.csv")
print(f"Total entries: {len(df)}")

# Show first few entries
print("\nFirst 5 entries:")
print(df.head())
