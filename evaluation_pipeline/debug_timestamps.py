import os
import re
import pandas as pd

# Holo-specific paths
HOLO_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data_holo")
GT_HOLO_ROOT = os.path.join(HOLO_DATA_ROOT, "vh_2img_holo_test")

def extract_timestamp(filename):
    """Extract timestamp from filename using regex"""
    # Pattern to match the timestamp format: YYYY-MM-DD_HH.MM.SS.microseconds
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}\.\d+)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def create_complete_mapping():
    # Prediction model directories
    pred_models = {
        "pix2vox_aug_holo_test": os.path.join(HOLO_DATA_ROOT, "pix2vox_aug_holo_test"),
        "Pixel2MeshPlusPlus": os.path.join(HOLO_DATA_ROOT, "Pixel2MeshPlusPlus"),
        "PixelNerf_aug_holo": os.path.join(HOLO_DATA_ROOT, "PixelNerf_aug_holo")
    }
    
    # Build timestamp to GT files mapping (keeping ALL files)
    timestamp_to_gt_files = {}
    if os.path.exists(GT_HOLO_ROOT):
        gt_files = [f for f in os.listdir(GT_HOLO_ROOT) if f.endswith(('.stl', '.obj'))]
        print(f"=== ALL GT FILES ({len(gt_files)} total) ===")
        for gt_file in gt_files:
            timestamp = extract_timestamp(gt_file)
            if timestamp:
                gt_path = os.path.join(GT_HOLO_ROOT, gt_file)
                if timestamp not in timestamp_to_gt_files:
                    timestamp_to_gt_files[timestamp] = []
                timestamp_to_gt_files[timestamp].append((gt_file, gt_path))
                print(f"  {timestamp} -> {gt_file}")
        
        print(f"\nUnique timestamps: {len(timestamp_to_gt_files)}")
        total_files = sum(len(files) for files in timestamp_to_gt_files.values())
        print(f"Total GT files: {total_files}")
        
        # Show duplicates
        for timestamp, files in timestamp_to_gt_files.items():
            if len(files) > 1:
                print(f"Duplicate timestamp {timestamp}: {len(files)} files")
                for gt_file, _ in files:
                    print(f"  - {gt_file}")
    
    # Create complete mapping for ALL GT files
    mapping_data = []
    
    for timestamp, gt_files in timestamp_to_gt_files.items():
        for gt_file, gt_path in gt_files:
            mapping_entry = {
                "timestamp": timestamp,
                "gt_file": gt_file,
                "pix2vox_file": "",
                "pixel2mesh_file": "",
                "pixelnerf_file": ""
            }
            
            # Find corresponding prediction files
            for model_name, model_dir in pred_models.items():
                if os.path.exists(model_dir):
                    files = [f for f in os.listdir(model_dir) if f.endswith(('.stl', '.obj'))]
                    # Find files that match this timestamp
                    matched_files = [f for f in files if timestamp in f]
                    
                    # Assign prediction files to match GT files
                    if matched_files:
                        # For duplicates, try to distribute files
                        pred_file = matched_files[0]  # Simple assignment for now
                        if len(matched_files) > 1:
                            # If this is a _1 GT file and we have multiple predictions, try to get second one
                            if "_1" in gt_file and len(matched_files) > 1:
                                pred_file = matched_files[1] if len(matched_files) > 1 else matched_files[0]
                        
                        if model_name == "pix2vox_aug_holo_test":
                            mapping_entry["pix2vox_file"] = pred_file
                        elif model_name == "Pixel2MeshPlusPlus":
                            mapping_entry["pixel2mesh_file"] = pred_file
                        elif model_name == "PixelNerf_aug_holo":
                            mapping_entry["pixelnerf_file"] = pred_file
            
            mapping_data.append(mapping_entry)
    
    # Save and analyze mapping
    print(f"\n=== MAPPING RESULTS ===")
    print(f"Total mappings created: {len(mapping_data)}")
    
    complete_mappings = 0
    incomplete_mappings = []
    
    for entry in mapping_data:
        has_all = entry["pix2vox_file"] and entry["pixel2mesh_file"] and entry["pixelnerf_file"]
        if has_all:
            complete_mappings += 1
        else:
            incomplete_mappings.append(entry)
    
    print(f"Complete mappings (all 3 models): {complete_mappings}/{len(mapping_data)}")
    
    if incomplete_mappings:
        print(f"\nIncomplete mappings: {len(incomplete_mappings)}")
        for entry in incomplete_mappings:
            missing = []
            if not entry["pix2vox_file"]: missing.append("pix2vox")
            if not entry["pixel2mesh_file"]: missing.append("pixel2mesh") 
            if not entry["pixelnerf_file"]: missing.append("pixelnerf")
            print(f"  {entry['gt_file']}: Missing {', '.join(missing)}")
    
    # Save to CSV
    mapping_df = pd.DataFrame(mapping_data)
    csv_path = os.path.join(os.path.dirname(__file__), "test", "holo_file_mapping_complete.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    mapping_df.to_csv(csv_path, index=False)
    print(f"\nComplete mapping saved to: {csv_path}")
    
    return mapping_df

if __name__ == "__main__":
    mapping_df = create_complete_mapping()
