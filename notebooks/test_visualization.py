# Quick test of the 3D visualization with clean axes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
from pathlib import Path

# Load the mapping
mapping_path = "C:/Users/super/Documents/Github/sequoia/Eval/Holo/holo_file_mapping_corrected.csv"
mapping_df = pd.read_csv(mapping_path)

# Test with first sample
row = mapping_df.iloc[0]
print(f"Testing with timestamp: {row['timestamp']}")

# Mock data for testing - create a simple mesh
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
])
faces = np.array([
    [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
])

# Create figure exactly like exp_1_fine_tuning
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# Apply exact axis removal from exp_1_fine_tuning
for ax in (ax1, ax2, ax3):
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.set_axis_off()

# Set titles
ax1.set_title("Ground Truth", fontsize=14, fontweight="bold")
ax2.set_title("Pix2Vox", fontsize=14, fontweight="bold")
ax3.set_title("Visual Hull", fontsize=14, fontweight="bold")

# Plot test meshes
colors = ['lightblue', 'lightcoral', 'lightgreen']
for i, (ax, color) in enumerate(zip([ax1, ax2, ax3], colors)):
    # Offset each mesh slightly for variety
    offset_vertices = vertices + np.array([i*0.1, 0, 0])
    
    ax.plot_trisurf(
        offset_vertices[:, 0], 
        offset_vertices[:, 1], 
        offset_vertices[:, 2],
        triangles=faces, 
        alpha=1.0, 
        color=color, 
        shade=True,
        linewidth=0, 
        edgecolor='none'
    )
    
    # Set equal aspect
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)

plt.suptitle("Test Visualization - Clean Axes", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

print("Test completed! You should see 3 clean meshes without any axes.")
