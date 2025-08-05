# FIXED CLEAN VISUALIZATION - NO AXES, NO TRANSPARENCY ISSUES
# Copy this code into your Interactive notebook to replace the plot_single_comparison method

def plot_single_comparison_fixed(self, timestamp, verbose=False):
    """Create a clean single-row comparison plot with NO overlay and COMPLETELY NO axes."""
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
    fig.patch.set_facecolor('white')

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

        # Create 3D subplot in single row
        ax = fig.add_subplot(1, 6, i + 3, projection="3d")
        
        # COMPLETELY REMOVE ALL AXES AND BACKGROUND
        ax.set_facecolor('white')
        ax.set_axis_off()
        
        # Hide ALL 3D elements completely
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.zaxis.set_visible(False)
        
        # Remove grid completely
        ax.grid(False)
        
        # Remove all pane backgrounds and edges with multiple fallback methods
        try:
            # Method 1: Direct pane access
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False  
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')
            ax.xaxis.pane.set_alpha(0)
            ax.yaxis.pane.set_alpha(0)
            ax.zaxis.pane.set_alpha(0)
        except AttributeError:
            try:
                # Method 2: w_axis access
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False  
                ax.w_zaxis.pane.fill = False
                ax.w_xaxis.pane.set_edgecolor('white')
                ax.w_yaxis.pane.set_edgecolor('white')
                ax.w_zaxis.pane.set_edgecolor('white')
                ax.w_xaxis.pane.set_alpha(0)
                ax.w_yaxis.pane.set_alpha(0)
                ax.w_zaxis.pane.set_alpha(0)
            except:
                pass
        
        # Additional cleanup for stubborn axes
        try:
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        except:
            pass

        if mesh is not None:
            vertices = mesh.vertices
            faces = mesh.faces
            
            # SOLID mesh visualization - NO transparency, NO wireframe
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces, 
                alpha=1.0,  # SOLID, no transparency
                color=color, 
                shade=True,
                linewidth=0,  # NO edge lines
                edgecolor='none'  # NO edge colors
            )
            
            # Set equal aspect ratio
            max_range = (
                np.array(
                    [
                        vertices[:, 0].max() - vertices[:, 0].min(),
                        vertices[:, 1].max() - vertices[:, 1].min(),
                        vertices[:, 2].max() - vertices[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )

            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # LARGE title with clean label and more padding
            clean_label = label.replace("\n", " ")
            ax.set_title(clean_label, fontsize=30, fontweight="bold", color="black", pad=20)
        else:
            ax.text(0.5, 0.5, 0.5, "Not Available", 
                   horizontalalignment="center", verticalalignment="center", 
                   transform=ax.transAxes, fontsize=30, color="red")
            clean_label = label.replace("\n", " ")
            ax.set_title(f"{clean_label} - Missing", fontsize=30, color="red", pad=20)

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

# TO USE THIS FIX:
# 1. Run this code in a new cell
# 2. Then run: visualizer.plot_single_comparison = plot_single_comparison_fixed.__get__(visualizer)
# 3. Test with: show_specific([0])

print("✅ FIXED visualization method created!")
print("📝 To apply the fix, run in your notebook:")
print("   visualizer.plot_single_comparison = plot_single_comparison_fixed.__get__(visualizer)")
print("   show_specific([0])  # Test the fix")
