import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Change this to the path of a file you want to check
FILE_PATH = "/DATA/CV/processed_data_2FPS/tvsum/JKpqYvAdIsw.pt" 
# OR use: "./processed_data/summe/Air_Force_One.pt"

def inspect_file(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    print(f"\n{'='*40}")
    print(f"INSPECTING: {os.path.basename(path)}")
    print(f"{'='*40}")

    # Load file (map_location ensures it loads even if you don't have a GPU right now)
    data = torch.load(path, map_location="cpu")

    # 1. Check Dictionary Keys
    print(f"Keys found: {list(data.keys())}")

    # 2. Analyze Features
    if 'features' in data:
        feats = data['features']
        print(f"\n[Features]")
        print(f"  Shape: {feats.shape}  (Frames x Hidden_Dim)")
        print(f"  Dtype: {feats.dtype}")
        print(f"  Device: {feats.device}")
        print(f"  Has NaNs: {torch.isnan(feats).any().item()}")
        print(f"  Min/Max: {feats.min().item():.4f} / {feats.max().item():.4f}")

    # 3. Analyze Ground Truth
    if 'gt_scores' in data:
        gt = data['gt_scores']
        print(f"\n[GT Scores]")
        print(f"  Shape: {gt.shape}")
        print(f"  Dtype: {gt.dtype}")
        print(f"  Min/Max: {gt.min().item():.4f} / {gt.max().item():.4f}")
        
        # Check alignment
        if 'features' in data:
            if feats.shape[0] == gt.shape[0]:
                print("  ✅ Alignment: Features and GT have same frame count.")
            else:
                print(f"  ❌ MISMATCH: Features {feats.shape[0]} vs GT {gt.shape[0]}")

    # 4. Visualization
    visualize(data, os.path.basename(path))

def visualize(data, title):
    plt.figure(figsize=(12, 4))
    
    # Plot GT Scores
    if 'gt_scores' in data:
        gt = data['gt_scores'].numpy()
        plt.plot(gt, label="Ground Truth Importance", color='orange', linewidth=2)

    # Plot Feature Norms (Visualizes how 'intense' the embedding is per frame)
    if 'features' in data:
        feats = data['features'].float().numpy()
        # L2 norm of features, normalized to 0-1 for plotting
        norms = np.linalg.norm(feats, axis=1)
        norms = (norms - norms.min()) / (norms.max() - norms.min())
        plt.plot(norms, label="Feature Magnitude (Llama3)", color='blue', alpha=0.3)

    plt.title(f"Content Analysis: {title}")
    plt.xlabel("Frame Segments")
    plt.ylabel("Score (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot or show
    plt.show()
    plt.savefig("inspection_plot_2FPS.png") # Uncomment to save instead

if __name__ == "__main__":
    # Allow running via command line: python inspect_pt.py path/to/file.pt
    if len(sys.argv) > 1:
        inspect_file(sys.argv[1])
    else:
        inspect_file(FILE_PATH)