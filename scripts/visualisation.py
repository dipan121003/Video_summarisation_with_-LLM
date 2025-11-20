import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import savgol_filter
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import f1_score

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_DIM = 4096

# Target Dataset (used to find models)
DATASET_NAME = "tvsum" 

# Video to Visualize
TEST_VIDEO_PATH = "/DATA/CV/processed_data/tvsum/WxtbjNsCQ8A.pt"

# ==========================================
# MODEL ARCHITECTURE (SOTA)
# ==========================================
class GlobalContextNetwork(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=256):
        super().__init__()
        self.conv_short = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv_long = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.pos_emb = nn.Parameter(torch.randn(1, 5000, hidden_dim) * 0.01)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=1024, 
            dropout=0.3, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x1 = self.conv_short(x)
        x2 = self.conv_long(x)
        x = torch.cat([x1, x2], dim=1) 
        x = self.bn1(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1) 
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        x = self.transformer(x)
        return self.head(x).squeeze(-1)

# ==========================================
# METRIC UTILS
# ==========================================
def calculate_metrics(preds, gts):
    # Correlation
    rho, _ = spearmanr(preds, gts)
    tau, _ = kendalltau(preds, gts)
    
    # F1 (Top 15% Knapsack)
    k = max(1, int(len(preds) * 0.15))
    p_bin = np.zeros_like(preds); p_bin[np.argsort(preds)[-k:]] = 1
    g_bin = np.zeros_like(gts); g_bin[np.argsort(gts)[-k:]] = 1
    f1 = f1_score(g_bin, p_bin)
    
    return rho, tau, f1

# ==========================================
# MAIN LOGIC
# ==========================================
def run_ensemble_vis():
    # 1. Load Video Data
    video_path = TEST_VIDEO_PATH
    if not os.path.exists(video_path):
        files = glob.glob(f"./processed_data/{DATASET_NAME}/*.pt")
        if not files:
            print(f"ERROR: No files found in ./processed_data/{DATASET_NAME}/")
            return
        video_path = files[0]
        print(f"-> Auto-selected video: {os.path.basename(video_path)}")
    else:
        print(f"-> Visualizing: {os.path.basename(video_path)}")

    data = torch.load(video_path)
    feats = data['features'].float().unsqueeze(0).to(DEVICE)
    gt = data['gt_scores'].float().numpy()
    
    # Normalize GT
    if gt.max() > 1.0: gt = gt / gt.max()

    # 2. Find All Models
    model_pattern = f"best_{DATASET_NAME}_f*.pth"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"ERROR: No model files found matching '{model_pattern}'")
        return
    
    print(f"-> Found {len(model_files)} models for ensemble: {model_files}")

    # 3. Run Inference Loop
    all_preds = []
    
    for m_path in model_files:
        model = GlobalContextNetwork(input_dim=LLM_DIM).to(DEVICE)
        model.load_state_dict(torch.load(m_path, map_location=DEVICE))
        model.eval()
        
        with torch.no_grad():
            p = model(feats).cpu().numpy().flatten()
            all_preds.append(p)
            
    # 4. Average Predictions (Ensembling)
    avg_preds = np.mean(all_preds, axis=0)
    
    # 5. Calculate Metrics
    rho, tau, f1 = calculate_metrics(avg_preds, gt)
    print("\n" + "="*30)
    print(f"ENSEMBLE METRICS FOR {os.path.basename(video_path)}")
    print(f"Spearman Rho: {rho:.4f}")
    print(f"Kendall Tau:  {tau:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print("="*30 + "\n")

    # 6. Smoothing for Plot
    try:
        if len(avg_preds) > 7:
            preds_plot = savgol_filter(avg_preds, window_length=7, polyorder=2)
        else:
            preds_plot = avg_preds
    except:
        preds_plot = avg_preds

    # 7. Plotting
    plt.figure(figsize=(15, 6))
    
    x_axis = range(len(gt))
    
    # Plot GT
    plt.fill_between(x_axis, gt, color='gray', alpha=0.3, label='Human Ground Truth')
    plt.plot(x_axis, gt, color='black', alpha=0.2, linewidth=0.5)
    
    # Plot Individual Models (Faint lines)
    for p in all_preds:
        plt.plot(x_axis, p, color='red', alpha=0.1, linewidth=0.5)

    # Plot Ensemble Average (Strong line)
    plt.plot(x_axis, preds_plot, color='#D32F2F', linewidth=3, label=f'Ensemble ({len(model_files)} Models)')
    
    # Text Box for Metrics
    stats_text = (f"$\\rho$ (Spearman): {rho:.3f}\n"
                  f"$\\tau$ (Kendall): {tau:.3f}\n"
                  f"F1 Score: {f1:.3f}")
    
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(f"Ensemble Prediction: {os.path.basename(video_path)}", fontsize=14, fontweight='bold')
    plt.xlabel("Frame Sequence", fontsize=12)
    plt.ylabel("Importance Score", fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-0.05, 1.1)
    
    save_name = "ensemble_result_plot2.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_name}")

if __name__ == "__main__":
    run_ensemble_vis()