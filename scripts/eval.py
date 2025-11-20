import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import f1_score

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Choose your dataset
TARGET_DATASET = "summe"  # or "summe"

DATA_PATHS = {
    "tvsum": "./processed_data_2FPS/tvsum",
    "summe": "./processed_data_2FPS/summe"
}

LLM_DIM = 4096
FOLDS = 5

# ==========================================
# MODEL ARCHITECTURE (Must Match Training)
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
# DATASET
# ==========================================
class VideoSumDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.pt"))) # Sorted for deterministic split
        if len(self.files) == 0: raise ValueError(f"No files in {folder}")
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        return d['features'].float(), d['gt_scores'].float(), os.path.basename(self.files[idx])

# ==========================================
# METRIC CALCULATION
# ==========================================
def get_metrics(preds, gts):
    preds = np.array(preds); gts = np.array(gts)
    
    # 1. Spearman (Rank Correlation)
    rho, _ = spearmanr(preds, gts)
    
    # 2. Kendall (Pairwise Order)
    tau, _ = kendalltau(preds, gts)
    
    # 3. F1 Score (Top 15% length)
    k = max(1, int(len(preds) * 0.15))
    p_bin = np.zeros_like(preds); p_bin[np.argsort(preds)[-k:]] = 1
    g_bin = np.zeros_like(gts); g_bin[np.argsort(gts)[-k:]] = 1
    f1 = f1_score(g_bin, p_bin)
    
    return rho, tau, f1

# ==========================================
# EVALUATION LOOP
# ==========================================
def evaluate_final_score():
    print(f"=== FINAL BENCHMARK EVALUATION: {TARGET_DATASET.upper()} ===")
    
    dataset = VideoSumDataset(DATA_PATHS[TARGET_DATASET])
    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=42) # MUST MATCH TRAINING SEED
    
    all_rhos = []
    all_taus = []
    all_f1s = []
    
    print(f"{'Fold':<5} | {'Rho':<8} | {'Tau':<8} | {'F1':<8} | {'Video Count'}")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        model_path = f"best_{TARGET_DATASET}_f{fold}.pth"
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_path} not found. Skipping fold.")
            continue
            
        # Load Model
        model = GlobalContextNetwork(input_dim=LLM_DIM).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Load Validation Data for this Fold
        val_sub = Subset(dataset, val_idx)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)
        
        fold_rhos = []
        fold_taus = []
        fold_f1s = []
        
        with torch.no_grad():
            for f, g, fname in val_loader:
                f, g = f.to(DEVICE), g.to(DEVICE)
                
                # Inference
                preds = model(f).cpu().numpy().flatten()
                gt = g.cpu().numpy().flatten()
                
                # Compute Metrics for this video
                r, t, f1 = get_metrics(preds, gt)
                
                fold_rhos.append(r)
                fold_taus.append(t)
                fold_f1s.append(f1)
        
        # Average for this Fold
        avg_r = np.nanmean(fold_rhos)
        avg_t = np.nanmean(fold_taus)
        avg_f = np.nanmean(fold_f1s)
        
        print(f"{fold+1:<5} | {avg_r:.4f}   | {avg_t:.4f}   | {avg_f:.4f}   | {len(val_idx)}")
        
        # Add to global list
        all_rhos.extend(fold_rhos)
        all_taus.extend(fold_taus)
        all_f1s.extend(fold_f1s)

    print("-" * 50)
    print("FINAL DATASET AVERAGE (Comparible to Papers):")
    print(f"Spearman Rho: {np.nanmean(all_rhos):.4f}")
    print(f"Kendall Tau:  {np.nanmean(all_taus):.4f}")
    print(f"F1 Score:     {np.nanmean(all_f1s):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_final_score()