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

TARGET_DATASET = "summe"  # Change to "summe" if needed
DATA_PATHS = {
    "tvsum": "./processed_data_2FPS/tvsum",
    "summe": "./processed_data_2FPS/summe"
}

LLM_DIM = 4096 
BATCH_SIZE = 1
EPOCHS = 100      
LR = 1e-4         
FOLDS = 5

# ==========================================
# 1. AUGMENTED DATASET
# ==========================================
class VideoSumDataset(Dataset):
    def __init__(self, folder, training=False):
        self.files = glob.glob(os.path.join(folder, "*.pt"))
        self.training = training
        if len(self.files) == 0: raise ValueError(f"No files in {folder}")
        print(f"Loaded {len(self.files)} samples ({'Train' if training else 'Val'})")
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        feats = d['features'].float()
        gt = d['gt_scores'].float()
        
        if self.training:
            # AUGMENTATION 1: Feature Noise (Reduced to 0.005 for Llama stability)
            noise = torch.randn_like(feats) * 0.005
            feats = feats + noise
            
            # AUGMENTATION 2: Random Subsampling
            seq_len = feats.shape[0]
            if seq_len > 50: 
                # Keep at least 85% of video
                crop_size = int(seq_len * np.random.uniform(0.85, 1.0))
                start = np.random.randint(0, seq_len - crop_size + 1)
                feats = feats[start : start+crop_size]
                gt = gt[start : start+crop_size]

        return feats, gt

# ==========================================
# 2. CORRECTED RANKING LOSS
# ==========================================
class RankMSELoss(nn.Module):
    def __init__(self, alpha=0.5, margin=0.15):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, preds, target):
        # MSE Component
        loss_mse = self.mse(preds, target)
        
        # Pairwise Ranking Component
        n = preds.shape[0]
        # Subsample if video is too long to save memory
        if n > 80:
            indices = torch.randperm(n)[:80]
            p_sub = preds[indices]
            t_sub = target[indices]
        else:
            p_sub = preds
            t_sub = target

        # Broadcasting for pairs
        # shape: [N, N]
        pred_diff = p_sub.unsqueeze(1) - p_sub.unsqueeze(0)
        target_diff = t_sub.unsqueeze(1) - t_sub.unsqueeze(0)
        
        # Create Indicator: 1 if target_i > target_j, -1 if <, 0 if ==
        target_signs = torch.sign(target_diff)
        
        # Mask: Only care if scores are DIFFERENT (ignore ties)
        mask = (target_signs != 0).float()
        
        # Ranking Hinge Loss
        # We want pred_diff to have same sign as target_diff
        # Loss = ReLU(margin - target_sign * pred_diff)
        hinge_loss = torch.relu(self.margin - target_signs * pred_diff)
        
        # Apply mask so we don't penalize ties
        loss_rank = (hinge_loss * mask).sum() / (mask.sum() + 1e-6)
        
        return (1 - self.alpha) * loss_mse + self.alpha * loss_rank

# ==========================================
# 3. ARCHITECTURE (SOTA Multi-Scale)
# ==========================================
class GlobalContextNetwork(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=256):
        super().__init__()
        
        # Multi-Scale Convs
        self.conv_short = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv_long = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=5, padding=2)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5) # Increased dropout
        
        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.randn(1, 5000, hidden_dim) * 0.01)
        
        # Transformer
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=1024, 
            dropout=0.3, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        
        # Regressor
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2), # Extra dropout in head
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # -> [B, C, L]
        
        # Parallel Branches
        x1 = self.conv_short(x)
        x2 = self.conv_long(x)
        x = torch.cat([x1, x2], dim=1) 
        
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Transformer
        x = x.permute(0, 2, 1) # -> [B, L, C]
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        x = self.transformer(x)
        
        return self.head(x).squeeze(-1)

# ==========================================
# UTILS & TRAINING
# ==========================================
def get_metrics(preds, gts):
    preds = np.array(preds); gts = np.array(gts)
    if len(preds) < 2: return 0,0,0
    
    rho, _ = spearmanr(preds, gts)
    tau, _ = kendalltau(preds, gts)
    
    k = max(1, int(len(preds) * 0.15))
    p_bin = np.zeros_like(preds); p_bin[np.argsort(preds)[-k:]] = 1
    g_bin = np.zeros_like(gts); g_bin[np.argsort(gts)[-k:]] = 1
    f1 = f1_score(g_bin, p_bin)
    
    return rho, tau, f1

def train_pipeline():
    print(f"Starting SOTA Training for: {TARGET_DATASET.upper()}")
    
    # 1. LOAD DATA
    full_train_set = VideoSumDataset(DATA_PATHS[TARGET_DATASET], training=True)
    full_val_set = VideoSumDataset(DATA_PATHS[TARGET_DATASET], training=False)
    
    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    final_results = []
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(full_train_set)):
        print(f"\n{'='*10} FOLD {fold+1}/{FOLDS} {'='*10}")
        
        tr_sub = Subset(full_train_set, tr_idx)
        val_sub = Subset(full_val_set, val_idx)
        
        tr_load = DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_load = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        # 2. SETUP MODEL
        model = GlobalContextNetwork(input_dim=LLM_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        criterion = RankMSELoss(alpha=0.5, margin=0.15)
        
        best_score = -1.0
        
        # 3. EPOCH LOOP
        for ep in range(EPOCHS):
            model.train()
            losses = []
            for f, g in tr_load:
                f, g = f.to(DEVICE), g.to(DEVICE)
                optimizer.zero_grad()
                
                preds = model(f)
                loss = criterion(preds, g)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            
            scheduler.step()
            
            # 4. VALIDATION
            model.eval()
            rhos, taus, f1s = [], [], []
            with torch.no_grad():
                for f, g in val_load:
                    f, g = f.to(DEVICE), g.to(DEVICE)
                    p = model(f).cpu().numpy().flatten()
                    gt = g.cpu().numpy().flatten()
                    r, t, f1 = get_metrics(p, gt)
                    rhos.append(r); taus.append(t); f1s.append(f1)
            
            score = (np.nanmean(rhos) + np.nanmean(taus) + np.nanmean(f1s)) / 3
            
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), f"best_{TARGET_DATASET}_f{fold}.pth")
                # print(f"Ep {ep+1:03d} | Loss {np.mean(losses):.4f} | Best Score: {score:.4f}")
            
            # Minimal logging to keep console clean
            if (ep+1) % 20 == 0:
                print(f"Ep {ep+1:03d} | Loss {np.mean(losses):.4f} | Val Score: {score:.4f}")
                    
        final_results.append(best_score)
        print(f"Fold {fold+1} Best: {best_score:.4f}")

    print(f"\nFinal Average Score: {np.mean(final_results):.4f}")

if __name__ == "__main__":
    train_pipeline()