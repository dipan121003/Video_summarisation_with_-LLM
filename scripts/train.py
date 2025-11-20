import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import f1_score

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DATASET SELECTION
# Change this to "tvsum" or "summe" to train specifically for that dataset
TARGET_DATASET = "tvsum" 

DATA_PATHS = {
    "tvsum": "./processed_data/tvsum",
    "summe": "./processed_data/summe"
}

LLM_DIM = 4096 
BATCH_SIZE = 1
EPOCHS = 100 # Increased epochs because Scheduler will manage overfitting
LR = 5e-5    # Slightly higher start, scheduler will lower it
FOLDS = 5

# ==========================================
# DATASET
# ==========================================
class VideoSumDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No files found in {folder}")
        print(f"Loaded {len(self.files)} samples from {folder}")
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        # Features: [Seq, 4096], GT: [Seq]
        return d['features'].float(), d['gt_scores'].float()

# ==========================================
# MODEL (CNN + Transformer)
# ==========================================
class GlobalContextNetwork(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=256):
        super().__init__()
        
        # 1. Local Pattern Detection (Smoothes the noisy Llama spikes)
        self.local_net = nn.Sequential(
            nn.Conv1d(input_dim, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(1024, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )
        
        # 2. Positional Encoding (Crucial for sequence order)
        self.pos_emb = nn.Parameter(torch.randn(1, 5000, hidden_dim) * 0.01)
        
        # 3. Global Context (Understanding Start vs End)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=512, 
            dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        
        # 4. Regressor
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [Batch, Seq, 4096]
        
        # CNN Expects [Batch, Channels, Seq]
        x = x.permute(0, 2, 1)
        x = self.local_net(x)
        
        # Transformer Expects [Batch, Seq, Channels]
        x = x.permute(0, 2, 1)
        seq_len = x.size(1)
        
        # Add Positional Embeddings
        x = x + self.pos_emb[:, :seq_len, :]
        
        # Run Transformer
        x = self.transformer(x)
        
        # Predict
        return self.head(x).squeeze(-1)

# ==========================================
# UTILS
# ==========================================
def get_metrics(preds, gts):
    preds = np.array(preds); gts = np.array(gts)
    if len(preds) < 2: return 0,0,0
    
    # Correlation
    rho, _ = spearmanr(preds, gts)
    tau, _ = kendalltau(preds, gts)
    
    # F1 (Top 15% Knapsack Proxy)
    k = max(1, int(len(preds) * 0.15))
    p_bin = np.zeros_like(preds); p_bin[np.argsort(preds)[-k:]] = 1
    g_bin = np.zeros_like(gts); g_bin[np.argsort(gts)[-k:]] = 1
    f1 = f1_score(g_bin, p_bin)
    
    return rho, tau, f1

def train_pipeline():
    print(f"Starting Training for Target: {TARGET_DATASET.upper()}")
    dataset = VideoSumDataset(DATA_PATHS[TARGET_DATASET])
    
    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    final_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*15} FOLD {fold+1}/{FOLDS} {'='*15}")
        
        # Setup Data
        tr_sub = Subset(dataset, tr_idx)
        val_sub = Subset(dataset, val_idx)
        tr_load = DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_load = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        # Setup Model
        model = GlobalContextNetwork(input_dim=LLM_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        criterion = nn.MSELoss()
        
        best_composite = -1.0
        
        for ep in range(EPOCHS):
            # --- TRAIN ---
            model.train()
            ep_loss = 0
            for f, g in tr_load:
                f, g = f.to(DEVICE), g.to(DEVICE)
                optimizer.zero_grad()
                
                preds = model(f)
                loss = criterion(preds, g)
                
                loss.backward()
                # STABILITY FIX: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                ep_loss += loss.item()
                
            # --- VALIDATE ---
            model.eval()
            rhos, taus, f1s = [], [], []
            with torch.no_grad():
                for f, g in val_load:
                    f, g = f.to(DEVICE), g.to(DEVICE)
                    p = model(f).cpu().numpy().flatten()
                    gt = g.cpu().numpy().flatten()
                    
                    r, t, f1 = get_metrics(p, gt)
                    rhos.append(r); taus.append(t); f1s.append(f1)
            
            avg_rho = np.nanmean(rhos)
            avg_tau = np.nanmean(taus)
            avg_f1 = np.nanmean(f1s)
            composite = (avg_rho + avg_tau + avg_f1) / 3
            
            # Step Scheduler
            scheduler.step(composite)
            curr_lr = optimizer.param_groups[0]['lr']
            
            print(f"Ep {ep+1:03d} | Loss: {ep_loss/len(tr_load):.4f} | "
                  f"Rho: {avg_rho:.3f} Tau: {avg_tau:.3f} F1: {avg_f1:.3f} | LR: {curr_lr:.2e}")
            
            # Save Best
            if composite > best_composite:
                best_composite = composite
                torch.save(model.state_dict(), f"best_model_{TARGET_DATASET}_fold{fold}.pth")
                print(f"   --> New Best ({composite:.4f})")
        
        final_scores.append(best_composite)

    print("\n" + "="*30)
    print(f"FINAL RESULTS FOR {TARGET_DATASET.upper()}")
    print(f"Avg Composite Score: {np.mean(final_scores):.4f}")
    print(f"Detailed Scores: {final_scores}")
    print("="*30)

if __name__ == "__main__":
    train_pipeline()