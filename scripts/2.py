import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from train2 import VideoSumDataset, GlobalContextNetwork, RankMSELoss, get_metrics, DEVICE, DATA_PATHS

# Force random labels
class RandomLabelDataset(VideoSumDataset):
    def __getitem__(self, idx):
        feats, gt = super().__getitem__(idx)
        # REPLACE GT WITH RANDOM NOISE
        random_gt = torch.rand_like(gt)
        return feats, random_gt

def run_sanity_check():
    print("Running SANITY CHECK: Training on RANDOM LABELS...")
    # Load dataset but FORCE random labels
    dataset = RandomLabelDataset(DATA_PATHS["tvsum"], training=True)
    
    # Just do 1 Fold for speed
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(kfold.split(dataset)))
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)
    
    model = GlobalContextNetwork(input_dim=4096).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = RankMSELoss()
    
    print(f"Train Size: {len(train_idx)} | Val Size: {len(val_idx)}")
    
    for ep in range(30): # Run 30 epochs
        model.train()
        train_losses = []
        for f, g in train_loader:
            f, g = f.to(DEVICE), g.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(f), g)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        rhos = []
        with torch.no_grad():
            for f, g in val_loader:
                f, g = f.to(DEVICE), g.to(DEVICE)
                p = model(f).cpu().numpy().flatten()
                gt = g.cpu().numpy().flatten()
                r, _, _ = get_metrics(p, gt)
                rhos.append(r)
        
        print(f"Ep {ep+1:02d} | Train Loss: {np.mean(train_losses):.4f} | Val Rho: {np.mean(rhos):.4f}")

if __name__ == "__main__":
    run_sanity_check()