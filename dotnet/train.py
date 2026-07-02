#!/usr/bin/env python3
"""
dotnet/train.py — 训练黑点检测 CNN

用法:
    python train.py ../dataset
    python train.py ../dataset --epochs 100 --batch 32
"""
import sys, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import DotNet
from dataset import DotDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="?", default="../dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dataset = DotDataset(args.data_root)
    if len(dataset) == 0: print("No data!"); return

    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = DotNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    out_path = args.out or (Path(args.data_root) / "dotnet.pt")
    best_val_loss = float('inf')

    print(f"Training {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for imgs, hmaps in train_loader:
            imgs, hmaps = imgs.to(DEVICE), hmaps.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), hmaps)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for imgs, hmaps in val_loader:
                imgs, hmaps = imgs.to(DEVICE), hmaps.to(DEVICE)
                val_loss += criterion(model(imgs), hmaps).item() * imgs.size(0)
        val_loss /= n_val
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)

    print(f"\nDone! Best val_loss={best_val_loss:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
