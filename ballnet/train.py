#!/usr/bin/env python3
"""
ballnet/train.py — 训练球心+半径回归 CNN (8:1:1 + early stop)

用法:
    python train.py ../dataset_ball
"""
import sys, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import BallNet
from dataset import BallDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 25
MIN_DELTA = 1e-5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def compute_loss(model, loader, criterion):
    model.eval(); total = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            total += criterion(model(imgs), targets).item() * imgs.size(0)
    return total / len(loader.dataset)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="?", default="../dataset_ball")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    train_ds = BallDataset(args.data_root, train=True)
    eval_ds = BallDataset(args.data_root, train=False)  # val + test share
    n_total = len(eval_ds)
    if len(train_ds) == 0: print("No data!"); return

    # 从 eval_ds 分 val/test
    n_val = int(n_total * 0.5)
    n_test = n_total - n_val
    val_split, test_split = random_split(
        eval_ds, [n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    n_train = len(train_ds)
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_split, batch_size=args.batch, shuffle=False)

    model = BallNet().to(DEVICE)
    criterion = nn.MSELoss()  # [0,1] 绝对值回归, MSE 直接
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=12, factor=0.5)

    out_path = args.out or (Path(args.data_root) / "ballnet.pt")
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training max {MAX_EPOCHS} epochs (early stop patience={EARLY_STOP_PATIENCE})...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        val_loss = compute_loss(model, val_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            test_loss = compute_loss(model, test_loader, criterion)
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | Test: {test_loss:.6f}",
                  flush=True)

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(out_path))
    final_test = compute_loss(model, test_loader, criterion)
    print(f"Done! Best val_loss={best_val_loss:.6f}  Test_loss={final_test:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
