"""DotDataset — 热图回归数据集"""
import torch
from torch.utils.data import Dataset
import cv2, numpy as np, csv
from pathlib import Path

IMG_SIZE = 60
GAUSS_SIGMA = 1.5


class DotDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for aug_dir in Path(root_dir).glob("data_*/aug"):
            label_file = aug_dir / "labels.csv"
            if not label_file.exists(): continue
            img_labels = {}
            with open(label_file, "r") as f:
                reader = csv.reader(f); next(reader, None)
                for row in reader:
                    if len(row) >= 6:
                        fname, _, _, _, px, py = (row[0], row[1], row[2], row[3],
                                                   float(row[4]), float(row[5]))
                        if fname not in img_labels: img_labels[fname] = []
                        img_labels[fname].append((px, py))
            for fname, dots in img_labels.items():
                img_path = aug_dir / fname
                if img_path.exists():
                    self.samples.append((str(img_path), dots))
        print(f"DotDataset: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, dots = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        hmap = torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        ys, xs = torch.meshgrid(torch.arange(IMG_SIZE, dtype=torch.float32),
                                torch.arange(IMG_SIZE, dtype=torch.float32), indexing='ij')
        for px, py in dots:
            gauss = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * GAUSS_SIGMA ** 2))
            hmap[0] = torch.maximum(hmap[0], gauss)

        return img, hmap
