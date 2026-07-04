"""DotDataset — 热图回归数据集 (训练时在线随机增强)"""
import torch, cv2, numpy as np, csv, random
from torch.utils.data import Dataset
from pathlib import Path

IMG_SIZE = 60
GAUSS_SIGMA = 1.5


class DotDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for data_dir in Path(root_dir).glob("data_*"):
            # 优先读增强数据 (aug/), 没有则读原始目录
            aug_dir = data_dir / "aug"
            if aug_dir.exists() and (aug_dir / "labels.csv").exists():
                img_base = aug_dir
                label_file = aug_dir / "labels.csv"
                cols = (0, 4, 5)  # aug_filename, px, py
            elif (data_dir / "labels.csv").exists():
                img_base = data_dir
                label_file = data_dir / "labels.csv"
                cols = (0, 2, 3)  # filename, px, py
            else:
                continue

            img_labels = {}
            with open(label_file, "r") as f:
                reader = csv.reader(f); next(reader, None)
                for row in reader:
                    if len(row) >= max(cols) + 1:
                        fname = row[cols[0]]
                        try: px, py = float(row[cols[1]]), float(row[cols[2]])
                        except ValueError: continue
                        if fname not in img_labels: img_labels[fname] = []
                        img_labels[fname].append((px, py))
            for fname, dots in img_labels.items():
                img_path = img_base / fname
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
