"""BallDataset — 图片 → [0,1]归一化靶标 (增强已在 aug/ 中预生成)"""
import torch, cv2, numpy as np, csv
from torch.utils.data import Dataset
from pathlib import Path

IMG_SIZE = 60


class BallDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.samples = []
        for data_dir in Path(root_dir).glob("data_*"):
            if train:
                aug_dir = data_dir / "aug"
                if aug_dir.exists() and (aug_dir / "labels_ball.csv").exists():
                    img_base = aug_dir
                    label_file = aug_dir / "labels_ball.csv"
                else:
                    continue
            else:
                if (data_dir / "labels_ball.csv").exists():
                    img_base = data_dir
                    label_file = data_dir / "labels_ball.csv"
                else:
                    continue

            with open(label_file, "r") as f:
                reader = csv.reader(f); next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        try:
                            fname = row[0]
                            cx, cy, r = float(row[1]), float(row[2]), float(row[3])
                            img_path = img_base / fname
                            if img_path.exists():
                                self.samples.append((str(img_path), cx, cy, r))
                        except ValueError:
                            continue
        tag = "train" if train else "val/test"
        print(f"BallDataset ({tag}): {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cx, cy, r = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.zeros(3)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        target = torch.tensor([
            np.clip(cx / IMG_SIZE, 0.0, 1.0),
            np.clip(cy / IMG_SIZE, 0.0, 1.0),
            np.clip(r / 30.0, 0.0, 1.0),
        ], dtype=torch.float32)
        return img, target

