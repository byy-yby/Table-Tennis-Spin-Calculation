#!/usr/bin/env python3
"""
getData/augment.py — 数据增强 (保持点坐标同步)

用法:
    python augment.py dataset              # 处理所有 data_*
    python augment.py dataset data_data6   # 只处理指定目录
"""
import cv2, numpy as np, os, sys, csv, random
from pathlib import Path


def augment_image(img, dots, aug_type, img_size):
    w, h = img_size
    aug_img = img.copy()
    aug_dots = [(px, py) for px, py in dots]

    if aug_type == "hflip":
        aug_img = cv2.flip(img, 1)
        aug_dots = [(w - 1 - px, py) for px, py in dots]
    elif aug_type == "vflip":
        aug_img = cv2.flip(img, 0)
        aug_dots = [(px, h - 1 - py) for px, py in dots]
    elif aug_type == "rot90":
        aug_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        aug_dots = [(py, w - 1 - px) for px, py in dots]
    elif aug_type == "rot180":
        aug_img = cv2.rotate(img, cv2.ROTATE_180)
        aug_dots = [(w - 1 - px, h - 1 - py) for px, py in dots]
    elif aug_type == "rot270":
        aug_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        aug_dots = [(h - 1 - py, px) for px, py in dots]
    elif aug_type == "bright_up":
        delta = random.randint(20, 50)
        aug_img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    elif aug_type == "bright_down":
        delta = random.randint(20, 50)
        aug_img = np.clip(img.astype(np.int16) - delta, 0, 255).astype(np.uint8)
    elif aug_type == "contrast":
        factor = random.uniform(0.6, 1.4)
        mean = np.mean(img)
        aug_img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    elif aug_type == "noise":
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        aug_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif aug_type == "blur":
        k = random.choice([3, 5])
        aug_img = cv2.GaussianBlur(img, (k, k), 0)

    if aug_img.shape[1] != w or aug_img.shape[0] != h:
        aug_img = cv2.resize(aug_img, (w, h))
    return aug_img, aug_dots


def process_dir(img_dir):
    label_file = img_dir / "labels.csv"
    if not label_file.exists(): return 0

    labels = {}
    with open(label_file, "r") as f:
        reader = csv.reader(f); next(reader, None)
        for row in reader:
            if len(row) >= 4:
                fname, _, px, py = row[0], int(row[1]), float(row[2]), float(row[3])
                if fname not in labels: labels[fname] = []
                labels[fname].append((px, py))

    n_frames = len(labels)
    if n_frames == 0: return 0

    aug_dir = img_dir / "aug"; aug_dir.mkdir(exist_ok=True)
    aug_types = ["hflip", "vflip", "rot90", "rot180", "rot270",
                 "bright_up", "bright_down", "contrast", "noise", "blur"]
    all_rows = []

    for fname, dots in labels.items():
        img = cv2.imread(str(img_dir / fname))
        if img is None: continue
        h, w = img.shape[:2]

        aug_id = 0
        out_name = f"{Path(fname).stem}_{aug_id:03d}.png"
        cv2.imwrite(str(aug_dir / out_name), img)
        for px, py in dots:
            all_rows.append([out_name, fname, aug_id, "original", f"{px:.2f}", f"{py:.2f}"])

        n_aug = random.randint(5, len(aug_types))
        for aug_type in random.sample(aug_types, n_aug):
            aug_id += 1
            out_name = f"{Path(fname).stem}_{aug_id:03d}.png"
            aug_img, aug_dots = augment_image(img, dots, aug_type, (w, h))
            cv2.imwrite(str(aug_dir / out_name), aug_img)
            for px, py in aug_dots:
                all_rows.append([out_name, fname, aug_id, aug_type, f"{px:.2f}", f"{py:.2f}"])

    with open(aug_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["aug_filename", "source", "aug_id", "aug_type", "px", "py"])
        for row in all_rows: writer.writerow(row)

    n_aug = len(set(r[0] for r in all_rows))
    print(f"  {img_dir.name}: {n_frames} -> {n_aug} augmented")
    return n_aug


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    if len(sys.argv) > 2:
        dirs = [d for d in root.iterdir() if d.is_dir() and sys.argv[2] in d.name]
    else:
        dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("data_")])

    if not dirs: print("No data directories"); return
    print(f"Augmenting {len(dirs)} directories:\n")
    total = sum(process_dir(d) for d in dirs)
    print(f"\nDone! {total} augmented images total -> dataset/data_*/aug/")


if __name__ == "__main__":
    main()
