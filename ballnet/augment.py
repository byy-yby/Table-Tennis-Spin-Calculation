#!/usr/bin/env python3
"""
ballnet/augment.py — 数据增强 (每图必平移, 其他随机叠加)

用法:
    python augment.py dataset_ball
    python augment.py dataset_ball/data_data1
"""
import cv2, numpy as np, os, sys, csv, random
from pathlib import Path


def augment_one(img, cx, cy, r, aug_type, img_size):
    w, h = img_size
    a_img, acx, acy, ar = img.copy(), cx, cy, r

    if aug_type == "translate":
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        a_img = cv2.warpAffine(img, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        acx += dx; acy += dy
    elif aug_type == "hflip":
        a_img = cv2.flip(img, 1); acx = w - 1 - cx
    elif aug_type == "vflip":
        a_img = cv2.flip(img, 0); acy = h - 1 - cy
    elif aug_type == "rot90":
        a_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        acx, acy, ar = cy, w - 1 - cx, r
    elif aug_type == "rot180":
        a_img = cv2.rotate(img, cv2.ROTATE_180)
        acx, acy = w - 1 - cx, h - 1 - cy
    elif aug_type == "rot270":
        a_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        acx, acy, ar = h - 1 - cy, cx, r
    elif aug_type == "bright_up":
        d = random.randint(20, 50)
        a_img = np.clip(img.astype(np.int16)+d, 0, 255).astype(np.uint8)
    elif aug_type == "bright_down":
        d = random.randint(20, 50)
        a_img = np.clip(img.astype(np.int16)-d, 0, 255).astype(np.uint8)
    elif aug_type == "contrast":
        f = random.uniform(0.6, 1.4)
        m = np.mean(img)
        a_img = np.clip((img.astype(np.float32)-m)*f+m, 0, 255).astype(np.uint8)
    elif aug_type == "noise":
        a_img = np.clip(img.astype(np.int16)+np.random.randint(-15,15,img.shape,dtype=np.int16),
                        0, 255).astype(np.uint8)
    elif aug_type == "blur":
        a_img = cv2.GaussianBlur(img, (random.choice([3,5]), random.choice([3,5])), 0)

    if a_img.shape[1] != w or a_img.shape[0] != h:
        a_img = cv2.resize(a_img, (w, h))
    return a_img, acx, acy, ar


def process_dir(img_dir):
    label_file = img_dir / "labels_ball.csv"
    if not label_file.exists(): return 0

    labels = {}
    with open(label_file) as f:
        for row in csv.reader(f):
            if row[0] == "filename": continue
            if len(row) >= 4:
                labels[row[0]] = (float(row[1]), float(row[2]), float(row[3]))

    if not labels: return 0
    aug_dir = img_dir / "aug"; aug_dir.mkdir(exist_ok=True)
    other_types = ["hflip", "vflip", "rot90", "rot180", "rot270",
                   "bright_up", "bright_down", "contrast", "noise", "blur"]
    all_rows = []

    for fname, (cx, cy, r) in labels.items():
        img = cv2.imread(str(img_dir / fname))
        if img is None: continue
        h, w = img.shape[:2]

        aug_id = 0
        # 原图 + 必平移
        a_img, acx, acy, ar = augment_one(img, cx, cy, r, "translate", (w, h))
        cv2.imwrite(str(aug_dir / f"{Path(fname).stem}_{aug_id:03d}.png"), a_img)
        all_rows.append([f"{Path(fname).stem}_{aug_id:03d}.png",
                         f"{acx:.2f}", f"{acy:.2f}", f"{ar:.2f}"])

        # 15 个变体: 先平移, 再随机叠加 0-3 个其他增强
        for _ in range(15):
            aug_id += 1
            a_img, acx, acy, ar = augment_one(img, cx, cy, r, "translate", (w, h))
            for t in random.sample(other_types, random.randint(0, 3)):
                a_img, acx, acy, ar = augment_one(a_img, acx, acy, ar, t, (w, h))
            cv2.imwrite(str(aug_dir / f"{Path(fname).stem}_{aug_id:03d}.png"), a_img)
            all_rows.append([f"{Path(fname).stem}_{aug_id:03d}.png",
                             f"{acx:.2f}", f"{acy:.2f}", f"{ar:.2f}"])

    with open(aug_dir / "labels_ball.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "cx_px", "cy_px", "r_px"])
        for row in all_rows: w.writerow(row)

    print(f"  {img_dir.name}: {len(labels)} -> {len(all_rows)} augmented")
    return len(all_rows)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset_ball")
    if root.is_dir() and root.name.startswith("data_") and (root / "labels_ball.csv").exists():
        dirs = [root]
    elif len(sys.argv) > 2:
        dirs = [d for d in root.iterdir() if d.is_dir() and sys.argv[2] in d.name]
    else:
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and d.name.startswith("data_")
                      and (d / "labels_ball.csv").exists())
    if not dirs: print("No data dirs"); return
    total = sum(process_dir(d) for d in dirs)
    print(f"\nDone! {total} total augmented")


if __name__ == "__main__":
    main()
