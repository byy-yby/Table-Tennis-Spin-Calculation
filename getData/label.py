#!/usr/bin/env python3
"""
getData/label.py — 手动标定球上黑点

用法:
    python label.py dataset              # 标注所有 data_* 目录
    python label.py dataset data_data6   # 指定目录
    python label.py dataset 5            # 从第5个目录开始
"""
import cv2, os, sys, csv
from pathlib import Path


def _save_csv(label_file, all_labels):
    with open(label_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "dot_id", "px", "py"])
        for fn in sorted(all_labels.keys()):
            for dot_id, (px, py) in enumerate(all_labels[fn]):
                writer.writerow([fn, dot_id, f"{px:.2f}", f"{py:.2f}"])


def label_one_dir(img_dir):
    images = sorted(img_dir.glob("*.png"))
    if not images: return -1

    print(f"\n{'='*50}\n  {img_dir.name}: {len(images)} images\n{'='*50}")

    label_file = img_dir / "labels.csv"
    all_labels = {}
    if label_file.exists():
        with open(label_file, "r") as f:
            reader = csv.reader(f); next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    fname, _, px, py = row[0], int(row[1]), float(row[2]), float(row[3])
                    if fname not in all_labels: all_labels[fname] = []
                    all_labels[fname].append((px, py))
        print(f"  Loaded {sum(1 for v in all_labels.values() if v)} labeled frames")

    win = f"Label — {img_dir.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 500, 500)

    idx = 0
    while idx < len(images):
        p = images[idx]; fname = p.name
        img = cv2.imread(str(p))
        if img is None: idx += 1; continue
        dots = list(all_labels.get(fname, []))

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                dots.append((float(x), float(y)))
                print(f"  {fname}: {len(dots)} dots")
            elif event == cv2.EVENT_RBUTTONDOWN and dots:
                dots.pop()
                print(f"  {fname}: undo, {len(dots)} left")

        cv2.setMouseCallback(win, mouse_cb)

        while True:
            display = img.copy()
            for dx, dy in dots:
                cv2.circle(display, (int(dx), int(dy)), 1, (0, 0, 255), -1)

            n_labeled = sum(1 for v in all_labels.values() if v)
            cv2.putText(display,
                        f"[{idx+1}/{len(images)}] {fname} | Dots:{len(dots)} | "
                        f"Labeled:{n_labeled} | ENTER=next B=back S=skip Q=quit",
                        (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.imshow(win, display)

            key = cv2.waitKey(30) & 0xFF
            if key == 13:  # ENTER
                all_labels[fname] = list(dots); _save_csv(label_file, all_labels)
                idx += 1; break
            elif key == ord('b'):  # back
                all_labels[fname] = list(dots); _save_csv(label_file, all_labels)
                if idx > 0: idx -= 1; break
            elif key == ord('s'): idx += 1; break
            elif key == 27 or key == ord('q'):
                all_labels[fname] = list(dots); _save_csv(label_file, all_labels)
                idx = len(images); break
            elif ord('0') <= key <= ord('9'):
                all_labels[fname] = list(dots); _save_csv(label_file, all_labels)
                return key - ord('0')

    cv2.destroyWindow(win)
    _save_csv(label_file, all_labels)
    print(f"  Saved: {sum(1 for v in all_labels.values() if v)} frames")
    return -1


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")

    # 确定起始目录
    start_idx = 0
    if len(sys.argv) > 2:
        try: start_idx = int(sys.argv[2])
        except ValueError: pass

    if not root.exists(): print(f"Not found: {root}"); return

    dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("data_")])
    if not dirs and root.name.startswith("data_"): dirs = [root]
    if not dirs: print("No data_* directories found"); return

    # 如果传了名字, 直接匹配
    if len(sys.argv) > 2 and start_idx == 0:
        for i, d in enumerate(dirs):
            if sys.argv[2] in d.name: start_idx = i; break

    print(f"Found {len(dirs)} directories:")
    for i, d in enumerate(dirs):
        n = len(list(d.glob("*.png")))
        has = " [labeled]" if (d / "labels.csv").exists() else ""
        print(f"  [{i}] {d.name} ({n} images){has}")

    cur = min(start_idx, len(dirs) - 1)
    while cur < len(dirs):
        result = label_one_dir(dirs[cur])
        if result == -1: cur += 1
        elif 0 <= result < len(dirs): cur = result
        else: cur += 1

    print("\nAll done!")


if __name__ == "__main__":
    main()
