#!/usr/bin/env python3
"""
ballnet/label.py — 交互式标定球心与半径

操作:
  鼠标左键拖拽  = 移动圆心
  滑动条         = 调整半径
  ENTER          = 保存并下一帧
  B              = 退回上一帧
  S              = 跳过
  Q / ESC        = 退出

用法:
  python label.py dataset_ball/data_data1
  python label.py dataset_ball               # 标注所有 data_* 目录
"""
import cv2, numpy as np, os, sys, csv
from pathlib import Path


def _save_csv(label_file, all_labels):
    with open(label_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cx_px", "cy_px", "r_px"])
        for fn in sorted(all_labels.keys()):
            cx, cy, r = all_labels[fn]
            writer.writerow([fn, f"{cx:.2f}", f"{cy:.2f}", f"{r:.2f}"])


def label_one_dir(img_dir):
    images = sorted(img_dir.glob("*.png"))
    if not images: return -1

    print(f"\n{'='*50}\n  {img_dir.name}: {len(images)} images\n{'='*50}")

    label_file = img_dir / "labels_ball.csv"
    all_labels = {}
    if label_file.exists():
        with open(label_file, "r") as f:
            reader = csv.reader(f); next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    fn, cx, cy, r = row[0], float(row[1]), float(row[2]), float(row[3])
                    all_labels[fn] = (cx, cy, r)
        print(f"  Loaded {len(all_labels)} existing labels")

    win = f"BallNet Label — {img_dir.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 600, 600)
    cv2.createTrackbar("Radius*100", win, 0, 6000, lambda x: None)

    idx = 0; dragging = False
    while idx < len(images):
        p = images[idx]; fname = p.name
        img = cv2.imread(str(p))
        if img is None: idx += 1; continue
        h, w = img.shape[:2]

        # 加载或默认
        if fname in all_labels:
            cx, cy, r = all_labels[fname]
        else:
            cx, cy, r = w / 2.0, h / 2.0, min(w, h) / 2.0 * 0.85
        cv2.setTrackbarPos("Radius*100", win, int(r * 100))

        def mouse_cb(event, x, y, flags, param):
            nonlocal cx, cy, dragging
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                cx, cy = float(x), float(y)

        cv2.setMouseCallback(win, mouse_cb)

        while True:
            r = cv2.getTrackbarPos("Radius*100", win) / 100.0
            display = img.copy()
            cv2.circle(display, (int(cx), int(cy)), int(r), (0, 255, 0), 1)
            cv2.circle(display, (int(cx), int(cy)), 1, (0, 0, 255), -1)

            n_labeled = len(all_labels)
            cv2.putText(display,
                        f"[{idx+1}/{len(images)}] {fname} | "
                        f"cx={cx:.0f} cy={cy:.0f} r={r:.0f} | "
                        f"Labeled: {n_labeled} | ENTER=next B=back S=skip Q=quit",
                        (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.imshow(win, display)

            key = cv2.waitKey(30) & 0xFF
            if key == 13:  # ENTER
                all_labels[fname] = (cx, cy, r)
                _save_csv(label_file, all_labels)
                idx += 1; break
            elif key == ord('b'):  # back
                all_labels[fname] = (cx, cy, r)
                _save_csv(label_file, all_labels)
                if idx > 0: idx -= 1; break
            elif key == ord('s'): idx += 1; break
            elif key == 27 or key == ord('q'):
                all_labels[fname] = (cx, cy, r)
                _save_csv(label_file, all_labels)
                idx = len(images); break
            elif ord('0') <= key <= ord('9'):
                all_labels[fname] = (cx, cy, r)
                _save_csv(label_file, all_labels)
                return key - ord('0')

    cv2.destroyWindow(win)
    _save_csv(label_file, all_labels)
    print(f"  Saved: {len(all_labels)} frames")
    return -1


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset_ball")

    # 如果 root 本身就是 data_* 目录
    if root.is_dir() and root.name.startswith("data_") and list(root.glob("*.png")):
        dirs = [root]
    else:
        dirs = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("data_") and list(d.glob("*.png"))])

    if not dirs:
        print("No data_* directories with PNG images found")
        return

    # 目录选择
    start_idx = 0
    if len(sys.argv) > 2:
        try: start_idx = int(sys.argv[2])
        except ValueError:
            for i, d in enumerate(dirs):
                if sys.argv[2] in d.name: start_idx = i; break

    print(f"Found {len(dirs)} directories:")
    for i, d in enumerate(dirs):
        n = len(list(d.glob("*.png")))
        has = " [has labels]" if (d / "labels_ball.csv").exists() else ""
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
