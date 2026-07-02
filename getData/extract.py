#!/usr/bin/env python3
"""
getData/extract.py — 从 AVI 视频批量提取完整球的裁剪帧

用法:
    python extract.py D:/highspeed/data              # 处理目录下所有 avi
    python extract.py D:/highspeed/data/data1.avi    # 单个文件
    python extract.py D:/highspeed/data --size 120   # 指定输出尺寸
"""
import cv2, numpy as np, os, sys, argparse, glob

MTX = np.array([[1914.91362, 0., 802.626903],
                 [0., 1906.44992, 542.879886],
                 [0., 0., 1.]], dtype=np.float64)
CALIB_IMAGE_HEIGHT = 1080
MOTION_AREA_THRESH = 2000
BALL_EDGE_MARGIN = 5
MIN_BALL_AREA = 2000


def build_background(cap, total):
    samples = []
    for fi in np.linspace(0, total - 1, min(12, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret: samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return np.median(np.array(samples), axis=0)


def process_video(video_path, out_size, pad, base_out_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(base_out_dir, f"data_{video_name}")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"  SKIP: {video_path}"); return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"  {video_name}: {vw}x{vh}, {fps:.0f}fps, {total_frames}frames")

    # 扫球窗口
    bg = build_background(cap, total_frames)
    bg_motion = {}
    for fi in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray.astype(np.float32) - bg)
        _, mask = cv2.threshold(diff.astype(np.uint8), 2, 255, cv2.THRESH_BINARY)
        nl, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best = max(((stats[i, cv2.CC_STAT_AREA], centroids[i], stats[i])
                     for i in range(1, nl)), key=lambda x: x[0], default=None)
        if best and best[0] > MOTION_AREA_THRESH:
            bcx, bcy = best[1]
            br = max(best[2][cv2.CC_STAT_WIDTH], best[2][cv2.CC_STAT_HEIGHT]) / 2.0
            bg_motion[fi] = (bcx, bcy, br)

    if not bg_motion: print(f"  SKIP: no motion"); cap.release(); return 0

    frames = sorted(bg_motion.keys())
    sf, ef, bl = frames[0], frames[0], 0; cur = frames[0]
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > 5:
            if frames[i - 1] - cur + 1 > bl:
                bl = frames[i - 1] - cur + 1; sf, ef = cur, frames[i - 1]
            cur = frames[i]
    if frames[-1] - cur + 1 > bl: sf, ef = cur, frames[-1]

    valid_radii = []
    for fi in range(sf, ef + 1):
        mot = bg_motion.get(fi)
        if mot is None: continue
        cx, cy, r = mot
        if cx - r < BALL_EDGE_MARGIN or cx + r > vw - BALL_EDGE_MARGIN: continue
        if cy - r < BALL_EDGE_MARGIN or cy + r > vh - BALL_EDGE_MARGIN: continue
        if np.pi * r * r < MIN_BALL_AREA: continue
        valid_radii.append(r)
    if not valid_radii: print(f"  SKIP: no complete-ball frames"); cap.release(); return 0

    median_r = np.median(valid_radii); crop_r = int(median_r * pad)

    saved = 0
    for fi in range(sf, ef + 1):
        mot = bg_motion.get(fi)
        if mot is None: continue
        cx, cy, r = mot
        if cx - r < BALL_EDGE_MARGIN or cx + r > vw - BALL_EDGE_MARGIN: continue
        if cy - r < BALL_EDGE_MARGIN or cy + r > vh - BALL_EDGE_MARGIN: continue
        if np.pi * r * r < MIN_BALL_AREA: continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ret, frame = cap.read()
        if not ret: continue

        x1, y1 = max(0, int(cx) - crop_r), max(0, int(cy) - crop_r)
        side = min(min(vw, int(cx) + crop_r) - x1, min(vh, int(cy) + crop_r) - y1)
        if side < 20: continue
        x1, y1 = max(0, int(cx) - side // 2), max(0, int(cy) - side // 2)
        if x1 + side > vw: x1 = vw - side
        if y1 + side > vh: y1 = vh - side
        side = min(side, vw - max(0, x1), vh - max(0, y1))
        if side < 20: continue

        roi = frame[y1:y1 + side, x1:x1 + side]
        roi = cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_dir, f"{fi:06d}.png"), roi)
        saved += 1

    cap.release()
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"video={video_path}\nfps={fps}\nmedian_r_px={median_r:.1f}\n"
                f"crop_half_px={crop_r}\noutput_size={out_size}\nframes={saved}\n")
    print(f"    -> {saved} frames -> {out_dir}/")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Batch extract ball ROIs from AVI videos")
    parser.add_argument("path", help="AVI file, or directory of AVIs")
    parser.add_argument("--size", type=int, default=60, help="Output size (default: 60)")
    parser.add_argument("--pad", type=float, default=1.5, help="Padding factor (default: 1.5)")
    parser.add_argument("--out", default="dataset", help="Output root (default: ./dataset)")
    args = parser.parse_args()

    videos = []
    if os.path.isdir(args.path):
        videos = sorted(glob.glob(os.path.join(args.path, "*.avi")))
    elif os.path.isfile(args.path):
        videos = [args.path]
    else:
        print(f"No videos: {args.path}"); return

    print(f"Processing {len(videos)} video(s) -> {args.out}/")
    total = sum(process_video(v, args.size, args.pad, args.out) for v in videos)
    print(f"\nDone! {total} frames total -> {args.out}/")


if __name__ == "__main__":
    main()
