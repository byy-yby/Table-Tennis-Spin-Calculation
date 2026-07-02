"""AVI 视频 → 裁剪帧"""
import cv2, numpy as np


def build_background(cap, total):
    samples = []
    for fi in np.linspace(0, total - 1, min(12, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret: samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return np.median(np.array(samples), axis=0)


def extract_frames_from_video(video_path, out_size=60, motion_thresh=2000):
    """从 AVI 提取完整球的裁剪帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, 0, {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Video: {vw}x{vh}, {fps:.0f} fps, {total_frames} frames")

    print("  Scanning for ball window...")
    bg = build_background(cap, total_frames)
    bg_motion = {}
    for fi in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray.astype(np.float32) - bg)
        _, mask = cv2.threshold(diff.astype(np.uint8), 2, 255, cv2.THRESH_BINARY)
        nl, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best = max(((stats[i, cv2.CC_STAT_AREA], centroids[i], stats[i])
                     for i in range(1, nl)), key=lambda x: x[0], default=None)
        if best and best[0] > motion_thresh:
            bcx, bcy = best[1]
            br = max(best[2][cv2.CC_STAT_WIDTH], best[2][cv2.CC_STAT_HEIGHT]) / 2.0
            bg_motion[fi] = (bcx, bcy, br)
        if fi % 400 == 0: print(f"    {fi}/{total_frames}...")

    if not bg_motion: return None, fps, {}

    frames_list = sorted(bg_motion.keys())
    sf, ef, bl = frames_list[0], frames_list[0], 0; cur = frames_list[0]
    for i in range(1, len(frames_list)):
        if frames_list[i] - frames_list[i - 1] > 5:
            if frames_list[i - 1] - cur + 1 > bl:
                bl = frames_list[i - 1] - cur + 1; sf, ef = cur, frames_list[i - 1]
            cur = frames_list[i]
    if frames_list[-1] - cur + 1 > bl: sf, ef = cur, frames_list[-1]

    valid_radii = []
    for fi in range(sf, ef + 1):
        mot = bg_motion.get(fi)
        if mot is None: continue
        cx, cy, r = mot
        if cx - r < 5 or cx + r > vw - 5 or cy - r < 5 or cy + r > vh - 5: continue
        if np.pi * r * r < 2000: continue
        valid_radii.append(r)
    if not valid_radii: return None, fps, {}

    median_r = np.median(valid_radii)
    crop_r = int(median_r * 1.5)

    print(f"  Complete-ball frames: {len(valid_radii)}, r={median_r:.0f}px")

    frames, frame_indices = [], []
    for fi in range(sf, ef + 1):
        mot = bg_motion.get(fi)
        if mot is None: continue
        cx, cy, r = mot
        if cx - r < 5 or cx + r > vw - 5 or cy - r < 5 or cy + r > vh - 5: continue
        if np.pi * r * r < 2000: continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue

        x1 = max(0, int(cx) - crop_r); y1 = max(0, int(cy) - crop_r)
        side = min(min(vw, int(cx) + crop_r) - x1, min(vh, int(cy) + crop_r) - y1)
        if side < 20: continue
        x1 = max(0, int(cx) - side // 2); y1 = max(0, int(cy) - side // 2)
        if x1 + side > vw: x1 = vw - side
        if y1 + side > vh: y1 = vh - side
        x1, y1 = max(0, x1), max(0, y1)
        side = min(side, vw - x1, vh - y1)
        if side < 20: continue

        roi = frame[y1:y1 + side, x1:x1 + side]
        roi = cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        frames.append(roi)
        frame_indices.append(fi)

    cap.release()
    print(f"  Extracted {len(frames)} frames")
    return frames, fps, {'median_r': median_r, 'crop_r': crop_r,
                         'frame_indices': frame_indices}
