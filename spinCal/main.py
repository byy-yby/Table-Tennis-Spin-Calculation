#!/usr/bin/env python3
"""spinCal 主入口 — AVI 视频 / 图片文件夹 → RPM + 旋转轴"""
import sys, cv2, argparse
from pathlib import Path

from .config import MATCH_MAX_DISP, DEVICE
from .model import load_model
from .detection import process_image
from .ballfit import detect_ball
from .matching import track_dots_across_frames
from .rotation import run_pipeline
from .video import extract_frames_from_video
from .viz import interactive_viewer


def main():
    parser = argparse.ArgumentParser(description="spinCal — Table Tennis Ball Spin Measurement")
    parser.add_argument("input", nargs="?", default="dataset/data_data1",
                        help="AVI video or image folder")
    parser.add_argument("--model", default=None, help="Path to dotnet.pt checkpoint")
    parser.add_argument("--match-disp", type=float, default=MATCH_MAX_DISP,
                        help="3D matching max displacement (default: 0.5)")
    parser.add_argument("--no-viz", action="store_true", help="Skip interactive viewer")
    args = parser.parse_args()

    # 模型路径
    if args.model is None:
        args.model = Path(__file__).parent.parent / "dataset" / "dotnet.pt"
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Specify with --model <path> or place at dataset/dotnet.pt")
        return

    print(f"Loading DotNet: {model_path}")
    model = load_model(model_path, DEVICE)

    # ── 加载数据 ──
    in_path = Path(args.input)
    fps = 628.0
    rough_r = None    # 粗球半径 (60×60 像素), 来自 video.py meta

    if in_path.suffix.lower() == '.avi':
        frames, fps, meta = extract_frames_from_video(str(in_path))
        if frames is None:
            print("Failed to extract frames!"); return
        images = frames
        use_indices = meta.get('frame_indices', list(range(len(frames))))
        # 粗球半径 (裁剪后 60×60 中的像素): median_r * (60 / (2*crop_r))
        _mr = meta.get('median_r'); _cr = meta.get('crop_r')
        if _mr and _cr:
            rough_r = _mr * 60.0 / (2.0 * _cr)
    elif in_path.is_dir():
        images = []
        for p in sorted(in_path.glob("*.png")):
            img = cv2.imread(str(p))
            if img is not None: images.append(img)
        use_indices = list(range(len(images)))
        info_file = in_path / "info.txt"
        if info_file.exists():
            for line in open(info_file):
                if line.startswith("fps="):
                    try: fps = float(line.split("=")[1]); break
                    except: pass
        print(f"Loaded {len(images)} images from folder")
    else:
        print(f"Input not found or unsupported: {args.input}"); return

    if len(images) < 2:
        print("Need >=2 frames!"); return

    # ── Phase A: 子像素球边缘检测 (径向梯度 + RANSAC 圆拟合) ──
    ball_guesses = {}  # {idx: (cx, cy, r)}
    n_refined = 0
    for idx, img in enumerate(images):
        h, w = img.shape[:2]
        rough = (w / 2.0, h / 2.0, rough_r) if rough_r else None
        b = detect_ball(img, rough=rough)
        if b is not None:
            ball_guesses[idx] = b
            n_refined += 1
        else:
            ball_guesses[idx] = (w / 2, h / 2, min(w, h) * 0.33)
    print(f"  Ball edge refined: {n_refined}/{len(images)} frames "
          f"(rest fall back to center)")

    # ── Phase B: 逐帧交互调整 ──
    print("\nAdjust ball per frame: drag center, slider=radius, ENTER=next, F=done")
    win_adj = "Adjust Ball"
    cv2.namedWindow(win_adj, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_adj, 600, 600)
    cv2.createTrackbar("Radius*100", win_adj, 0, 6000, lambda x: None)

    adj_idx = 0; dragging = False
    bcx, bcy, br = ball_guesses[0]
    cv2.setTrackbarPos("Radius*100", win_adj, int(br * 100))

    def adj_cb(event, x, y, flags, param):
        nonlocal bcx, bcy, dragging
        if event == cv2.EVENT_LBUTTONDOWN: dragging = True
        elif event == cv2.EVENT_LBUTTONUP: dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and dragging: bcx, bcy = float(x)/6.0, float(y)/6.0

    cv2.setMouseCallback(win_adj, adj_cb)

    while adj_idx < len(images):
        br = cv2.getTrackbarPos("Radius*100", win_adj) / 100.0
        img = images[adj_idx]
        display = cv2.resize(img, (360, 360), interpolation=cv2.INTER_NEAREST)
        cv2.circle(display, (int(bcx * 6), int(bcy * 6)), int(br * 6), (0, 255, 0), 1)
        cv2.circle(display, (int(bcx * 6), int(bcy * 6)), 1, (0, 0, 255), -1)
        cv2.putText(display, f"[{adj_idx+1}/{len(images)}] cx={bcx:.0f} cy={bcy:.0f} r={br:.0f} "
                    "| ENTER=next B=back F=done",
                    (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imshow(win_adj, display)

        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # ENTER
            ball_guesses[adj_idx] = (bcx, bcy, br)
            adj_idx += 1
            if adj_idx < len(images):
                bcx, bcy, br = ball_guesses[adj_idx]
                cv2.setTrackbarPos("Radius*100", win_adj, int(br * 100))
        elif key == ord('b'):
            ball_guesses[adj_idx] = (bcx, bcy, br)
            if adj_idx > 0: adj_idx -= 1; bcx, bcy, br = ball_guesses[adj_idx]
            cv2.setTrackbarPos("Radius*100", win_adj, int(br * 100))
        elif key == ord('f') or key == ord('F'):
            ball_guesses[adj_idx] = (bcx, bcy, br); break
        elif key == 27 or key == ord('q'): return

    cv2.destroyWindow(win_adj)
    print(f"  Adjusted {len(ball_guesses)} frames")

    # ── Phase C: 逐帧处理 (用调整后的球) ──
    print(f"Processing {len(images)} frames...")
    frames_2d, frames_3d, frames_ball = {}, {}, {}
    for idx, img in enumerate(images):
        if img is None: continue
        dots_2d, dots_3d, ball = process_image(img, model, ball_detector=None,
                                                fixed_ball=ball_guesses[idx])
        if ball is None: continue
        fi = use_indices[idx]
        frames_2d[fi] = dots_2d
        frames_3d[fi] = dots_3d
        frames_ball[fi] = ball

    valid_fi = sorted(frames_3d.keys())
    if len(valid_fi) < 2:
        print("Need >=2 frames with ball+dots!"); return
    print(f"  {len(valid_fi)} valid frames")

    # ── 匹配 ──
    print("Matching dots by 3D proximity...")
    frame_labels, _, _ = track_dots_across_frames(frames_3d, valid_fi, args.match_disp)

    # ── 旋转估计流水线 ──
    print(f"Computing rotation (RANSAC + Kabsch, {fps:.0f} fps)...")
    rpm, rps, axis_user, cw, spin_type, frame_labels, matched_gids, err = \
        run_pipeline(frames_3d, valid_fi, fps)

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Frames: {len(valid_fi)} | Method: RANSAC+ICP")
    print(f"  RPM: {rpm:.1f} | RPS: {rps:.2f} | {cw} | {spin_type}")
    print(f"  Axis: [{axis_user[0]:.4f},{axis_user[1]:.4f},{axis_user[2]:.4f}]")
    print(f"{'='*55}")

    # ── 可视化 ──
    if not args.no_viz:
        print(f"\nViewer: ENTER=next B=back M=3D Q=quit")
        interactive_viewer(images, use_indices, frames_2d, frames_3d, frames_ball,
                           frame_labels, axis_user, rpm, rps, spin_type, cw, valid_fi,
                           matched_gids=matched_gids, fps=fps)


if __name__ == "__main__":
    main()
