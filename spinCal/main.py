#!/usr/bin/env python3
"""spinCal 主入口 — AVI 视频 / 图片文件夹 → RPM + 旋转轴"""
import sys, cv2, argparse
from pathlib import Path

from .config import MATCH_MAX_DISP, DEVICE
from .model import load_model
from .detection import process_image
from .matching import track_dots_across_frames
from .rotation import compute_rotation
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

    print(f"Loading model: {model_path}")
    model = load_model(model_path, DEVICE)

    # ── 加载数据 ──
    in_path = Path(args.input)
    fps = 628.0

    if in_path.suffix.lower() == '.avi':
        frames, fps, meta = extract_frames_from_video(str(in_path))
        if frames is None:
            print("Failed to extract frames!"); return
        images = frames
        use_indices = meta.get('frame_indices', list(range(len(frames))))
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

    # ── 逐帧处理 ──
    print(f"Processing {len(images)} frames...")
    frames_2d, frames_3d, frames_ball = {}, {}, {}
    for idx, img in enumerate(images):
        if img is None: continue
        dots_2d, dots_3d, ball = process_image(img, model, DEVICE)
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
    frame_labels, next_gid = track_dots_across_frames(frames_3d, valid_fi, args.match_disp)
    n_multi = sum(1 for gid in range(next_gid)
                  if sum(1 for fi in valid_fi if gid in frame_labels[fi].values()) >= 2)
    print(f"  {next_gid} total IDs, {n_multi} multi-frame")

    # ── 旋转 ──
    print(f"Computing rotation (3-frame windows, {fps:.0f} fps)...")
    rpm, rps, axis_user, cw, spin_type = compute_rotation(
        frames_3d, frame_labels, valid_fi, fps)

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Frames: {len(valid_fi)}")
    print(f"  RPM: {rpm:.1f} | RPS: {rps:.2f} | {cw} | {spin_type}")
    print(f"  Axis: [{axis_user[0]:.4f},{axis_user[1]:.4f},{axis_user[2]:.4f}]")
    print(f"{'='*55}")

    # ── 可视化 ──
    if not args.no_viz:
        print(f"\nViewer: ENTER=next B=back M=3D Q=quit")
        interactive_viewer(images, use_indices, frames_2d, frames_3d, frames_ball,
                           frame_labels, axis_user, rpm, rps, spin_type, cw, valid_fi)


if __name__ == "__main__":
    main()
