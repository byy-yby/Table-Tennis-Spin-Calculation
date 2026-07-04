# Table Tennis Spin Calculation

![spinCal Screenshot](spinCal.png)

Automated table tennis ball spin measurement from high-speed camera video. Uses subpixel ball-edge fitting, a CNN heatmap model for dot detection, 3D spherical projection, per-frame-pair RANSAC Kabsch rotation with ICP refinement, and interactive visualization.

---

## Measurement Principle

### 1. Ball Localization & Cropping

From high-speed video, background subtraction finds the frames where the ball appears and crops each to 60×60. The ball edge is then refined to **subpixel** accuracy by `ballfit.py`: radial Sobel-gradient scanning along 72 rays locates the outermost strong edge point per ray (rejecting interior black-dot edges), then a RANSAC circle fit aggregates ~50+ edge points into a precise center + radius. This replaces Hough/BallNet for the ball boundary — circle-fit precision ≈ single-point noise / √N ≈ 0.3 px even on jagged low-res edges. A manual adjustment step (drag center + radius slider) remains for extreme cases.

### 2. CNN Dot Detection

A pre-trained DotNet (fully convolutional heatmap regression network) processes the 60×60 ball image to produce a dot heatmap. Non-maximum suppression extracts 2D pixel coordinates. Dots beyond 85% of the ball radius are dropped — edge points suffer amplified ray-sphere projection error.

### 3. 2D → 3D Projection

Using calibrated camera intrinsics, each 2D dot is projected onto the ball surface via ray-sphere intersection. The result is a set of 3D unit vectors on the sphere, ball-center-relative. **Translation is automatically removed** by the center subtraction, and a physical rotation is frame-independent, so the camera-frame rotation equals the true ball rotation — no camera extrinsics needed for RPM/axis magnitude.

### 4. Cross-Frame Matching & 5-Stage Pipeline

Rotation is estimated by a single-pass pipeline (`run_pipeline`, no outer loop — preserving real spin decay/precession):

| Stage | Action |
|-------|--------|
| 1 | **3D proximity matching** (wide threshold) → initial dot IDs |
| 2 | **Rough rotation** — per-frame-pair RANSAC Kabsch (≥3 common dots) → initial RPM/axis |
| 3 | **ICP re-matching** — rotation-constrained Hungarian + internal Kabsch correction (tight threshold) → refined matches (falls back to Stage 2 if it fails) |
| 4 | **Trajectory consistency** — drop dot trajectories whose RMS deviation from the global rotation is a relative outlier |
| 5 | **Final rotation + error** — per-pair RANSAC Kabsch aggregated by median |

Per adjacent frame pair `(fi, fi+1)`, an independent RANSAC Kabsch (3-point sampling, inlier consensus) yields one signed omega + axis. The Kabsch implementation does **not** subtract the centroid (points are already ball-center-relative). Requiring ≥3 common dots avoids the underdetermined 2-point Kabsch (free rotation about the 2-point axis).

### 5. Aggregation & Output

RPM = median of per-pair `|omega|` (robust to outlier clusters for small N), with 2×-median outlier rejection. Axis = inlier-weighted average. Output:
- **RPM** / **RPS** (revolutions per minute / second)
- **Rotation axis** (3D unit vector)
- **Spin type** (topspin / backspin / sidespin / gyro)
- **Spin direction** (CW / CCW)
- **Error estimate**: RPS median ± 95% CI of the median + per-pair robust σ (CV%), and axis RMS deviation from consensus

---

## Coordinate System

```
+X : image right
+Y : image down
+Z : camera forward (away from camera)
```

Visible dots have negative Z (near hemisphere). The 3D visualization shows the rotation axis as a cyan arrow.

---

## Project Structure

```
Table-Tennis-Spin-Calculation/
├── spinCal/              # Core spin measurement package
│   ├── main.py           # CLI entry point
│   ├── config.py         # Camera parameters & thresholds
│   ├── geometry.py       # 3D geometry (ray-sphere, Kabsch)
│   ├── ballfit.py        # Subpixel ball-edge detection (radial gradient + RANSAC circle)
│   ├── model.py          # DotNet CNN definition
│   ├── detection.py      # Per-frame processing (dot detection + 2D→3D)
│   ├── matching.py       # 3D proximity + ICP matching & ID tracking
│   ├── rotation.py       # 5-stage pipeline: RANSAC Kabsch + ICP + trajectory + error
│   ├── video.py          # AVI → cropped frames
│   └── viz.py            # Interactive viewer & 3D sphere plot
│
├── getData/              # Data production pipeline
├── dotnet/               # DotNet training (model/dataset/train)
├── ballnet/              # BallNet training (legacy, optional)
├── spinCal.png           # Screenshot
└── README.md             # This file
```

Datasets (`dataset/`, `dataset_ball/`), models (`*.pt`), and videos (`highspeed/`) are git-ignored.

---

## Usage

```bash
# From AVI video (subpixel ball-edge detection auto-runs, then manual adjust)
python -m spinCal.main <video.avi>

# From image folder
python -m spinCal.main <image folder> --model <dotnet.pt>

# Skip interactive viewer
python -m spinCal.main <video.avi> --no-viz
```

After loading, drag the ball center / adjust the radius slider per frame (ENTER=next, B=back, F=done) to override the auto-detected edge for any frame.

### Custom Training Workflow

```bash
python getData/extract.py <video directory>     # 1. Extract ball ROIs
python getData/label.py <dataset directory>     # 2. Label black dots
python getData/augment.py <dataset directory>   # 3. Augment
python dotnet/train.py <dataset directory> --epochs 100   # 4. Train DotNet
python -m spinCal.main <video.avi> --model <dotnet.pt>    # 5. Measure
```

---

## Adjustable Parameters

In `spinCal/config.py`:

| Parameter            | Default | Description                                                  |
| -------------------- | ------- | ------------------------------------------------------------ |
| `MATCH_MAX_DISP`     | 0.3     | Stage 1 matching max 3D displacement (must cover per-frame rotation) |
| `ICP_MATCH_DISP`     | 0.12    | Stage 3 ICP tight threshold (rotation-constrained)           |
| `MIN_COMMON_DOTS`    | 3       | Min common dots per frame-pair (2-point Kabsch is underdetermined) |
| `CNN_HMAP_THRESH`    | 0.3     | DotNet heatmap detection threshold                           |
| `MAX_DOT_EDGE_RATIO` | 0.85    | Drop dots beyond 85% of ball radius (edge projection error)  |
| `TRAJ_RMS_THRESH`    | 0.10    | Trajectory consistency RMS floor (relative outlier rejection) |

Override matching threshold via CLI: `--match-disp 0.3`

---

## Dependencies

- Python >= 3.10
- PyTorch + CUDA
- OpenCV, NumPy, SciPy
- Matplotlib

---

## References

- Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors"
- Besl & McKay (1992). "A Method for Registration of 3-D Shapes" (ICP)
- SpinDOE: Gossard et al., "A ball spin estimation method for table tennis robot", arXiv:2303.03879
