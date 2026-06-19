"""Render the hero figure: why SO(3) 3D IoU matters.

Deterministically projects the ground-truth and predicted 3D cuboids from the
shipped example onto the image plane (using the camera intrinsics ``K``) and
annotates each match with its full-``SO(3)`` 3D IoU and TP/FP verdict at the
0.50 threshold. The figure makes the core point of EvalMono3D visible:

    The most confident detection is near-perfect in 2D yet misaligned in 3D
    (IoU 0.43 < 0.50). 2D AP rewards it; SO(3) 3D AP does not.

Usage::

    python scripts/visualize.py            # -> assets/hero.png
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from evalmono3d import evaluate
from evalmono3d.iou3d import box3d_overlap

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = os.path.join(HERE, "example", "cdrone_test_example.json")
PRED_PATH = os.path.join(HERE, "example", "cdrone_test_example_pred.pth")
OUT_PATH = os.path.join(HERE, "assets", "hero.png")

# Palette (dark, high-contrast).
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21323f"
INK = "#e6edf3"
MUTED = "#8b949e"
GT_C = "#3fb950"      # ground truth   -> green
TP_C = "#58a6ff"      # true positive  -> blue
FP_C = "#f85149"      # false positive -> red

# Cuboid faces by corner index (front 0-3, back 4-7); see DATA.md.
FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def project(corners, K):
    """Project ``(8, 3)`` camera-space corners to ``(8, 2)`` pixels."""
    K = np.asarray(K)
    uvw = corners @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def draw_cuboid(ax, pts2d, color, lw=2.4, fill_alpha=0.10, ls="-", zbase=2):
    """Draw a projected cuboid with lightly shaded, depth-sorted faces."""
    for f in sorted(FACES, key=lambda f: -pts2d[f, 1].mean()):  # paint lower (nearer) faces last
        ax.add_patch(Polygon(pts2d[f], closed=True, facecolor=color, edgecolor="none",
                             alpha=fill_alpha, zorder=zbase))
    for i, j in EDGES:
        ax.plot(pts2d[[i, j], 0], pts2d[[i, j], 1], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=zbase + 1)


def ground_grid(ax, K, y=7.0, x_range=(-26, 30), z_range=(11, 46), step=4):
    """Faint perspective floor grid for a drone-view feel."""
    xs = np.arange(x_range[0], x_range[1] + 1, step)
    zs = np.arange(z_range[0], z_range[1] + 1, step)
    for x in xs:
        line = np.array([[x, y, z] for z in (z_range[0], z_range[1])])
        uv = project(line, K)
        ax.plot(uv[:, 0], uv[:, 1], color=GRID, lw=0.8, zorder=0)
    for z in zs:
        line = np.array([[x_range[0], y, z], [x_range[1], y, z]])
        uv = project(line, K)
        ax.plot(uv[:, 0], uv[:, 1], color=GRID, lw=0.8, zorder=0)


def main():
    gt = json.load(open(GT_PATH))
    K = gt["images"][0]["K"]
    W, H = gt["images"][0]["width"], gt["images"][0]["height"]
    gt_boxes = [np.array(a["bbox3D_cam"]) for a in gt["annotations"]]

    preds = torch.load(PRED_PATH)[0]["instances"]
    cars = sorted([p for p in preds if p["category_id"] == 0], key=lambda p: -p["score"])[:2]
    pred_boxes = [np.array(c["bbox3D"]) for c in cars]

    iou = box3d_overlap(
        torch.tensor(np.stack(pred_boxes), dtype=torch.float32),
        torch.tensor(np.stack(gt_boxes), dtype=torch.float32),
    ).numpy()

    # Headline AP numbers, computed (not hardcoded) so the banner always matches.
    results = evaluate("hero", GT_PATH, PRED_PATH)
    ap2d, ap3d = results["AP2D"], results["AP3D"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.edgecolor": GRID, "figure.facecolor": BG})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=140)
    fig.subplots_adjust(left=0, right=1, top=0.90, bottom=0.10)
    ax.set_facecolor(BG)
    # Crop to the action (both cars + their callouts) for a larger, punchier view.
    ax.set_xlim(560, 1540)
    ax.set_ylim(1010, 455)
    ax.axis("off")

    ground_grid(ax, K)

    # Ground truth.
    for g in gt_boxes:
        draw_cuboid(ax, project(g, K), GT_C, lw=2.2, fill_alpha=0.12, zbase=2)

    # Predictions, coloured by their verdict at IoU >= 0.50.
    for j, (c, pb) in enumerate(zip(cars, pred_boxes)):
        best_gt = int(iou[j].argmax())
        v = float(iou[j, best_gt])
        is_tp = v >= 0.5
        color = TP_C if is_tp else FP_C
        pts = project(pb, K)
        draw_cuboid(ax, pts, color, lw=2.8, fill_alpha=0.14, ls="-" if is_tp else (0, (5, 3)), zbase=4)

        cx, top = pts[:, 0].mean(), pts[:, 1].min()
        verdict = "TRUE POSITIVE" if is_tp else "FALSE POSITIVE"
        sign = "≥" if is_tp else "<"
        ax.annotate(
            f"score {c['score']:.2f}\n3D IoU = {v:.2f}  ({sign} 0.50)\n{verdict}",
            xy=(cx, top), xytext=(cx + (70 if j == 0 else -70), top - 150),
            ha="left" if j == 0 else "right", va="bottom", fontsize=13, color=INK, weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=color, lw=1.8),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))

    # Title.
    fig.text(0.5, 0.955, "EvalMono3D  —  monocular 3D detection with full SO(3) IoU",
             ha="center", fontsize=22, weight="bold", color=INK)
    fig.text(0.5, 0.915,
             "Same detections, two verdicts: 2D overlap looks great, but oriented 3D IoU is strict.",
             ha="center", fontsize=14, color=MUTED)

    # Legend.
    handles = [
        plt.Line2D([], [], color=GT_C, lw=3, label="Ground-truth 3D box"),
        plt.Line2D([], [], color=TP_C, lw=3, label="Detection  —  TP (IoU ≥ 0.50)"),
        plt.Line2D([], [], color=FP_C, lw=3, ls="--", label="Detection  —  FP (IoU < 0.50)"),
    ]
    ax.legend(handles=handles, loc="upper left", facecolor=PANEL, edgecolor=GRID,
              labelcolor=INK, fontsize=12, framealpha=0.95).set_zorder(10)

    # Result banner.
    for x0, lab, val, col in [(0.300, "2D AP", f"{ap2d:.2f}", TP_C),
                              (0.545, "3D AP @ 0.50", f"{ap3d:.2f}", FP_C)]:
        ax.figure.text(x0, 0.045, lab, ha="right", fontsize=15, color=MUTED, transform=fig.transFigure)
        ax.figure.text(x0 + 0.012, 0.045, val, ha="left", fontsize=17, color=col, weight="bold",
                       transform=fig.transFigure)
    fig.text(0.70, 0.045, "←  the gap SO(3) evaluation reveals", ha="left", fontsize=13,
             color=MUTED, style="italic")

    fig.savefig(OUT_PATH, facecolor=BG, bbox_inches=None)
    w, h = (fig.get_size_inches() * fig.dpi).astype(int)
    print(f"wrote {OUT_PATH} ({w}x{h})")


if __name__ == "__main__":
    main()
