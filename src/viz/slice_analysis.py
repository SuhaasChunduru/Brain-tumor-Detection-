"""Find the best, median, and worst test slices for one region and plot them.

Reports per-slice Dice, which is a stricter and more informative view than the
dataset-level Dice in final_metrics.csv.

Run:  python -m src.viz.slice_analysis [ET|TC|WT]
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import METRICS_DIR, MODALITY_NAMES, REGION_NAMES  # noqa: E402

OUT_DIR = "results/multi_analysis"

# Slices with a nearly empty ground truth make Dice meaningless, so they are
# excluded from the ranking.
MIN_GT_PIXELS = 30


def dice_score(gt, pr):
    gt = gt.astype(np.uint8)
    pr = pr.astype(np.uint8)

    tp = np.sum((gt == 1) & (pr == 1))
    fp = np.sum((gt == 0) & (pr == 1))
    fn = np.sum((gt == 1) & (pr == 0))

    if tp + fp + fn == 0:
        return 1.0
    return (2 * tp) / (2 * tp + fp + fn + 1e-7)


def show_case(X_test, Y_test, pred_bin, idx, region_idx, region, name):
    img = X_test[idx, :, :, MODALITY_NAMES.index("FLAIR")]
    gt = Y_test[idx, :, :, region_idx]
    pr = pred_bin[idx, :, :, region_idx]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("FLAIR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title(f"Ground Truth ({region})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap="gray")
    plt.imshow(pr, alpha=0.45, cmap="jet")
    plt.title(f"Prediction ({region})")
    plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Wrote", out_path)


def main():
    region = sys.argv[1].upper() if len(sys.argv) > 1 else "WT"
    if region not in REGION_NAMES:
        raise SystemExit(f"Region must be one of {REGION_NAMES}, got {region!r}")
    region_idx = REGION_NAMES.index(region)

    X_test = np.load(os.path.join(METRICS_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(METRICS_DIR, "Y_test.npy"))
    pred_bin = np.load(os.path.join(METRICS_DIR, "pred_bin.npy"))

    os.makedirs(OUT_DIR, exist_ok=True)

    dice_scores, indices = [], []
    for i in range(len(X_test)):
        gt = Y_test[i, :, :, region_idx]
        if np.sum(gt) < MIN_GT_PIXELS:
            continue
        dice_scores.append(dice_score(gt, pred_bin[i, :, :, region_idx]))
        indices.append(i)

    if not dice_scores:
        raise SystemExit(f"No test slice has at least {MIN_GT_PIXELS} {region} pixels.")

    dice_scores = np.array(dice_scores)
    print(f"\nPer-slice Dice summary for {region}")
    print("Slices evaluated:", len(dice_scores))
    print("Mean:", np.mean(dice_scores))
    print("Best:", np.max(dice_scores))
    print("Worst:", np.min(dice_scores))

    order = np.argsort(dice_scores)
    cases = {
        "best_case": indices[int(np.argmax(dice_scores))],
        "average_case": indices[int(order[len(order) // 2])],
        "worst_case": indices[int(np.argmin(dice_scores))],
    }
    for name, idx in cases.items():
        show_case(X_test, Y_test, pred_bin, idx, region_idx, region, name)


if __name__ == "__main__":
    main()
