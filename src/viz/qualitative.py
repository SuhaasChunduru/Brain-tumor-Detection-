"""Six-panel figures: all four input modalities beside ground truth and prediction.

Run:  python -m src.viz.qualitative [n_samples]
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import METRICS_DIR, MODALITY_NAMES  # noqa: E402

OUT_DIR = "results/advanced_visuals"

# Nested regions are painted largest-first so the smaller ones stay visible.
REGION_COLORS = {
    1: [1.0, 1.0, 0.0],  # ET  - yellow
    2: [1.0, 0.5, 0.5],  # TC  - pink
    3: [0.5, 0.7, 1.0],  # WT  - blue
}


def colorize(mask):
    colored = np.zeros((*mask.shape, 3))
    for value, rgb in REGION_COLORS.items():
        colored[mask == value] = rgb
    return colored


def to_multiclass(y):
    """Flatten the three overlapping binary channels into one label image.

    ET is written last so it wins wherever the regions overlap.
    """
    et, tc, wt = y[:, :, 0], y[:, :, 1], y[:, :, 2]

    out = np.zeros_like(et)
    out[wt == 1] = 3
    out[tc == 1] = 2
    out[et == 1] = 1
    return out


def show_case(X_test, Y_test, pred_bin, idx):
    img = X_test[idx]
    # Overlays go on FLAIR, where whole-tumour edema is most visible.
    backdrop = img[:, :, MODALITY_NAMES.index("FLAIR")]

    gt_color = colorize(to_multiclass(Y_test[idx]))
    pr_color = colorize(to_multiclass(pred_bin[idx]))

    plt.figure(figsize=(18, 5))

    # Panel order follows MODALITY_NAMES, i.e. the channel order the
    # preprocessing script stacks.
    for c, name in enumerate(MODALITY_NAMES):
        plt.subplot(1, 6, c + 1)
        plt.imshow(img[:, :, c], cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.imshow(backdrop, cmap="gray")
    plt.imshow(gt_color, alpha=0.5)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 6, 6)
    plt.imshow(backdrop, cmap="gray")
    plt.imshow(pr_color, alpha=0.5)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"sample_{idx}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Wrote", out_path)


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    X_test = np.load(os.path.join(METRICS_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(METRICS_DIR, "Y_test.npy"))
    pred_bin = np.load(os.path.join(METRICS_DIR, "pred_bin.npy"))

    os.makedirs(OUT_DIR, exist_ok=True)

    for i in range(min(n_samples, len(X_test))):
        show_case(X_test, Y_test, pred_bin, i)


if __name__ == "__main__":
    main()
