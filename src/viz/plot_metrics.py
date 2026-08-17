"""Regenerate the bar charts and training curves from saved run artifacts.

Every value plotted here is read from disk -- final_metrics.csv written by
src/evaluate.py and the history arrays written by src/train.py. Nothing is
hardcoded.

Run:  python -m src.viz.plot_metrics
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # save to file; never open a window
import matplotlib.pyplot as plt  # noqa: E402

from src.config import METRICS_DIR, MODEL_DIR, PLOTS_DIR  # noqa: E402

# Metric name -> (output filename, y-axis label, fixed y-limit or None)
BAR_CHARTS = {
    "Dice": ("dice_bar.png", "Dice", (0.0, 1.0)),
    "IoU": ("iou_bar.png", "IoU", (0.0, 1.0)),
    "HD95": ("hd95_bar.png", "HD95 (pixels)", None),
    "ASSD": ("assd_bar.png", "ASSD (pixels)", None),
}


def plot_metric_bars(df):
    for metric, (fname, ylabel, ylim) in BAR_CHARTS.items():
        if metric not in df.columns:
            print(f"Skipping {metric}: not in {METRICS_DIR}/final_metrics.csv")
            continue

        plt.figure()
        plt.bar(df.index, df[metric])
        plt.xlabel("Region")
        plt.ylabel(ylabel)
        plt.title(f"{metric} by Tumor Region")
        if ylim:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=200)
        plt.close()
        print("Wrote", os.path.join(PLOTS_DIR, fname))


def plot_curve(train_file, val_file, title, ylabel, out_name):
    train_path = os.path.join(MODEL_DIR, train_file)
    val_path = os.path.join(MODEL_DIR, val_file)

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"Skipping {out_name}: run src.train to produce {train_file}")
        return

    train = np.load(train_path)
    val = np.load(val_path)
    epochs = np.arange(1, len(train) + 1)

    plt.figure()
    plt.plot(epochs, train, label=f"Train {ylabel}")
    plt.plot(epochs, val, label=f"Val {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, out_name), dpi=200)
    plt.close()
    print("Wrote", os.path.join(PLOTS_DIR, out_name))


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    csv_path = os.path.join(METRICS_DIR, "final_metrics.csv")
    if os.path.exists(csv_path):
        plot_metric_bars(pd.read_csv(csv_path, index_col=0))
    else:
        print(f"Skipping bar charts: {csv_path} not found. Run src.evaluate first.")

    plot_curve("train_loss.npy", "val_loss.npy", "Loss Curve", "Loss", "loss.png")
    plot_curve("train_acc.npy", "val_acc.npy", "Accuracy Curve", "Accuracy", "accuracy.png")


if __name__ == "__main__":
    main()
