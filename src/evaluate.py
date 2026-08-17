"""Score the trained model on the held-out test patients.

Writes results/metrics/final_metrics.csv plus the raw test tensors and
predictions that the visualization scripts reload.

Run:  python -m src.evaluate
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.config import METRICS_DIR, MODEL_DIR, REGION_NAMES
from src.dataset import make_dataset, patient_splits

# ET is small and under-predicted at 0.5, so its decision threshold is lowered.
THRESHOLDS = {"ET": 0.40, "TC": 0.50, "WT": 0.50}


def dice(tp, fp, fn, eps=1e-7):
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou(tp, fp, fn, eps=1e-7):
    return (tp + eps) / (tp + fp + fn + eps)


def sensitivity(tp, fn, eps=1e-7):
    return (tp + eps) / (tp + fn + eps)


def specificity(tn, fp, eps=1e-7):
    return (tn + eps) / (tn + fp + eps)


def surface_distances(a, b):
    """Symmetric surface-to-surface distances, in pixels, for two 2D masks."""
    a = a.astype(bool)
    b = b.astype(bool)

    if a.sum() == 0 and b.sum() == 0:
        return np.array([0.0])
    if a.sum() == 0 or b.sum() == 0:
        return np.array([np.inf])

    a_surf = a ^ binary_erosion(a)
    b_surf = b ^ binary_erosion(b)

    d_ab = distance_transform_edt(~b_surf)[a_surf]
    d_ba = distance_transform_edt(~a_surf)[b_surf]

    return np.concatenate([d_ab, d_ba])


def hd95_assd(a, b):
    d = surface_distances(a, b)
    if np.isinf(d).any():
        return np.inf, np.inf
    return np.percentile(d, 95), np.mean(d)


def load_test_arrays():
    """Materialize the whole test split into memory as (X, Y)."""
    _, _, test_files = patient_splits()
    test_ds = make_dataset(test_files)

    X_list, Y_list = [], []
    for xb, yb in test_ds:
        X_list.append(xb.numpy())
        Y_list.append(yb.numpy())

    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)


def binarize(preds):
    pred_bin = np.zeros_like(preds, dtype=np.uint8)
    for c, name in enumerate(REGION_NAMES):
        pred_bin[..., c] = (preds[..., c] > THRESHOLDS[name]).astype(np.uint8)
    return pred_bin


def score(gt_bin, pred_bin):
    """Micro-averaged overlap metrics plus slice-averaged surface distances."""
    results = {}

    for c, name in enumerate(REGION_NAMES):
        TP = FP = FN = TN = 0
        hd95_list, assd_list = [], []

        for i in range(gt_bin.shape[0]):
            g = gt_bin[i, :, :, c]
            p = pred_bin[i, :, :, c]

            # Voxel counts accumulate across the whole split before Dice/IoU are
            # taken, so large tumours weigh more than small ones.
            TP += np.sum((g == 1) & (p == 1))
            FP += np.sum((g == 0) & (p == 1))
            FN += np.sum((g == 1) & (p == 0))
            TN += np.sum((g == 0) & (p == 0))

            # Distance metrics are undefined when either mask is empty; those
            # slices are skipped rather than counted as a maximum penalty.
            h, a = hd95_assd(g, p)
            if np.isfinite(h):
                hd95_list.append(h)
            if np.isfinite(a):
                assd_list.append(a)

        results[name] = {
            "Dice": float(dice(TP, FP, FN)),
            "IoU": float(iou(TP, FP, FN)),
            "Sensitivity": float(sensitivity(TP, FN)),
            "Specificity": float(specificity(TN, FP)),
            "HD95": float(np.mean(hd95_list)) if hd95_list else np.inf,
            "ASSD": float(np.mean(assd_list)) if assd_list else np.inf,
        }

    return pd.DataFrame(results).T


def main():
    X_test, Y_test = load_test_arrays()

    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "best_model.keras"), compile=False
    )
    preds = model.predict(X_test, verbose=1)

    pred_bin = binarize(preds)
    gt_bin = (Y_test > 0.5).astype(np.uint8)

    df = score(gt_bin, pred_bin)

    os.makedirs(METRICS_DIR, exist_ok=True)
    df.to_csv(os.path.join(METRICS_DIR, "final_metrics.csv"))
    print("\nFinal results")
    print(df)

    # Cached so the visualization scripts do not need to re-run inference.
    np.save(os.path.join(METRICS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(METRICS_DIR, "Y_test.npy"), Y_test)
    np.save(os.path.join(METRICS_DIR, "preds.npy"), preds)
    np.save(os.path.join(METRICS_DIR, "pred_bin.npy"), pred_bin)
    print(f"Metrics and predictions saved to {METRICS_DIR}/")


if __name__ == "__main__":
    main()
