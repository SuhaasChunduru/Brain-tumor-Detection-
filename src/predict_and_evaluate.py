import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_erosion, distance_transform_edt
import pandas as pd

CACHE_DIR = "data/cache_npz"
IMG_SIZE = 160
BATCH_SIZE = 4

all_files = sorted([
    os.path.join(CACHE_DIR, f)
    for f in os.listdir(CACHE_DIR)
    if f.endswith(".npz")
])

train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

def npz_generator(file_list):
    for path in file_list:
        data = np.load(path)
        X = data["X"]
        Y = data["Y"]
        for i in range(len(X)):
            yield X[i], Y[i]

output_signature = (
    tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
)

test_ds = tf.data.Dataset.from_generator(
    lambda: npz_generator(test_files),
    output_signature=output_signature
).batch(BATCH_SIZE)

X_test_list, Y_test_list = [], []
for xb, yb in test_ds:
    X_test_list.append(xb.numpy())
    Y_test_list.append(yb.numpy())

X_test = np.concatenate(X_test_list, axis=0)
Y_test = np.concatenate(Y_test_list, axis=0)

model = tf.keras.models.load_model(
    "results/models/best_model.keras",
    custom_objects={"hybrid_boundary_et_loss": None},
    compile=False
)

preds = model.predict(X_test, verbose=1)

# ET threshold lowered slightly
pred_bin = np.zeros_like(preds, dtype=np.uint8)
pred_bin[..., 0] = (preds[..., 0] > 0.40).astype(np.uint8)  # ET
pred_bin[..., 1] = (preds[..., 1] > 0.50).astype(np.uint8)  # TC
pred_bin[..., 2] = (preds[..., 2] > 0.50).astype(np.uint8)  # WT

gt_bin = (Y_test > 0.5).astype(np.uint8)

def dice(tp, fp, fn, eps=1e-7):
    return (2*tp + eps) / (2*tp + fp + fn + eps)

def iou(tp, fp, fn, eps=1e-7):
    return (tp + eps) / (tp + fp + fn + eps)

def sensitivity(tp, fn, eps=1e-7):
    return (tp + eps) / (tp + fn + eps)

def specificity(tn, fp, eps=1e-7):
    return (tn + eps) / (tn + fp + eps)

def surface_distances(a, b):
    a = a.astype(bool)
    b = b.astype(bool)

    if a.sum() == 0 and b.sum() == 0:
        return np.array([0.0])
    if a.sum() == 0 or b.sum() == 0:
        return np.array([np.inf])

    a_er = binary_erosion(a)
    b_er = binary_erosion(b)

    a_surf = a ^ a_er
    b_surf = b ^ b_er

    dt_b = distance_transform_edt(~b_surf)
    dt_a = distance_transform_edt(~a_surf)

    d_ab = dt_b[a_surf]
    d_ba = dt_a[b_surf]

    return np.concatenate([d_ab, d_ba])

def hd95_assd(a, b):
    d = surface_distances(a, b)
    if np.isinf(d).any():
        return np.inf, np.inf
    return np.percentile(d, 95), np.mean(d)

names = ["ET", "TC", "WT"]
results = {}

for c in range(3):
    TP = FP = FN = TN = 0
    hd95_list = []
    assd_list = []

    for i in range(gt_bin.shape[0]):
        g = gt_bin[i, :, :, c]
        p = pred_bin[i, :, :, c]

        TP += np.sum((g == 1) & (p == 1))
        FP += np.sum((g == 0) & (p == 1))
        FN += np.sum((g == 1) & (p == 0))
        TN += np.sum((g == 0) & (p == 0))

        h, a = hd95_assd(g, p)
        if np.isfinite(h):
            hd95_list.append(h)
        if np.isfinite(a):
            assd_list.append(a)

    results[names[c]] = {
        "Dice": float(dice(TP, FP, FN)),
        "IoU": float(iou(TP, FP, FN)),
        "Sensitivity": float(sensitivity(TP, FN)),
        "Specificity": float(specificity(TN, FP)),
        "HD95": float(np.mean(hd95_list)) if hd95_list else np.inf,
        "ASSD": float(np.mean(assd_list)) if assd_list else np.inf
    }

df = pd.DataFrame(results).T
os.makedirs("results/metrics", exist_ok=True)
df.to_csv("results/metrics/final_metrics.csv")

print("\n✅ Final Results")
print(df)

np.save("results/metrics/X_test.npy", X_test)
np.save("results/metrics/Y_test.npy", Y_test)
np.save("results/metrics/preds.npy", preds)
np.save("results/metrics/pred_bin.npy", pred_bin)
print("✅ Predictions saved")