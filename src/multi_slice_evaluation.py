import os
import numpy as np
import matplotlib.pyplot as plt

X_test = np.load("results/metrics/X_test.npy")
Y_test = np.load("results/metrics/Y_test.npy")
pred_bin = np.load("results/metrics/pred_bin.npy")

os.makedirs("results/multi_analysis", exist_ok=True)

# -----------------------------
# Dice function
# -----------------------------
def dice_score(gt, pr):
    gt = gt.astype(np.uint8)
    pr = pr.astype(np.uint8)

    tp = np.sum((gt == 1) & (pr == 1))
    fp = np.sum((gt == 0) & (pr == 1))
    fn = np.sum((gt == 1) & (pr == 0))

    if tp + fp + fn == 0:
        return 1.0
    return (2 * tp) / (2 * tp + fp + fn + 1e-7)

# -----------------------------
# Evaluate multiple slices
# -----------------------------
region_idx = 2  # WT (change to 0 for ET, 1 for TC)

dice_scores = []
indices = []

for i in range(len(X_test)):
    gt = Y_test[i, :, :, region_idx]
    pr = pred_bin[i, :, :, region_idx]

    if np.sum(gt) < 30:
        continue

    d = dice_score(gt, pr)
    dice_scores.append(d)
    indices.append(i)

dice_scores = np.array(dice_scores)

print("\n📊 SUMMARY")
print("Total evaluated slices:", len(dice_scores))
print("Average Dice:", np.mean(dice_scores))
print("Best Dice:", np.max(dice_scores))
print("Worst Dice:", np.min(dice_scores))

# -----------------------------
# Select representative cases
# -----------------------------
best_idx = indices[np.argmax(dice_scores)]
worst_idx = indices[np.argmin(dice_scores)]
mid_idx = indices[np.argsort(dice_scores)[len(dice_scores)//2]]

# -----------------------------
# Plot function
# -----------------------------
def show_case(idx, name):
    img = X_test[idx, :, :, 3]
    gt = Y_test[idx, :, :, region_idx]
    pr = pred_bin[idx, :, :, region_idx]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap="gray")
    plt.imshow(pr, alpha=0.45, cmap="jet")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"results/multi_analysis/{name}.png", dpi=200)
    plt.show()

# -----------------------------
# Show 3 important examples
# -----------------------------
print("\nShowing BEST case")
show_case(best_idx, "best_case")

print("\nShowing AVERAGE case")
show_case(mid_idx, "average_case")

print("\nShowing WORST case")
show_case(worst_idx, "worst_case")