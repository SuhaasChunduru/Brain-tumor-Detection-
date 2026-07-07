# src/advanced_visual_results.py

import numpy as np
import matplotlib.pyplot as plt
import os

# load data
X_test = np.load("results/metrics/X_test.npy")
Y_test = np.load("results/metrics/Y_test.npy")
pred_bin = np.load("results/metrics/pred_bin.npy")

os.makedirs("results/advanced_visuals", exist_ok=True)

# -------- COLOR MAP ----------
# 0=bg, 1=ET, 2=TC, 3=WT
def colorize(mask):
    colored = np.zeros((*mask.shape, 3))
    colored[mask == 1] = [1, 1, 0]      # ET = yellow
    colored[mask == 2] = [1, 0.5, 0.5]  # TC = pink
    colored[mask == 3] = [0.5, 0.7, 1]  # WT = blue
    return colored

# -------- CREATE MULTI-CLASS ----------
def to_multiclass(y):
    et = y[:, :, 0]
    tc = y[:, :, 1]
    wt = y[:, :, 2]

    out = np.zeros_like(et)
    out[wt == 1] = 3
    out[tc == 1] = 2
    out[et == 1] = 1
    return out

# -------- SHOW RESULTS ----------
def show_case(idx):
    img = X_test[idx]

    t1 = img[:, :, 0]
    t2 = img[:, :, 1]
    t1c = img[:, :, 2]
    flair = img[:, :, 3]

    gt = to_multiclass(Y_test[idx])
    pr = to_multiclass(pred_bin[idx])

    gt_color = colorize(gt)
    pr_color = colorize(pr)

    plt.figure(figsize=(18,5))

    plt.subplot(1,6,1); plt.imshow(t1, cmap='gray'); plt.title("T1"); plt.axis('off')
    plt.subplot(1,6,2); plt.imshow(t2, cmap='gray'); plt.title("T2"); plt.axis('off')
    plt.subplot(1,6,3); plt.imshow(t1c, cmap='gray'); plt.title("T1ce"); plt.axis('off')
    plt.subplot(1,6,4); plt.imshow(flair, cmap='gray'); plt.title("FLAIR"); plt.axis('off')

    plt.subplot(1,6,5)
    plt.imshow(flair, cmap='gray')
    plt.imshow(gt_color, alpha=0.5)
    plt.title("GT")
    plt.axis('off')

    plt.subplot(1,6,6)
    plt.imshow(flair, cmap='gray')
    plt.imshow(pr_color, alpha=0.5)
    plt.title("Prediction")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"results/advanced_visuals/sample_{idx}.png", dpi=200)
    plt.show()

# -------- RUN MULTIPLE ----------
for i in range(5):
    show_case(i)