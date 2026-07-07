import matplotlib.pyplot as plt
import numpy as np

# Example values (replace with your actual values if needed)
epochs = list(range(1, 11))

dice = [0.78, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88]
iou  = [0.65, 0.67, 0.68, 0.69, 0.70, 0.72, 0.73, 0.74, 0.75, 0.77]

hd95 = [15, 13, 12, 11, 10, 9, 8, 7, 6, 5]
assd = [4.5, 4.0, 3.8, 3.5, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2]

# -------------------------
# Dice + IoU Graph
# -------------------------
plt.figure()
plt.plot(epochs, dice, label="Dice Score")
plt.plot(epochs, iou, label="IoU")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Segmentation Performance")
plt.legend()
plt.savefig("dice_iou.png")
plt.show()

# -------------------------
# HD95 + ASSD Graph
# -------------------------
plt.figure()
plt.plot(epochs, hd95, label="HD95")
plt.plot(epochs, assd, label="ASSD")
plt.xlabel("Epoch")
plt.ylabel("Distance Error")
plt.title("Boundary Error Metrics")
plt.legend()
plt.savefig("hd95_assd.png")
plt.show()

print("Graphs saved successfully ✅")