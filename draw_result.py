import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

csv = "y8s128"
CSV_PATH = csv+".csv"

def smooth(y, weight=0.6):
    smoothed = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return smoothed

df = pd.read_csv(CSV_PATH)

cols = [
    ("train/box_loss",       "train/box_loss"),
    ("train/cls_loss",       "train/cls_loss"),
    ("train/dfl_loss",       "train/dfl_loss"),
    ("metrics/precision(B)", "metrics/precision(B)"),
    ("metrics/recall(B)",    "metrics/recall(B)"),
    ("val/box_loss",         "val/box_loss"),
    ("val/cls_loss",         "val/cls_loss"),
    ("val/dfl_loss",         "val/dfl_loss"),
    ("metrics/mAP50(B)",     "metrics/mAP50(B)"),
    ("metrics/mAP50-95(B)",  "metrics/mAP50-95(B)"),
]

fig = plt.figure(figsize=(18, 7.5))
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

epochs = df["epoch"].values

for i, (col, title) in enumerate(cols):
    row, col_idx = divmod(i, 5)
    ax = fig.add_subplot(gs[row, col_idx])

    y = df[col].values
    y_smooth = smooth(y)

    ax.plot(epochs, y, color="#1f77b4", linewidth=1.0,
            marker="o", markersize=2.5, label="results")
    ax.plot(epochs, y_smooth, color="#ff7f0e", linewidth=1.2,
            linestyle="dotted", label="smooth")

    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(axis="both", labelsize=7)
    ax.margins(x=0.02)

    if i == 1:
        ax.legend(fontsize=7, loc="upper right",
                  framealpha=0.7, handlelength=1.5)
        
save_path = csv+".png"
plt.savefig(save_path, dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print(f"Saved: {save_path}")