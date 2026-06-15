"""
Render the DermWise classifier confusion matrices from the recorded TEST/VAL
results (TTA variant) produced by the training notebook (training/dermwise_pipeline.ipynb).

These counts are transcribed verbatim from the notebook's evaluation output
(Cell 4 — EVALUATION). This script only re-renders the same numbers as a clean
PNG for the repo; it does not recompute anything.

Run:  python docs/assets/make_confusion_matrix.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Rows = true class, Cols = predicted class. TTA variant, held-out splits.
VAL = np.array([
    [33,  4,  3,  0,  2,   0,  1],   # akiec  (43)
    [ 2, 63,  3,  0,  4,   0,  0],   # bcc    (72)
    [14, 10,102,  0, 26,  15,  0],   # bkl    (167)
    [ 0,  4,  0, 12,  0,   1,  0],   # df     (17)
    [ 5,  3, 11,  1,119,  27,  2],   # mel    (168)
    [ 2, 14, 18,  0, 64, 876,  1],   # nv     (975)
    [ 2,  0,  0,  0,  0,   1, 19],   # vasc   (22)
])

TEST = np.array([
    [38,  4,  6,  1,  2,   0,  0],   # akiec  (51)
    [10, 59,  1,  3,  1,   2,  1],   # bcc    (77)
    [13,  7, 90,  2, 20,  25,  0],   # bkl    (157)
    [ 6,  1,  0, 13,  1,   1,  0],   # df     (22)
    [ 8,  4,  5,  0,123,  28,  0],   # mel    (168)
    [ 7, 16, 29,  4, 59, 884,  1],   # nv     (1000)
    [ 0,  1,  0,  0,  1,   1, 19],   # vasc   (22)
])

TITLES = {
    "Validation": (VAL, 0.7509, 0.8361),
    "Test":       (TEST, 0.7033, 0.8190),
}


def _plot(ax, cm, title, f1, acc):
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_title(f"{title} Confusion Matrix (TTA)\nMacro F1={f1:.4f}  Acc={acc:.4f}", fontsize=12)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)", ha="center", va="center",
                    fontsize=8, color="white" if cm_pct[i, j] > 60 else "black")
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, (name, (cm, f1, acc)) in zip(axes, TITLES.items()):
        im = _plot(ax, cm, name, f1, acc)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Recall %")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
