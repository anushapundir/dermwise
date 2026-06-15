"""
Grad-CAM explainability for the DermWise EfficientNet-B0 classifier.

Produces a grid of Grad-CAM heatmap overlays — one correctly-classified test
image per class — to visually confirm the model attends to the LESION rather
than to image artifacts (rulers, ink, hair, vignetting).

How Grad-CAM works (no external library needed — implemented with hooks):
  1. forward pass; capture the activations of the last conv block
  2. backprop the predicted-class score; capture the gradients of those activations
  3. weight each activation map by its mean gradient, sum, ReLU
  4. upsample the resulting map to the image size and overlay it

Designed for the Kaggle environment used in dermwise_pipeline.ipynb (paths under
/kaggle/working and /kaggle/input). Override the paths with flags if needed.

Run on Kaggle (after the training cells have produced best_model.pth + test_split.csv):
    python gradcam.py
or with explicit paths:
    python gradcam.py \
        --weights    /kaggle/working/artifacts/classifier/best_model.pth \
        --test-split /kaggle/working/artifacts/classifier/test_split.csv \
        --image-root /kaggle/input \
        --out        gradcam_examples.png

Then download gradcam_examples.png and place it in docs/assets/ in the repo.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL = {
    "akiec": "Actinic Keratosis", "bcc": "Basal Cell Carcinoma", "bkl": "Benign Keratosis",
    "df": "Dermatofibroma", "mel": "Melanoma", "nv": "Melanocytic Nevi", "vasc": "Vascular Lesion",
}
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_classifier(weights_path):
    """Same head as the deployed app.py / eval.py so we explain the deployed weights."""
    model = models.efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_feat, len(CLASS_NAMES)))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def build_image_lookup(image_root):
    lookup = {}
    if image_root and os.path.isdir(image_root):
        for dp, _, files in os.walk(image_root):
            for f in files:
                if f.lower().endswith(".jpg"):
                    lookup[os.path.splitext(f)[0]] = os.path.join(dp, f)
    return lookup


def resolve_path(row, lookup):
    """Prefer the notebook's saved absolute image_path; else look up by image_id."""
    p = row.get("image_path")
    if isinstance(p, str) and os.path.exists(p):
        return p
    return lookup.get(row.get("image_id"))


def grad_cam(model, target_layer, x, class_idx):
    """Return a [224,224] heatmap in [0,1] for class_idx given input x [1,3,224,224]."""
    store = {}
    h_fwd = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o.detach()))
    h_bwd = target_layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0].detach()))
    x = x.clone().requires_grad_(True)
    model.zero_grad()
    logits = model(x)
    logits[0, class_idx].backward()
    h_fwd.remove()
    h_bwd.remove()

    act = store["act"][0]            # [C, h, w]
    grad = store["grad"][0]          # [C, h, w]
    weights = grad.mean(dim=(1, 2))  # [C]  global-average-pooled gradients
    cam = F.relu((weights[:, None, None] * act).sum(0))      # [h, w]
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False)[0, 0]
    return cam.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/kaggle/working/artifacts/classifier/best_model.pth")
    ap.add_argument("--test-split", default="/kaggle/working/artifacts/classifier/test_split.csv")
    ap.add_argument("--image-root", default="/kaggle/input", help="Fallback dir to find <image_id>.jpg")
    ap.add_argument("--out", default="gradcam_examples.png")
    ap.add_argument("--max-try", type=int, default=40, help="Images to scan per class to find a correct one")
    # parse_known_args (not parse_args) so this also runs when pasted into a Jupyter/Kaggle
    # cell, where sys.argv carries kernel flags like -f that argparse would otherwise reject.
    args, _ = ap.parse_known_args()

    model = load_classifier(args.weights)
    target_layer = model.features[-1]   # last conv block of EfficientNet-B0
    df = pd.read_csv(args.test_split)
    lookup = build_image_lookup(args.image_root)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, cls_idx in zip(axes, range(len(CLASS_NAMES))):
        sub = df[df["label"].astype(int) == cls_idx].head(args.max_try)
        chosen = None
        for _, row in sub.iterrows():
            path = resolve_path(row, lookup)
            if not path:
                continue
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            x = TF(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = int(model(x).argmax(1).item())
            if pred == cls_idx:                 # prefer a correctly-classified example
                chosen = (img, x, pred)
                break
        if chosen is None and path:             # fallback: last scanned image
            chosen = (img, x, pred)
        if chosen is None:
            ax.axis("off")
            continue

        img, x, pred = chosen
        cam = grad_cam(model, target_layer, x, cls_idx)
        ax.imshow(np.asarray(img))
        ax.imshow(cam, cmap="jet", alpha=0.45)
        mark = "OK" if pred == cls_idx else "MISS"
        ax.set_title(f"{CLASS_FULL[CLASS_NAMES[cls_idx]]} [{mark}]", fontsize=11)
        ax.axis("off")

    axes[-1].axis("off")
    fig.suptitle("DermWise — Grad-CAM (model attention per class)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
