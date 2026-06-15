"""
DermWise — offline evaluation of the EfficientNet-B0 classifier.

Reproduces the held-out test metrics reported in README.md / MODEL_CARD.md
(Accuracy 0.819, Macro-F1 0.703, Melanoma recall 0.732) from:
  - models/best_model.pth   (the same weights deployed in app.py)
  - the saved lesion-level test split (test_split.csv, from training/dermwise_pipeline.ipynb)
  - the HAM10000 image files

This is an OFFLINE script. It is NOT imported by app.py and never runs in the
request path — running it cannot affect the live demo.

Usage:
  python eval.py \
      --test-split  /path/to/test_split.csv \
      --image-root  /path/to/HAM10000_images   # folder(s) containing <image_id>.jpg
      [--no-tta]                                 # evaluate without test-time augmentation

The test_split.csv produced by the notebook contains at least the columns
`image_id` and `label` (0..6). Image paths are resolved as <image-root>/<image_id>.jpg
(searched recursively), so you can point --image-root at the dataset root.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_image_lookup(image_root):
    """Map <image_id> -> absolute path by walking image_root for *.jpg."""
    lookup = {}
    for dirpath, _, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(".jpg"):
                lookup[os.path.splitext(f)[0]] = os.path.join(dirpath, f)
    return lookup


class TestDataset(Dataset):
    def __init__(self, df, image_lookup):
        self.ids = df["image_id"].values
        self.labels = df["label"].astype(int).values
        self.lookup = image_lookup

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.lookup[self.ids[idx]]
        img = Image.open(path).convert("RGB")
        return EVAL_TF(img), int(self.labels[idx])


def load_classifier(weights_path):
    """Same head as the deployed app.py so we evaluate the deployed weights."""
    model = models.efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_feat, len(CLASS_NAMES)))
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, loader, use_tta):
    """Return (probs[N,7], labels[N]). TTA = avg softmax over 4 flip views."""
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        if use_tta:
            p = (F.softmax(model(imgs), 1)
                 + F.softmax(model(torch.flip(imgs, [3])), 1)   # h-flip
                 + F.softmax(model(torch.flip(imgs, [2])), 1)   # v-flip
                 + F.softmax(model(torch.flip(imgs, [2, 3])), 1)  # both
                 ) / 4.0
        else:
            p = F.softmax(model(imgs), 1)
        all_probs.append(p.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-split", required=True, help="Path to test_split.csv from the notebook")
    ap.add_argument("--image-root", required=True, help="Folder containing HAM10000 <image_id>.jpg files")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "models", "best_model.pth"))
    ap.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.test_split)
    if "image_id" not in df.columns or "label" not in df.columns:
        sys.exit("test_split.csv must contain 'image_id' and 'label' columns")

    lookup = build_image_lookup(args.image_root)
    missing = [i for i in df["image_id"] if i not in lookup]
    if missing:
        sys.exit(f"{len(missing)} image(s) from the split not found under {args.image_root} "
                 f"(e.g. {missing[:3]})")

    loader = DataLoader(TestDataset(df, lookup), batch_size=args.batch_size, shuffle=False)
    model = load_classifier(args.weights)
    use_tta = not args.no_tta

    probs, labels = predict(model, loader, use_tta)
    preds = probs.argmax(1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"\n{'='*56}")
    print(f"  DermWise classifier — TEST set  ({'TTA' if use_tta else 'standard'})")
    print(f"{'='*56}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Melanoma recall : "
          f"{confusion_matrix(labels, preds)[CLASS_NAMES.index('mel')][CLASS_NAMES.index('mel')] / (labels == CLASS_NAMES.index('mel')).sum():.4f}")
    print("\n" + classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
