"""Train a Matador-C1 patch classifier.

Usage:
    python -m boilerplate.learning.train --model hgnn

Hyperparameters (Matador-C1):
    optimizer       AdamW, lr 3e-4, weight decay 1e-4
    label smoothing 0.05 (flat ResNet models)
    epochs          15
    image size      224, ImageNet normalization
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from boilerplate.learning.data import PatchDataset
from boilerplate.learning.losses import (
    hgnn_leaf_accuracy,
    hgnn_loss,
    leaf_accuracy,
)
from boilerplate.learning.models import (
    GlobalResNetPatchClassifier,
    HGNNPatchClassifier,
    ResNetPatchClassifier,
)
from boilerplate.learning.taxonomy import load_taxonomy

# --- fixed paths and hyperparameters -----------------------------------------
TAXONOMY_JSON = "taxonomy/assets/matador-c1-taxonomy.json"
IMAGE_ROOT = "data/raw/matador"
TRAIN_CSV = "data/processed/matador_c1/splits/train.csv"
VAL_CSV = "data/processed/matador_c1/splits/val.csv"

IMAGE_SIZE = 224
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
BATCH_SIZE = 128
NUM_WORKERS = 4


def build_transform():
    return T.Compose(
        [
            T.Resize(IMAGE_SIZE, antialias=True),
            T.CenterCrop(IMAGE_SIZE),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collate(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "leaf_target": torch.stack([b["leaf_target"] for b in batch]),
        "node_target": torch.stack([b["node_target"] for b in batch]),
    }
    if "context_image" in batch[0]:
        out["context_image"] = torch.stack([b["context_image"] for b in batch])
    return out


def make_loader(csv_path, taxonomy, use_context, shuffle):
    ds = PatchDataset(
        manifest_csv=csv_path,
        taxonomy=taxonomy,
        image_root=IMAGE_ROOT,
        context_col="context_path" if use_context else None,
        transform=build_transform(),
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
    )


def make_model(model_name, taxonomy):
    if model_name == "resnet":
        return ResNetPatchClassifier()
    if model_name == "global_resnet":
        return GlobalResNetPatchClassifier()
    if model_name == "hgnn":
        return HGNNPatchClassifier(taxonomy, use_context=False)
    if model_name == "global_hgnn":
        return HGNNPatchClassifier(taxonomy, use_context=True)
    raise ValueError(f"unknown model: {model_name}")


def run_epoch(model, loader, taxonomy, optimizer, device, train):
    model.train(train)
    leaf_indices = taxonomy.leaf_indices.to(device)
    levels = taxonomy.levels()
    is_hgnn = model.__class__.__name__ == "HGNNPatchClassifier"

    total_loss = total_acc = total = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        context = batch.get("context_image")
        context = context.to(device) if context is not None else None
        node_target = batch["node_target"].to(device)
        leaf_target = batch["leaf_target"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(image, context)
            if is_hgnn:
                loss = hgnn_loss(logits, node_target, levels)
                acc = hgnn_leaf_accuracy(logits, node_target, leaf_indices)
            else:
                # flat models predict over leaves only; remap to leaf-local index
                target = torch.searchsorted(leaf_indices, leaf_target)
                loss = torch.nn.functional.cross_entropy(
                    logits, target, label_smoothing=LABEL_SMOOTHING
                )
                acc = leaf_accuracy(logits, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        n = image.shape[0]
        total_loss += loss.item() * n
        total_acc += acc * n
        total += n

    return total_loss / total, total_acc / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet", "global_resnet", "hgnn", "global_hgnn"],
    )
    parser.add_argument("--out-dir", default="runs/patch_classifier")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_context = args.model in ("global_resnet", "global_hgnn")

    taxonomy = load_taxonomy(TAXONOMY_JSON)
    train_loader = make_loader(TRAIN_CSV, taxonomy, use_context, shuffle=True)
    val_loader = make_loader(VAL_CSV, taxonomy, use_context, shuffle=False)

    model = make_model(args.model, taxonomy).to(device)
    if args.model in ("hgnn", "global_hgnn"):
        # initialize taxonomy-node prototypes from average CNN embeddings
        model.init_prototypes(train_loader, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, taxonomy, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, taxonomy, optimizer, device, False)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save(
                {"model": model.state_dict(), "model_name": args.model},
                out_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
