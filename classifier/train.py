import argparse
import random
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader

from classifier.data import PatchDataset
from classifier.losses import hgnn_leaf_accuracy, hgnn_loss, leaf_accuracy
from classifier.models import (
    GlobalResNetPatchClassifier,
    HGNNPatchClassifier,
    ResNetPatchClassifier,
)
from classifier.taxonomy import load_taxonomy


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_transform(image_size):
    return T.Compose(
        [
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collate(batch):
    out = {
        "image": torch.stack([item["image"] for item in batch]),
        "leaf_target": torch.stack([item["leaf_target"] for item in batch]),
        "node_target": torch.stack([item["node_target"] for item in batch]),
        "label": [item["label"] for item in batch],
    }
    if "context_image" in batch[0]:
        out["context_image"] = torch.stack([item["context_image"] for item in batch])
    return out


def make_loader(config, split):
    taxonomy = load_taxonomy(config["taxonomy"], root=config.get("root", "root"))
    data_cfg = config["data"]
    ds = PatchDataset(
        manifest_csv=data_cfg[f"{split}_csv"],
        taxonomy=taxonomy,
        image_root=data_cfg.get("image_root"),
        image_tar=data_cfg.get("image_tar"),
        label_col=data_cfg.get("label_col", "c1_label"),
        image_col=data_cfg.get("image_col", "image_path"),
        context_col=data_cfg.get("context_col"),
        transform=build_transform(config["training"].get("image_size", 224)),
    )
    loader = DataLoader(
        ds,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=(split == "train"),
        num_workers=config["training"].get("num_workers", 2),
        collate_fn=collate,
    )
    return loader, taxonomy


def make_model(config, taxonomy):
    model_type = config["model"]["type"]
    backbone = config["model"].get("backbone", "resnet50")
    pretrained = config["model"].get("pretrained", True)
    if model_type == "resnet":
        return ResNetPatchClassifier(len(taxonomy.leaves), backbone, pretrained)
    if model_type == "global_resnet":
        return GlobalResNetPatchClassifier(len(taxonomy.leaves), backbone, pretrained)
    if model_type == "hgnn":
        return HGNNPatchClassifier(
            taxonomy,
            backbone=backbone,
            pretrained=pretrained,
            image_embed_dim=config["model"].get("image_embed_dim", 1024),
            hidden_dim=config["model"].get("hidden_dim", 512),
            output_dim=config["model"].get("output_dim", 256),
            num_heads=config["model"].get("num_heads", 1),
            skip_connection=config["model"].get("skip_connection", True),
            dropout=config["model"].get("dropout", 0.1),
            use_context=config["model"].get("use_context", False),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def run_epoch(model, loader, taxonomy, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    leaf_indices = taxonomy.leaf_indices.to(device)
    levels = taxonomy.levels()

    for batch in loader:
        image = batch["image"].to(device)
        context = batch.get("context_image")
        if context is not None:
            context = context.to(device)
        leaf_target = batch["leaf_target"].to(device)
        node_target = batch["node_target"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(image, context)
            if logits.shape[1] == taxonomy.num_nodes:
                loss = hgnn_loss(logits, node_target, levels)
                acc = hgnn_leaf_accuracy(logits, node_target, leaf_indices)
            else:
                leaf_target = torch.searchsorted(leaf_indices, leaf_target)
                loss = torch.nn.functional.cross_entropy(logits, leaf_target)
                acc = leaf_accuracy(logits, leaf_target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = image.shape[0]
        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total += batch_size

    return total_loss / total, total_acc / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--taxonomy", default=None, help="Override taxonomy JSON from the config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.taxonomy:
        config["taxonomy"] = args.taxonomy

    seed = config["training"].get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, taxonomy = make_loader(config, "train")
    val_loader, _ = make_loader(config, "val")
    model = make_model(config, taxonomy).to(device)
    if config["model"]["type"] == "hgnn":
        # initialize taxonomy-node prototypes from average CNN embeddings
        model.init_prototypes(train_loader, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("lr", 1e-4),
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    out_dir = Path(config["training"].get("out_dir", "runs/patch_classifier"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    epochs = 1 if args.dry_run else config["training"].get("epochs", 10)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, taxonomy, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, taxonomy, optimizer, device, False)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "config": config, "best_acc": best_acc},
                out_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

