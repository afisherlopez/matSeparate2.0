import argparse

import torch

from patch_classifier.losses import hgnn_leaf_accuracy, leaf_accuracy
from patch_classifier.train import make_loader, make_model
from patch_classifier.taxonomy import load_taxonomy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--taxonomy", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    if args.taxonomy:
        config["taxonomy"] = args.taxonomy

    loader, taxonomy = make_loader(config, args.split)
    taxonomy = load_taxonomy(config["taxonomy"], root=config.get("root", "root"))
    model = make_model(config, taxonomy)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    total_acc = 0.0
    total = 0
    leaf_indices = taxonomy.leaf_indices.to(device)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            context = batch.get("context_image")
            if context is not None:
                context = context.to(device)
            logits = model(image, context)
            if logits.shape[1] == taxonomy.num_nodes:
                acc = hgnn_leaf_accuracy(logits, batch["node_target"].to(device), leaf_indices)
            else:
                target = torch.searchsorted(leaf_indices, batch["leaf_target"].to(device))
                acc = leaf_accuracy(logits, target)
            total_acc += acc * image.shape[0]
            total += image.shape[0]

    print(f"{args.split}_leaf_accuracy={total_acc / total:.4f}")


if __name__ == "__main__":
    main()

