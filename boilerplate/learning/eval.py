"""Evaluate a patch classifier on Matador-C1 (fixed boilerplate eval).

Reports top-1 accuracy at each taxonomy level: state,
composition, form, and material (leaf). For HGNN models the internal-level
predictions are read directly from the taxonomy-node logits. For flat ResNet
models the predicted leaf label is mapped up to its ancestors.
"""

import argparse

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from boilerplate.learning.data import PatchDataset
from boilerplate.learning.taxonomy import load_taxonomy
from boilerplate.learning.train import (
    IMAGE_ROOT,
    IMAGE_SIZE,
    VAL_CSV,
    collate,
    make_model,
)

# taxonomy depth -> reported level name (root is depth 0)
LEVEL_NAMES = {1: "state", 2: "composition", 3: "form", 4: "material"}


def ancestor_at_depth(taxonomy, leaf, depth):
    path = taxonomy.path_to(leaf)
    return path[depth] if depth < len(path) else leaf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]

    taxonomy = load_taxonomy("taxonomy/assets/matador-c1-taxonomy.json")
    model = make_model(model_name, taxonomy)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    transform = T.Compose(
        [
            T.Resize(IMAGE_SIZE, antialias=True),
            T.CenterCrop(IMAGE_SIZE),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    use_context = model_name in ("global_resnet", "global_hgnn")
    ds = PatchDataset(
        VAL_CSV, taxonomy, IMAGE_ROOT,
        context_col="context_path" if use_context else None,
        transform=transform,
    )
    loader = DataLoader(ds, batch_size=64, collate_fn=collate)

    leaf_indices = taxonomy.leaf_indices.to(device)
    leaf_names = taxonomy.leaves
    correct = {name: 0 for name in LEVEL_NAMES.values()}
    total = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            context = batch.get("context_image")
            context = context.to(device) if context is not None else None
            logits = model(image, context)

            if logits.shape[1] == taxonomy.num_nodes:
                leaf_pos = logits[:, leaf_indices].argmax(dim=1).cpu()
            else:
                leaf_pos = logits.argmax(dim=1).cpu()
            preds = [leaf_names[i] for i in leaf_pos]

            # ground-truth leaf names, recomputed from the leaf-target index
            gt_leaf_pos = torch.searchsorted(leaf_indices.cpu(), batch["leaf_target"])
            gts = [leaf_names[i] for i in gt_leaf_pos]

            for pred, gt in zip(preds, gts):
                for depth, name in LEVEL_NAMES.items():
                    if ancestor_at_depth(taxonomy, pred, depth) == ancestor_at_depth(
                        taxonomy, gt, depth
                    ):
                        correct[name] += 1
                total += 1

    for name in LEVEL_NAMES.values():
        print(f"{name}_accuracy={correct[name] / total:.4f}")


if __name__ == "__main__":
    main()
