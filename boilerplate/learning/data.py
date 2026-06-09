"""Matador-C1 patch dataset.

Each manifest row points at a local material crop and (optionally) a global
context image. Images are loaded as RGB float tensors in [0, 1]; the training
transform handles resize/center-crop to 224 and ImageNet normalization.

This is the fixed boilerplate loader: a CSV manifest read from an extracted
image directory. (Tar-backed loading, caching, and augmentation are agentic
extensions; see PROMPTS.md.)
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from boilerplate.learning.taxonomy import Taxonomy


class PatchDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        taxonomy: Taxonomy,
        image_root: str | Path,
        label_col: str = "c1_label",
        image_col: str = "image_path",
        context_col: Optional[str] = None,
        transform=None,
    ):
        self.taxonomy = taxonomy
        self.image_root = Path(image_root)
        self.label_col = label_col
        self.image_col = image_col
        self.context_col = context_col
        self.transform = transform
        with open(manifest_csv, newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        arr = np.array(Image.open(self.image_root / rel_path).convert("RGB"))
        arr = arr.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        label = row[self.label_col]

        image = self._load_image(row[self.image_col])
        if self.transform:
            image = self.transform(image)

        item = {
            "image": image,
            "label": label,
            "leaf_target": torch.tensor(
                self.taxonomy.node_to_idx[label], dtype=torch.long
            ),
            "node_target": self.taxonomy.multihot_path(label),
        }

        if self.context_col and row.get(self.context_col):
            context = self._load_image(row[self.context_col])
            if self.transform:
                context = self.transform(context)
            item["context_image"] = context

        return item
