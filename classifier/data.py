import csv
import io
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

from classifier.taxonomy import Taxonomy


class PatchDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        taxonomy: Taxonomy,
        image_root: Optional[str | Path] = None,
        image_tar: Optional[str | Path] = None,
        label_col: str = "c1_label",
        image_col: str = "image_path",
        context_col: Optional[str] = None,
        transform=None,
        context_transform=None,
    ):
        if image_root is None and image_tar is None:
            raise ValueError("Provide image_root or image_tar")
        if image_root is not None and image_tar is not None:
            raise ValueError("Use only one of image_root or image_tar")

        self.manifest_csv = Path(manifest_csv)
        self.taxonomy = taxonomy
        self.image_root = Path(image_root) if image_root else None
        self.image_tar = Path(image_tar) if image_tar else None
        self.label_col = label_col
        self.image_col = image_col
        self.context_col = context_col
        self.transform = transform
        self.context_transform = context_transform or transform
        self.rows = self._load_rows()
        self._tar = tarfile.open(self.image_tar, "r:*") if self.image_tar else None

    def _load_rows(self):
        with open(self.manifest_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        required = {self.label_col, self.image_col}
        missing = required - set(rows[0].keys()) if rows else required
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        label = row[self.label_col]
        image = self._load_image(row[self.image_col])
        if self.transform:
            image = self.transform(image)

        item = {
            "image": image,
            "label": label,
            "node_target": self.taxonomy.multihot_path(label),
            "leaf_target": torch.tensor(self.taxonomy.node_to_idx[label], dtype=torch.long),
        }

        if self.context_col and row.get(self.context_col):
            context_image = self._load_image(row[self.context_col])
            if self.context_transform:
                context_image = self.context_transform(context_image)
            item["context_image"] = context_image

        return item

    def _load_image(self, rel_path: str) -> torch.Tensor:
        if self._tar is not None:
            member = self._tar.getmember(rel_path)
            f = self._tar.extractfile(member)
            if f is None:
                raise ValueError(f"Could not read {rel_path} from {self.image_tar}")
            data = io.BytesIO(f.read())
            suffix = Path(rel_path).suffix.lower()
        else:
            path = self.image_root / rel_path
            data = path
            suffix = path.suffix.lower()

        if suffix in {".tif", ".tiff"}:
            arr = tifffile.imread(data)
        else:
            arr = np.array(Image.open(data).convert("RGB"))

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)

        arr = arr.astype(np.float32)
        arr /= 65535.0 if arr.max() > 255 else 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tar = tarfile.open(self.image_tar, "r:*") if self.image_tar else None

    def __del__(self):
        if getattr(self, "_tar", None) is not None:
            self._tar.close()

