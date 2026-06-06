import io
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

from taxonomy.tree import get_taxonomy


class MatadorC1Dataset(Dataset):
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        taxonomy_json: Union[str, Path],
        appearance_tar: Optional[Union[str, Path]] = None,
        extracted_root: Optional[Union[str, Path]] = None,
        node_index_json: Optional[Union[str, Path]] = None,
        transform=None,
    ):
        if appearance_tar is None and extracted_root is None:
            raise ValueError("Must provide either appearance_tar or extracted_root")
        if appearance_tar is not None and extracted_root is not None:
            raise ValueError("Provide only one of appearance_tar or extracted_root")

        self.manifest_csv = Path(manifest_csv)
        self.appearance_tar = Path(appearance_tar) if appearance_tar else None
        self.extracted_root = Path(extracted_root) if extracted_root else None
        self.transform = transform

        self.samples = self._load_manifest(self.manifest_csv)

        self.taxonomy = get_taxonomy(taxonomy_json)
        self.nodes = sorted(self.taxonomy.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = self.nodes
        self.num_nodes = len(self.nodes)

        if node_index_json is None:
            node_index_json = self.manifest_csv.parent / "node_index.json"
        self.node_index_json = Path(node_index_json)
        self._persist_node_index()

        self.root_name = "root"
        self.c1_label_to_path = {}
        for c1_label in {s["c1_label"] for s in self.samples}:
            try:
                path = nx.shortest_path(self.taxonomy, self.root_name, c1_label)
                self.c1_label_to_path[c1_label] = path
            except nx.NetworkXNoPath:
                raise ValueError(f"No path from '{self.root_name}' to '{c1_label}' in taxonomy")

        self._tar = None
        if self.appearance_tar:
            self._tar = tarfile.open(self.appearance_tar, "r:*")

    def _load_manifest(self, path: Path) -> List[Dict]:
        import csv
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def _persist_node_index(self):
        if self.node_index_json.exists():
            with open(self.node_index_json) as f:
                saved = json.load(f)
            if saved["idx_to_node"] != self.nodes:
                raise ValueError(
                    f"node_index.json ordering mismatch.\n"
                    f"Saved:   {saved['idx_to_node']}\n"
                    f"Current: {self.nodes}"
                )
            return
        self.node_index_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.node_index_json, "w") as f:
            json.dump(
                {"node_to_idx": self.node_to_idx, "idx_to_node": self.idx_to_node},
                f,
                indent=2,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> torch.Tensor:
        if self._tar is not None:
            member = self._tar.getmember(image_path)
            f = self._tar.extractfile(member)
            if f is None:
                raise ValueError(f"Cannot extract tar member: {image_path}")
            buf = io.BytesIO(f.read())
            arr = tifffile.imread(buf)
        else:
            arr = tifffile.imread(str(self.extracted_root / image_path))

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[1] == arr.shape[2]:
                arr = arr.transpose(1, 2, 0)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
        else:
            raise ValueError(f"Unexpected image ndim={arr.ndim}, shape={arr.shape}")

        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 65535.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, idx: int) -> Dict:
        row = self.samples[idx]
        c1_label = row["c1_label"]
        taxa = self.c1_label_to_path[c1_label]

        target_multihot = torch.zeros(self.num_nodes, dtype=torch.float32)
        for node in taxa:
            target_multihot[self.node_to_idx[node]] = 1.0

        return {
            "image": self._load_image(row["image_path"]),
            "sample_id": row["sample_id"],
            "material_label": row["material_label"],
            "c1_label": c1_label,
            "taxa": taxa,
            "target_multihot": target_multihot,
            "target_indices": torch.tensor(
                [self.node_to_idx[n] for n in taxa], dtype=torch.long
            ),
            "image_path": row["image_path"],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.appearance_tar is not None:
            self._tar = tarfile.open(self.appearance_tar, "r:*")

    def __del__(self):
        if self._tar is not None:
            self._tar.close()
