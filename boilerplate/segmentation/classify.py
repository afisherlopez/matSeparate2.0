"""Patch classifier wrapper for the segmentation pipeline.

Wraps a trained learning model so the pipeline can turn a batch of sampled
windows into a coarse grid of leaf-material probabilities. Each window is
resized to 224x224 and ImageNet-normalized before inference, matching training.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchvision.transforms as T

from boilerplate.learning.models import HGNNPatchClassifier


class PatchClassifier:
    def __init__(self, model, taxonomy, device="cpu", batch_size: int = 32):
        self.model = model.to(device).eval()
        self.taxonomy = taxonomy
        self.device = device
        self.batch_size = batch_size
        self.leaf_names: List[str] = list(taxonomy.leaves)
        self.leaf_indices = taxonomy.leaf_indices.to(device)
        self.is_hgnn = isinstance(model, HGNNPatchClassifier)
        self.transform = T.Compose(
            [
                T.Resize(224, antialias=True),
                T.CenterCrop(224),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _prepare(self, patch: np.ndarray) -> torch.Tensor:
        arr = patch.astype(np.float32) / 255.0
        return self.transform(torch.from_numpy(arr).permute(2, 0, 1))

    @torch.inference_mode()
    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        out = np.empty((patches.shape[0], len(self.leaf_names)), dtype=np.float32)
        for start in range(0, patches.shape[0], self.batch_size):
            chunk = patches[start : start + self.batch_size]
            batch = torch.stack([self._prepare(p) for p in chunk]).to(self.device)
            logits = self.model(batch)
            if self.is_hgnn:
                logits = logits[:, self.leaf_indices]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            out[start : start + probs.shape[0]] = probs
        return out

    def classify(self, sample) -> np.ndarray:
        """Assemble per-patch leaf probabilities into a (gh, gw, L) grid."""
        gh, gw = sample.grid_shape
        probs = self.predict_leaf_probs(sample.patches)
        p_grid = np.zeros((gh, gw, len(self.leaf_names)), dtype=np.float32)
        for (gi, gj), row in zip(sample.grid_coords, probs):
            p_grid[gi, gj] = row
        return p_grid
