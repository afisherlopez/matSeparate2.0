from __future__ import annotations

from typing import Iterable, List, Protocol

import numpy as np

from segmentation.patches import SampleResult


def _progress(iterable: Iterable, total: int, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="batch")


class LeafProbPredictor(Protocol):

    leaf_names: List[str]

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        ...


class PatchClassifierPredictor:

    def __init__(self, api, batch_size: int = 32, show_progress: bool = True):
        self.api = api
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.leaf_names: List[str] = list(api.leaf_names)

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        n = patches.shape[0]
        out = np.empty((n, len(self.leaf_names)), dtype=np.float32)
        starts = list(range(0, n, self.batch_size))
        for start in _progress(
            starts, total=len(starts), desc="classifying patches", enabled=self.show_progress
        ):
            chunk = [patches[i] for i in range(start, min(start + self.batch_size, n))]
            results = self.api.infer_batch(chunk, return_probs=True, decode_path=False)
            for j, res in enumerate(results):
                out[start + j] = np.asarray(res["leaf_probs"], dtype=np.float32)
        return out


class StubLeafPredictor:

    def __init__(self, leaf_names: List[str], num_buckets: int = 6):
        self.leaf_names = list(leaf_names)
        n = len(self.leaf_names)
        step = max(1, n // max(1, num_buckets))
        self._target_idx = list(range(0, n, step))[:num_buckets] or [0]

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        n = patches.shape[0]
        out = np.full((n, len(self.leaf_names)), 1e-3, dtype=np.float32)
        for i in range(n):
            v = float(patches[i].mean()) / 255.0
            b = min(int(v * len(self._target_idx)), len(self._target_idx) - 1)
            out[i, self._target_idx[b]] = 5.0
        out /= out.sum(axis=1, keepdims=True)
        return out


class PatchClassifier:

    def __init__(self, predictor: LeafProbPredictor):
        self.predictor = predictor

    @property
    def leaf_names(self) -> List[str]:
        return list(self.predictor.leaf_names)

    @classmethod
    def from_patch_classifier(
        cls, api, batch_size: int = 32, show_progress: bool = True
    ) -> "PatchClassifier":
        return cls(PatchClassifierPredictor(api, batch_size=batch_size, show_progress=show_progress))

    def classify(self, sample: SampleResult) -> np.ndarray:
        
        gh, gw = sample.grid_shape
        num_leaves = len(self.leaf_names)
        probs = self.predictor.predict_leaf_probs(sample.patches) 
        if probs.shape[0] != len(sample.grid_coords):
            raise ValueError(
                f"predictor returned {probs.shape[0]} rows for "
                f"{len(sample.grid_coords)} patches"
            )
        p_grid = np.zeros((gh, gw, num_leaves), dtype=np.float32)
        for (gi, gj), row in zip(sample.grid_coords, probs):
            p_grid[gi, gj] = row
        return p_grid
