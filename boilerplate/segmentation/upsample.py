"""Bilinear upsampling of the coarse patch-probability grid.

The classifier produces a coarse grid of leaf-material probabilities, one row
per sampled window. We bilinearly upsample it to full image resolution, clip to
be nonnegative, and renormalize so every pixel holds a valid distribution.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def upsample_probs(p_grid: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if p_grid.ndim != 3:
        raise ValueError(f"p_grid must be (gh, gw, L), got {p_grid.shape}")
    h, w = target_hw

    tensor = torch.from_numpy(np.ascontiguousarray(p_grid, dtype=np.float32))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    dense = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
    dense = dense.squeeze(0).permute(1, 2, 0).contiguous().numpy()

    dense = np.clip(dense, 0.0, None)
    denom = dense.sum(axis=-1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return (dense / denom).astype(np.float32)
