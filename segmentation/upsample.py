from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from __future__ import annotations


def upsample_probs(
    p_grid: np.ndarray,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = False,
    renormalize: bool = True,
) -> np.ndarray:

    if p_grid.ndim != 3:
        raise ValueError(f"p_grid must be (gh, gw, L), got {p_grid.shape}")
    h, w = target_hw
    gh, gw, num_leaves = p_grid.shape

    tensor = torch.from_numpy(np.ascontiguousarray(p_grid, dtype=np.float32))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  

    interp_kwargs = {"size": (h, w), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        interp_kwargs["align_corners"] = align_corners
    dense = F.interpolate(tensor, **interp_kwargs) 

    dense = dense.squeeze(0).permute(1, 2, 0).contiguous().numpy()  
    dense = np.clip(dense, 0.0, None)

    if renormalize:
        denom = dense.sum(axis=-1, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)
        dense = dense / denom

    return dense.astype(np.float32)
