"""Sliding-window patch sampler.

Samples overlapping square windows on a regular grid. The main MINC-S
evaluation uses window size 128 px and stride 32 px. If the grid does not
exactly cover the image, the final window in each dimension is snapped to the
image boundary so the full extent is covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

WINDOW_SIZE = 128
STRIDE = 32


@dataclass
class SampleResult:
    patches: np.ndarray  # (N, ws, ws, 3) uint8
    grid_coords: List[Tuple[int, int]]
    grid_shape: Tuple[int, int]
    orig_shape: Tuple[int, int]


def _positions(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    pos = list(range(0, length - window + 1, stride))
    if pos[-1] != length - window:
        pos.append(length - window)  # snap last window to the edge
    return pos


class SlidingWindowSampler:
    def __init__(self, window_size: int = WINDOW_SIZE, stride: int = STRIDE,
                 pad_mode: str = "reflect"):
        self.window_size = window_size
        self.stride = stride
        self.pad_mode = pad_mode

    def sample(self, image: np.ndarray) -> SampleResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HWC RGB image, got {image.shape}")
        h, w = image.shape[:2]
        ws = self.window_size

        # pad up to at least one full window per dimension
        hp, wp = max(h, ws), max(w, ws)
        if (hp, wp) != (h, w):
            image = np.pad(image, ((0, hp - h), (0, wp - w), (0, 0)), mode=self.pad_mode)

        ys = _positions(hp, ws, self.stride)
        xs = _positions(wp, ws, self.stride)
        patches, coords = [], []
        for gi, y0 in enumerate(ys):
            for gj, x0 in enumerate(xs):
                patches.append(image[y0 : y0 + ws, x0 : x0 + ws, :])
                coords.append((gi, gj))

        return SampleResult(
            patches=np.stack(patches, axis=0),
            grid_coords=coords,
            grid_shape=(len(ys), len(xs)),
            orig_shape=(h, w),
        )
