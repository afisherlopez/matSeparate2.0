from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SampleResult:
    patches: np.ndarray
    grid_coords: List[Tuple[int, int]]
    grid_shape: Tuple[int, int]
    orig_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    window_size: int = -1
    stride: int = -1

    @property
    def num_patches(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]


class PatchSampler:
    def sample(self, image: np.ndarray) -> SampleResult:
        raise NotImplementedError


class GridSampler(PatchSampler):
    def __init__(self, patch_size: int = 224, pad_mode: str = "reflect"):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.pad_mode = pad_mode

    def sample(self, image: np.ndarray) -> SampleResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        h, w = image.shape[:2]
        p = self.patch_size
        gh = math.ceil(h / p)
        gw = math.ceil(w / p)
        hp, wp = gh * p, gw * p

        pad_h = hp - h
        pad_w = wp - w
        if pad_h or pad_w:
            # reflect needs the pad to be < dimension; fall back to edge for tiny images
            mode = self.pad_mode
            if mode == "reflect" and (pad_h >= h or pad_w >= w):
                mode = "edge"
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        else:
            padded = image

        patches = []
        grid_coords = []
        for gi in range(gh):
            for gj in range(gw):
                y0, x0 = gi * p, gj * p
                tile = padded[y0 : y0 + p, x0 : x0 + p, :]
                patches.append(tile)
                grid_coords.append((gi, gj))

        patches_arr = np.stack(patches, axis=0) if patches else np.empty((0, p, p, 3))
        return SampleResult(
            patches=patches_arr,
            grid_coords=grid_coords,
            grid_shape=(gh, gw),
            orig_shape=(h, w),
            padded_shape=(hp, wp),
        )


def _positions(length: int, window: int, stride: int) -> List[int]:
    # last window snapped to edge so the full extent is covered
    if length <= window:
        return [0]
    stride = max(1, stride)
    pos = list(range(0, length - window + 1, stride))
    if pos[-1] != length - window:
        pos.append(length - window)
    return pos


def _count(length: int, window: int, stride: int) -> int:
    if length <= window:
        return 1
    stride = max(1, stride)
    n = len(range(0, length - window + 1, stride))
    if (n - 1) * stride != length - window:
        n += 1
    return n


_MIN_WINDOW = 16  # never shrink a window below this when chasing min_patches


def _resolve_window_stride(
    h: int,
    w: int,
    window: int,
    stride: int,
    min_patches: Optional[int],
    max_patches: Optional[int],
) -> Tuple[int, int]:
    if min_patches is None and max_patches is None:
        return window, stride if (stride and stride > 0) else max(1, window // 2)

    if min_patches is not None and max_patches is not None and min_patches > max_patches:
        raise ValueError(f"min_patches ({min_patches}) > max_patches ({max_patches})")

    window = max(1, min(window, h, w))
    stride = stride if (stride and stride > 0) else max(1, window // 2)
    s_cap = max(h, w)  # any larger stride yields a single position per dim

    def total(ws: int, s: int) -> int:
        return _count(h, ws, s) * _count(w, ws, s)

    if max_patches is not None:
        while total(window, stride) > max_patches and stride < s_cap:
            stride += 1

    if min_patches is not None:
        while total(window, stride) < min_patches and stride > 1:
            stride -= 1
        # still short at stride 1: shrink the window so more positions fit
        while total(window, stride) < min_patches and window > _MIN_WINDOW:
            window -= 1
        # shrinking the window can overshoot the cap; re-coarsen if needed
        if max_patches is not None:
            while total(window, stride) > max_patches and stride < s_cap:
                stride += 1

    return window, stride


class SlidingWindowSampler(PatchSampler):
    def __init__(
        self,
        window_size: int = 128,
        stride: int = 32,
        pad_mode: str = "reflect",
        min_patches: Optional[int] = None,
        max_patches: Optional[int] = None,
        window_area_pct: Optional[float] = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_area_pct is not None and window_area_pct <= 0:
            raise ValueError("window_area_pct must be positive")
        self.window_size = window_size
        self.window_area_pct = window_area_pct
        self._stride_arg = stride
        self.stride = stride if stride and stride > 0 else max(1, window_size // 2)
        self.pad_mode = pad_mode
        self.min_patches = min_patches
        self.max_patches = max_patches

    def _pad(self, image: np.ndarray, hp: int, wp: int) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h, pad_w = hp - h, wp - w
        if not (pad_h or pad_w):
            return image
        mode = self.pad_mode
        if mode == "reflect" and (pad_h >= h or pad_w >= w):
            mode = "edge"
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)

    def _base_window(self, h: int, w: int) -> int:
        if self.window_area_pct is not None:
            ws = int(round(math.sqrt(self.window_area_pct / 100.0 * h * w)))
            return max(1, ws)
        return self.window_size

    def _grid(self, h: int, w: int):
        base_window = self._base_window(h, w)
        base_stride = (
            self._stride_arg
            if (self._stride_arg and self._stride_arg > 0)
            else max(1, base_window // 2)
        )
        ws, stride = _resolve_window_stride(
            h, w, base_window, base_stride, self.min_patches, self.max_patches
        )
        hp, wp = max(h, ws), max(w, ws)
        ys = _positions(hp, ws, stride)
        xs = _positions(wp, ws, stride)
        return ws, stride, hp, wp, ys, xs

    def sample(self, image: np.ndarray) -> SampleResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        h, w = image.shape[:2]
        ws, stride, hp, wp, ys, xs = self._grid(h, w)
        padded = self._pad(image, hp, wp)

        patches, grid_coords = [], []
        for gi, y0 in enumerate(ys):
            for gj, x0 in enumerate(xs):
                patches.append(padded[y0 : y0 + ws, x0 : x0 + ws, :])
                grid_coords.append((gi, gj))

        patches_arr = np.stack(patches, axis=0) if patches else np.empty((0, ws, ws, 3))
        if self.min_patches is not None or self.max_patches is not None:
            print(
                f"[sampler] {len(ys)}x{len(xs)} = {len(ys) * len(xs)} patches "
                f"(window={ws}px, stride={stride}px) for {h}x{w} image"
            )
        return SampleResult(
            patches=patches_arr,
            grid_coords=grid_coords,
            grid_shape=(len(ys), len(xs)),
            orig_shape=(h, w),
            padded_shape=(hp, wp),
            window_size=ws,
            stride=stride,
        )

def build_sampler(sampling_config) -> PatchSampler:
    if sampling_config.type == "grid":
        return GridSampler(
            patch_size=sampling_config.patch_size,
            pad_mode=sampling_config.pad_mode,
        )
    if sampling_config.type == "sliding":
        return SlidingWindowSampler(
            window_size=sampling_config.window_size,
            stride=sampling_config.stride,
            pad_mode=sampling_config.pad_mode,
            min_patches=getattr(sampling_config, "min_patches", None),
            max_patches=getattr(sampling_config, "max_patches", None),
            window_area_pct=getattr(sampling_config, "window_area_pct", None),
        )
    raise NotImplementedError(f"Unknown sampler type '{sampling_config.type}'")
