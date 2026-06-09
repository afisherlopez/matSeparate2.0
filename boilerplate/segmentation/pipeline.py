"""End-to-end patch-based material segmentation used for the MINC-S evaluation.

    sliding window -> classify patches -> bilinear upsample
        -> SLIC superpixel refine -> taxonomy-level projection
        -> argmax label map -> 8-connected components
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

from boilerplate.segmentation.classify import PatchClassifier
from boilerplate.segmentation.objects import (
    Instance,
    build_label_map,
    extract_instances,
)
from boilerplate.segmentation.refine import refine_crf, refine_slic
from boilerplate.segmentation.sampler import SlidingWindowSampler
from boilerplate.segmentation.taxonomy_cut import apply_frontier, build_frontier
from boilerplate.segmentation.upsample import upsample_probs


@dataclass
class SegmentationResult:
    label_map: np.ndarray
    frontier_names: List[str]
    instances: List[Instance]
    level: Union[str, int]
    image_size: tuple
    confidence: np.ndarray = field(repr=False)


def load_image_uint8(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(image.astype(np.uint8))


class MaterialSegmenter:
    def __init__(self, classifier: PatchClassifier, refine: str = "slic"):
        # "slic" is default
        # "crf" enables the dense-CRF backend, which we experimented with but excluded from the paper (too slow).
        if refine not in ("slic", "crf"):
            raise ValueError(f"refine must be 'slic' or 'crf', got {refine!r}")
        self.classifier = classifier
        self.refine = refine
        self.graph = classifier.taxonomy.graph
        self.leaf_names = classifier.leaf_names
        self.sampler = SlidingWindowSampler()

    def segment(self, image, level: Union[str, int] = "leaf") -> SegmentationResult:
        image_uint8 = load_image_uint8(image)
        h, w = image_uint8.shape[:2]

        p_grid = self.classifier.classify(self.sampler.sample(image_uint8))
        p_dense = upsample_probs(p_grid, target_hw=(h, w))
        if self.refine == "crf":
            refined = refine_crf(image_uint8, p_dense)
        else:
            refined = refine_slic(image_uint8, p_dense)

        cut = build_frontier(self.graph, self.leaf_names, level=level)
        frontier = apply_frontier(refined, cut)
        label_map, confidence = build_label_map(frontier)
        instances = extract_instances(label_map, confidence, cut.frontier_names)

        return SegmentationResult(
            label_map=label_map,
            frontier_names=cut.frontier_names,
            instances=instances,
            level=level,
            image_size=(h, w),
            confidence=confidence,
        )
