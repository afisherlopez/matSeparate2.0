from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from PIL import Image

from segmentation import crf
from segmentation.classify import PatchClassifier
from segmentation.config import SegmentationConfig
from segmentation.formats import save_results
from segmentation.objects import Instance, build_label_map, extract_instances
from segmentation.patches import build_sampler
from segmentation.taxonomy_cut import apply_frontier, build_frontier
from segmentation.upsample import upsample_probs


def load_image_uint8(image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
    raise TypeError(f"Unsupported image type: {type(image)}")


@dataclass
class SegmentationResult:
    label_map: np.ndarray
    frontier_names: List[str]
    instances: List[Instance]
    level: Union[str, int]
    image_size: tuple
    confidence: np.ndarray = field(repr=False)
    refined_leaf_probs: np.ndarray = field(repr=False)
    image_uint8: np.ndarray = field(repr=False)
    meta: Dict = field(default_factory=dict)
    _merger: Optional["MaterialMerger"] = field(default=None, repr=False)

    def recut(self, level: Union[str, int]) -> "SegmentationResult":
        if self._merger is None:
            raise RuntimeError("result is not attached to a MaterialMerger")
        return self._merger.recut(self, level)

    def save(self, out_dir: Union[str, Path]) -> Dict[str, str]:
        cfg = self._merger.config if self._merger else None
        write_color = cfg.output.write_color_viz if cfg else True
        write_pngs = cfg.output.write_instance_pngs if cfg else False
        return save_results(
            out_dir=out_dir,
            label_map=self.label_map,
            frontier_names=self.frontier_names,
            instances=self.instances,
            level=self.level,
            image_size=self.image_size,
            meta=self.meta,
            write_color_viz=write_color,
            write_instance_pngs=write_pngs,
        )


class MaterialMerger:
    def __init__(
        self,
        classifier: PatchClassifier,
        graph: nx.DiGraph,
        config: Optional[SegmentationConfig] = None,
    ):
        self.classifier = classifier
        self.graph = graph
        self.config = config or SegmentationConfig()
        self.sampler = build_sampler(self.config.sampling)
        self.leaf_names = classifier.leaf_names

    @classmethod
    def from_patch_classifier(
        cls,
        checkpoint: Union[str, Path],
        config: Optional[SegmentationConfig] = None,
        device: str = "auto",
    ) -> "MaterialMerger":
        import sys

        seg_root = Path(__file__).resolve().parent.parent
        repo_root = seg_root.parent
        for path in (repo_root, seg_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from classifier.infer import PatchClassifierInference

        config = config or SegmentationConfig()
        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            api = PatchClassifierInference.from_run_dir(checkpoint, device=device)
        else:
            api = PatchClassifierInference.from_checkpoint(checkpoint, device=device)
        classifier = PatchClassifier.from_patch_classifier(
            api, batch_size=config.sampling.batch_size
        )
        return cls(classifier=classifier, graph=api.taxonomy.graph, config=config)

    def segment(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        level: Optional[Union[str, int]] = None,
    ) -> SegmentationResult:
        if level is None:
            level = self.config.level.target

        image_uint8 = load_image_uint8(image)
        h, w = image_uint8.shape[:2]

        p_grid = self.classifier.classify(self.sampler.sample(image_uint8))

        p_dense = upsample_probs(
            p_grid,
            target_hw=(h, w),
            mode=self.config.upsample.mode,
            align_corners=self.config.upsample.align_corners,
            renormalize=self.config.upsample.renormalize,
        )
        refined = crf.refine(
            image_uint8,
            p_dense,
            self.config.crf,
            leaf_names=self.leaf_names,
            graph=self.graph,
        )

        return self._finish(refined, image_uint8, level)

    def recut(
        self, result: SegmentationResult, level: Union[str, int]
    ) -> SegmentationResult:
        return self._finish(result.refined_leaf_probs, result.image_uint8, level)

    def _finish(
        self,
        refined_leaf_probs: np.ndarray,
        image_uint8: np.ndarray,
        level: Union[str, int],
    ) -> SegmentationResult:
        cut = build_frontier(
            self.graph,
            self.leaf_names,
            level=level,
            shallow_leaf=self.config.level.shallow_leaf,
        )
        frontier = apply_frontier(refined_leaf_probs, cut)
        label_map, confidence = build_label_map(
            frontier, bg_threshold=self.config.objects.bg_threshold
        )
        instances = extract_instances(
            label_map,
            confidence,
            cut.frontier_names,
            connectivity=self.config.objects.connectivity,
            min_object_area=self.config.objects.min_object_area,
            morph_close=self.config.objects.morph_close,
            morph_close_radius=self.config.objects.morph_close_radius,
        )
        h, w = image_uint8.shape[:2]
        meta = {
            "level": level,
            "bg_threshold": self.config.objects.bg_threshold,
            "num_classes": len(cut.frontier_names),
            "num_objects": len(instances),
            "crf_backend": self.config.crf.backend,
        }
        return SegmentationResult(
            label_map=label_map,
            frontier_names=cut.frontier_names,
            instances=instances,
            level=level,
            image_size=(h, w),
            confidence=confidence,
            refined_leaf_probs=refined_leaf_probs,
            image_uint8=image_uint8,
            meta=meta,
            _merger=self,
        )
