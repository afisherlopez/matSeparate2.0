"""Segmentation evaluation metrics reported in the paper.

Only the metrics actually reported are implemented here:

  geometric  : mean best IoU, Recall@0.50, mean components/image, mean sec/image
  semantic   : mapped semantic accuracy (patch-based pipelines only)

For each ground-truth MINC-S segment we keep the predicted component with the
highest IoU (the same best-match protocol used for the SAM baseline). SAM is
included only in geometric metrics because its automatic masks are unlabeled.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 0.0


def best_component(gt_mask: np.ndarray, instances) -> dict:
    """Highest-IoU predicted component for one ground-truth segment."""
    best = {"best_iou": 0.0, "matched_label": ""}
    for inst in instances:
        score = iou(inst.mask, gt_mask)
        if score > best["best_iou"]:
            best = {"best_iou": score, "matched_label": inst.material}
    return best


def summarize(segment_rows: List[dict], image_rows: List[dict],
              crosswalk: Optional[Dict[str, str]] = None) -> dict:
    """Aggregate per-segment / per-image rows into the reported metrics.

    Each segment row: {"best_iou", "matched_label", "gt_label", "gt_mapped"}.
    Each image row:   {"num_components", "seconds"}.
    crosswalk maps a Matador-C1 material label to its MINC-S class.
    """
    crosswalk = crosswalk or {}
    mapped = [r for r in segment_rows if r.get("gt_mapped") and r.get("matched_label")]

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "num_images": len(image_rows),
        "num_segments": len(segment_rows),
        "mean_best_iou": mean([r["best_iou"] for r in segment_rows]),
        "recall@0.50": mean([r["best_iou"] >= 0.50 for r in segment_rows]),
        "mean_components_per_image": mean([r["num_components"] for r in image_rows]),
        "mean_sec_per_image": mean([r["seconds"] for r in image_rows]),
        "mapped_semantic_accuracy": mean(
            [crosswalk.get(r["matched_label"], "") == r["gt_label"] for r in mapped]
        ),
    }
