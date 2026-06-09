#!/usr/bin/env python3
"""Evaluate segmentation components against MINC-S binary segment masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from segmentation.config import SegmentationConfig  # noqa: E402
from segmentation.pipeline import MaterialMerger  # noqa: E402


DEFAULT_MATADOR_TO_MINC = {
    "brick": "brick",
    "carpet": "carpet",
    "pottery": "ceramic",
    "natural_fiber": "fabric",
    "wool": "fabric",
    "satin": "fabric",
    "nylon": "fabric",
    "suede": "fabric",
    "foliage": "foliage",
    "grass": "foliage",
    "ivy": "foliage",
    "moss": "foliage",
    "bread": "food",
    "fruit": "food",
    "vegetable": "food",
    "leather": "leather",
    "generic_metal": "metal",
    "paper": "paper",
    "foam": "plastic",
    "wax": "plastic",
    "marble": "polishedstone",
    "granite": "stone",
    "limestone": "stone",
    "shale": "stone",
    "gravel": "stone",
    "sand": "stone",
    "timber": "wood",
    "tree_bark": "wood",
}


def read_categories(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_segments(dataset_root: Path, segments_txt: Path, categories: list[str], max_images=None):
    photos_dir = dataset_root / "photos"
    masks_dir = dataset_root / "segments"
    grouped = defaultdict(list)
    missing = {"photos": 0, "masks": 0, "bad_rows": 0}
    for line in segments_txt.read_text().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            missing["bad_rows"] += 1
            continue
        label_idx, photo_id, shape_id = parts
        try:
            label_idx = int(label_idx)
        except ValueError:
            missing["bad_rows"] += 1
            continue
        photo_path = photos_dir / f"{photo_id}.jpg"
        mask_path = masks_dir / f"{photo_id}_{shape_id}.png"
        if not photo_path.exists():
            missing["photos"] += 1
            continue
        if not mask_path.exists():
            missing["masks"] += 1
            continue
        grouped[photo_id].append(
            {
                "label_idx": label_idx,
                "label_name": categories[label_idx],
                "photo_id": photo_id,
                "shape_id": shape_id,
                "photo_path": photo_path,
                "mask_path": mask_path,
            }
        )
    items = list(sorted(grouped.items()))
    if max_images is not None:
        items = items[:max_images]
    return items, missing


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def resize_photo(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def iou(pred, gt) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 1.0


def dice(pred, gt) -> float:
    denom = pred.sum() + gt.sum()
    return float(2 * np.logical_and(pred, gt).sum() / denom) if denom else 1.0


def best_component(gt_mask, instances):
    best = {
        "best_iou": 0.0,
        "best_dice": 0.0,
        "matched_label": "",
        "matched_score": 0.0,
        "matched_area": 0,
    }
    for inst in instances:
        score = iou(inst.mask, gt_mask)
        if score > best["best_iou"]:
            best = {
                "best_iou": score,
                "best_dice": dice(inst.mask, gt_mask),
                "matched_label": inst.material,
                "matched_score": float(inst.score),
                "matched_area": int(inst.area),
            }
    return best


def make_config(args) -> SegmentationConfig:
    cfg = SegmentationConfig.from_yaml(args.config) if args.config else SegmentationConfig()
    cfg.classifier = args.classifier
    cfg.run_dir = str(args.run_dir) if args.run_dir else cfg.run_dir
    cfg.checkpoint = str(args.checkpoint) if args.checkpoint else cfg.checkpoint
    cfg.device = args.device
    cfg.sampling.type = args.sampler
    cfg.sampling.batch_size = args.batch_size
    cfg.sampling.window_size = args.window_size
    cfg.sampling.stride = args.stride
    cfg.crf.backend = args.crf_backend
    cfg.level.target = args.level
    cfg.objects.bg_threshold = args.bg_threshold
    cfg.objects.min_object_area = args.min_object_area
    cfg.output.write_color_viz = False
    cfg.output.write_instance_pngs = False
    return cfg


def build_merger(cfg: SegmentationConfig):
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    return MaterialMerger.from_patch_classifier(cfg.checkpoint, config=cfg, device=cfg.device)


def build_stub_merger(cfg: SegmentationConfig):
    from segmentation.classify import PatchClassifier, StubLeafPredictor
    from taxonomy.tree import get_taxonomy

    graph = get_taxonomy(str(repo_root / "taxonomy" / "assets" / "matador-c1-taxonomy.json"))
    leaves = sorted(node for node in graph.nodes() if graph.out_degree(node) == 0)
    if cfg.crf.backend == "dense":
        cfg.crf.backend = "none"
    return MaterialMerger(PatchClassifier(StubLeafPredictor(leaves)), graph=graph, config=cfg)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(segment_rows, image_rows, crosswalk):
    recalls = {
        f"recall@{t:.2f}": float(np.mean([r["best_iou"] >= t for r in segment_rows]))
        if segment_rows else 0.0
        for t in (0.25, 0.50, 0.75)
    }
    mapped = [r for r in segment_rows if r["gt_mapped"] and r["matched_label"]]
    return {
        "num_images": len(image_rows),
        "num_segments": len(segment_rows),
        "mean_best_iou": float(np.mean([r["best_iou"] for r in segment_rows])) if segment_rows else 0.0,
        "mean_best_dice": float(np.mean([r["best_dice"] for r in segment_rows])) if segment_rows else 0.0,
        "mean_components_per_image": float(np.mean([r["num_components"] for r in image_rows])) if image_rows else 0.0,
        "mean_sec_per_image": float(np.mean([r["seconds"] for r in image_rows])) if image_rows else 0.0,
        "mapped_semantic_accuracy": float(
            np.mean([crosswalk.get(r["matched_label"], "") == r["label_name"] for r in mapped])
        ) if mapped else 0.0,
        **recalls,
    }


def evaluate(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = read_categories(args.dataset_root / "categories.txt")
    segments_txt = args.segments_txt or args.dataset_root / "test-segments.txt"
    photos, missing = read_segments(args.dataset_root, segments_txt, categories, args.max_images)
    crosswalk = json.loads(args.crosswalk.read_text()) if args.crosswalk else (
        DEFAULT_MATADOR_TO_MINC if args.default_crosswalk else {}
    )
    gt_mapped = set(crosswalk.values())
    cfg = make_config(args)
    merger = build_stub_merger(cfg) if args.stub else build_merger(cfg)

    image_rows, segment_rows = [], []
    for idx, (photo_id, samples) in enumerate(photos, start=1):
        by_shape = defaultdict(list)
        for sample in samples:
            mask = load_mask(sample["mask_path"])
            by_shape[mask.shape].append((sample, mask))
        for shape, shape_samples in by_shape.items():
            photo = resize_photo(shape_samples[0][0]["photo_path"], shape)
            start = time.perf_counter()
            result = merger.segment(photo, level=args.level)
            seconds = time.perf_counter() - start
            for sample, gt_mask in shape_samples:
                if gt_mask.sum() == 0:
                    continue
                segment_rows.append(
                    {
                        **{k: sample[k] for k in ("photo_id", "shape_id", "label_idx", "label_name")},
                        **best_component(gt_mask, result.instances),
                        "gt_area": int(gt_mask.sum()),
                        "gt_mapped": sample["label_name"] in gt_mapped,
                    }
                )
            image_rows.append(
                {
                    "photo_id": photo_id,
                    "image_index": idx,
                    "num_components": len(result.instances),
                    "num_gt_segments": len(shape_samples),
                    "seconds": seconds,
                }
            )
        if args.progress_every and idx % args.progress_every == 0:
            print(f"processed {idx}/{len(photos)} images", flush=True)

    metrics = summarize(segment_rows, image_rows, crosswalk)
    payload = {
        "classifier": cfg.classifier,
        "run_dir": cfg.run_dir,
        "checkpoint": cfg.checkpoint,
        "dataset_root": str(args.dataset_root),
        "segments_txt": str(segments_txt),
        "missing": missing,
        "metrics": metrics,
    }
    write_csv(args.output_dir / "per_image_metrics.csv", image_rows)
    write_csv(args.output_dir / "per_segment_metrics.csv", segment_rows)
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def parse_level(value):
    if str(value).lower() == "leaf":
        return "leaf"
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/minc/minc-s"))
    parser.add_argument("--segments-txt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--classifier", choices=["patch_classifier"], default="patch_classifier")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stub", action="store_true", help="use a fake classifier to test eval plumbing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sampler", choices=["grid", "sliding"], default="sliding")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--crf-backend", choices=["dense", "superpixel", "none"], default="superpixel")
    parser.add_argument("--level", type=parse_level, default="leaf")
    parser.add_argument("--bg-threshold", type=float, default=0.0)
    parser.add_argument("--min-object-area", type=int, default=64)
    parser.add_argument("--crosswalk", type=Path, default=None)
    parser.add_argument("--default-crosswalk", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    payload = evaluate(args)
    metrics = payload["metrics"]
    print(f"Images: {metrics['num_images']}")
    print(f"Segments: {metrics['num_segments']}")
    print(f"Mean best IoU: {metrics['mean_best_iou']:.4f}")
    print(f"Recall@0.50: {metrics['recall@0.50']:.4f}")
    print(f"Mean sec/image: {metrics['mean_sec_per_image']:.3f}")
    print(f"Mapped semantic accuracy: {metrics['mapped_semantic_accuracy']:.4f}")


if __name__ == "__main__":
    main()
