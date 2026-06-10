from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from segmentation.formats import build_palette
from segmentation.objects import BACKGROUND_ID


def semantic_overlay(
    image_uint8: np.ndarray,
    label_map: np.ndarray,
    num_classes: int,
    alpha: float = 0.5,
) -> np.ndarray:
    palette = build_palette(num_classes)
    color = palette[np.clip(label_map, 0, num_classes)].astype(np.float32)
    base = image_uint8.astype(np.float32)
    blended = (1 - alpha) * base + alpha * color
    mask = (label_map != BACKGROUND_ID)[..., None]
    out = np.where(mask, blended, base)
    return np.clip(out, 0, 255).astype(np.uint8)


def instance_overlay(
    image_uint8: np.ndarray,
    instances,
    alpha: float = 0.55,
) -> np.ndarray:
    base = image_uint8.astype(np.float32)
    out = base.copy()
    palette = build_palette(max(len(instances), 1))
    for k, inst in enumerate(instances, start=1):
        color = palette[k].astype(np.float32)
        m = inst.mask[..., None]
        out = np.where(m, (1 - alpha) * out + alpha * color, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _present_legend_handles(label_map, frontier_names, max_entries=24):
    num_classes = len(frontier_names)
    palette = build_palette(num_classes)
    present = [c for c in np.unique(label_map) if c != BACKGROUND_ID]
    handles = []
    for c in present[:max_entries]:
        name = frontier_names[c - 1] if 1 <= c <= num_classes else str(c)
        handles.append(Patch(facecolor=np.array(palette[c]) / 255.0, label=name))
    return handles


def save_panel(
    out_path: Union[str, Path],
    image_uint8: np.ndarray,
    result,
    gt_label_map: Optional[np.ndarray] = None,
    gt_frontier_names: Optional[List[str]] = None,
    alpha: float = 0.5,
    dpi: int = 130,
) -> str:
    num_classes = len(result.frontier_names)
    sem = semantic_overlay(image_uint8, result.label_map, num_classes, alpha=alpha)
    inst = instance_overlay(image_uint8, result.instances, alpha=alpha + 0.05)

    n_panels = 3 + (1 if gt_label_map is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.6))
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(image_uint8)
    axes[0].set_title("input")

    axes[1].imshow(sem)
    axes[1].set_title(f"materials (level={result.level})")
    handles = _present_legend_handles(result.label_map, result.frontier_names)
    if handles:
        axes[1].legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    axes[2].imshow(inst)
    axes[2].set_title(f"objects (n={len(result.instances)})")

    if gt_label_map is not None:
        gt_names = gt_frontier_names or result.frontier_names
        gt_vis = semantic_overlay(image_uint8, gt_label_map, len(gt_names), alpha=alpha)
        axes[3].imshow(gt_vis)
        axes[3].set_title("ground truth")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def save_level_comparison(
    out_path: Union[str, Path],
    image_uint8: np.ndarray,
    result,
    levels: List[Union[str, int]],
    alpha: float = 0.5,
    dpi: int = 130,
    show_input: bool = True,
) -> str:
    if result._merger is None:
        raise RuntimeError("result must be attached to a MaterialMerger to re-cut levels")

    results_by_level = []
    for lvl in levels:
        r = result if result.level == lvl else result.recut(lvl)
        results_by_level.append((lvl, r))

    n_panels = len(results_by_level) + (1 if show_input else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.6))
    if n_panels == 1:
        axes = [axes]

    col = 0
    if show_input:
        axes[col].imshow(image_uint8)
        axes[col].set_title("input")
        col += 1

    for lvl, r in results_by_level:
        sem = semantic_overlay(image_uint8, r.label_map, len(r.frontier_names), alpha=alpha)
        axes[col].imshow(sem)
        axes[col].set_title(f"level={lvl}  (k={len(r.frontier_names)}, obj={len(r.instances)})")
        handles = _present_legend_handles(r.label_map, r.frontier_names)
        if handles:
            axes[col].legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)
        col += 1

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def save_overlays(
    out_dir: Union[str, Path],
    image_uint8: np.ndarray,
    result,
    alpha: float = 0.5,
) -> Dict[str, str]:
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(result.frontier_names)
    sem = semantic_overlay(image_uint8, result.label_map, num_classes, alpha=alpha)
    inst = instance_overlay(image_uint8, result.instances, alpha=alpha + 0.05)
    sem_path = out_dir / "overlay_semantic.png"
    inst_path = out_dir / "overlay_instances.png"
    Image.fromarray(sem).save(sem_path)
    Image.fromarray(inst).save(inst_path)
    return {"overlay_semantic": str(sem_path), "overlay_instances": str(inst_path)}
