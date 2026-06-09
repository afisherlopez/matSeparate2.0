"""Connected-component instance extraction (boilerplate).

Each pixel is assigned to the highest-probability frontier class. We extract
8-connected components separately for each material class and discard
components smaller than 64 pixels. Each remaining component becomes a predicted
material region with a binary mask, label, bounding box, area, and mean
confidence.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import ndimage

MIN_OBJECT_AREA = 64
CONNECTIVITY = 8


@dataclass
class Instance:
    id: int
    material: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    score: float
    mask: np.ndarray = field(repr=False)


def build_label_map(frontier_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax over frontier classes. Class ids are 1-based (0 = background)."""
    conf = frontier_probs.max(axis=-1)
    label_map = frontier_probs.argmax(axis=-1).astype(np.int32) + 1
    return label_map, conf.astype(np.float32)


def extract_instances(label_map: np.ndarray, confidence: np.ndarray,
                      frontier_names: List[str]) -> List[Instance]:
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connected
    instances: List[Instance] = []
    next_id = 1

    for class_id in range(1, len(frontier_names) + 1):
        class_mask = label_map == class_id
        if not class_mask.any():
            continue
        labeled, n_comp = ndimage.label(class_mask, structure=structure)
        for comp_idx, sl in enumerate(ndimage.find_objects(labeled), start=1):
            if sl is None:
                continue
            comp_mask = labeled == comp_idx
            area = int(comp_mask.sum())
            if area < MIN_OBJECT_AREA:
                continue
            ys, xs = sl
            instances.append(
                Instance(
                    id=next_id,
                    material=frontier_names[class_id - 1],
                    bbox=(int(xs.start), int(ys.start),
                          int(xs.stop - xs.start), int(ys.stop - ys.start)),
                    area=area,
                    score=float(confidence[comp_mask].mean()),
                    mask=comp_mask,
                )
            )
            next_id += 1
    return instances
