import numpy as np

from scripts.eval.evaluate_minc_s_components import dice, iou, summarize


def test_overlap_metrics():
    pred = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    gt = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
    assert abs(iou(pred, gt) - 0.5) < 1e-9
    assert abs(dice(pred, gt) - (4 / 6)) < 1e-9


def test_summary_fields():
    rows = [
        {
            "photo_id": "a",
            "label_name": "wood",
            "best_iou": 0.8,
            "best_dice": 0.9,
            "gt_mapped": True,
            "matched_label": "timber",
        },
        {
            "photo_id": "b",
            "label_name": "metal",
            "best_iou": 0.2,
            "best_dice": 0.3,
            "gt_mapped": False,
            "matched_label": "foam",
        },
    ]
    image_rows = [
        {"photo_id": "a", "num_components": 3, "seconds": 1.0},
        {"photo_id": "b", "num_components": 5, "seconds": 3.0},
    ]
    metrics = summarize(rows, image_rows, {"timber": "wood"})
    assert metrics["num_images"] == 2
    assert metrics["num_segments"] == 2
    assert metrics["mean_best_iou"] == 0.5
    assert metrics["recall@0.50"] == 0.5
    assert metrics["mean_components_per_image"] == 4.0
    assert metrics["mean_sec_per_image"] == 2.0
    assert metrics["mapped_semantic_accuracy"] == 1.0
