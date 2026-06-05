All of the best and none of the rest. Shoutout MatBev.

## Components

- `classifier/`: ResNet50, HGNN, and global-context HGNN patch classifiers.
- `segmentation/`: material segmentation pipeline (SAM-based, CRF refinement, taxonomy-guided merging).
- `gnn_classifier/`: HGNN model and hierarchical loss.
- `datasets/`: MINC-2500 and Matador-C1 dataset loaders.
- `taxonomy/`: shared taxonomy assets and tree utilities.

## Usage

**Segment an image:**
```bash
python tools/segment_image.py --image path/to/image.jpg --out output/ --run-dir runs/<checkpoint>/
```

**Train a classifier:**
```bash
python scripts/train/train_classifier.py --model flat
python scripts/train/train_classifier.py --model hierloss --epochs 10
```

**Train HGNN:**
```bash
python scripts/train/train_hgnn.py --config configs/classifier/hgnn.yaml
```

**Evaluate:**
```bash
python scripts/eval/eval_classifiers.py --run-dir runs/<checkpoint>/
python scripts/eval/eval_segmentation.py --predictions path/to/preds/ --gt path/to/gt/
```

**Run tests:**
```bash
pytest tests/ -q
```

Data preparation scripts are in `scripts/data/`. Diagnostic tools (mask comparison, dataset inspection) are in `tools/`.
