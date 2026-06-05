All of the best and none of the rest. Shoutout MatBev.

## Pipelines

**Patch classifier** (`classifier/`, `gnn_classifier/`): ResNet50 backbone with three head variants — flat cross-entropy, HGNN with hierarchical loss, and a global-context HGNN that fuses local patch features with a scene-level embedding. Trained on Matador-C1 or MINC-2500. The HGNN predicts probabilities over all nodes in the taxonomy DAG simultaneously, exploiting parent-child structure at training time.

**Patch-and-merge segmentation** (`segmentation/`): Runs a sliding-window or grid patch classifier over an image to produce a dense per-pixel probability map, upsamples it bilinearly, optionally refines boundaries with a dense CRF, then cuts the taxonomy tree at a chosen coarseness level to produce a label map and connected-component instances. Re-cutting to a different level is a cheap matmul on the cached leaf probabilities — no re-classification needed. Entry point: `tools/segment_image.py`.

**SAM baseline** (`segmentation/sam_baseline.py`): Uses Segment Anything (SAM) to generate segmentation masks from a photo (auto, point-prompt, or box-prompt mode), then evaluates those masks against ground-truth material segments from the MINC-S dataset. This measures how well proposal-stage masks align with material boundaries before any classification step. Entry point: `scripts/eval/eval_sam_minc.py`.

## Directory

```
classifier/          ResNet50 / HGNN patch classifier (models, training, inference)
gnn_classifier/      HGNN architecture and hierarchical loss function
segmentation/        Patch-and-merge pipeline, CRF, SAM baseline, output formats
taxonomy/            Taxonomy DAG assets (JSON) and tree utilities (networkx)
datasets/            MINC-2500 and Matador-C1 dataset loaders
configs/             Training configs (configs/classifier/, configs/segmentation.yaml)
scripts/train/       train_hgnn.py, train_classifier.py, train_ablation_variants.py
scripts/eval/        eval_classifiers.py, eval_ablations.py, eval_sam_minc.py, eval_segmentation.py
scripts/data/        Manifest and split builders for Matador and MINC
tools/               Inference CLIs: segment_image.py, infer_api.py, patch_diagnostic.py
tests/               Full test suite (pytest tests/ -q)
```

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
