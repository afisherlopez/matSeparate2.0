# Extension Prompts from Boilerplate to Agentic

This document records how the fixed boilerplate implementations in
`boilerplate/` were extended into the fuller, agent-assisted versions used to
run the experiments in the paper. We wrote each pipeline in a functional, fixed
way first, then used coding agents (Claude Opus /GPT-5) to add the experiment-management machinery: grid search, logging,alternative backends, additional metrics, and test scaffolding.

Each section names the boilerplate file it starts from, the extended
module in the main repo, and the prompts we used to produce it.

---

## 1. Hyperparameter grid search over the segmentation sampler

- **From:** `boilerplate/segmentation/sampler.py` (fixed `window_size=128`, `stride=32`)
- **To:** `segmentation/patches.py` (`min_patches`/`max_patches`/`window_area_pct`,
  `_resolve_window_stride`, adaptive grids, debug prints)
- **Prompt:**
  > Extend the `SlidingWindowSampler` so the window size and stride can be chosen
  > automatically to hit a target patch budget. Add optional `min_patches`,
  > `max_patches`, and `window_area_pct` parameters. When a patch budget is set,
  > scale the stride (and shrink the window down to a floor) until the number of
  > sampled windows falls within `[min_patches, max_patches]`. Keep the fixed
  > 128/32 behavior as the default when no budget is given. Add a `build_sampler`
  > factory driven by the segmentation config and log the resolved window/stride
  > per image. Make this sweepable so we can run a grid search over window/stride
  > on MINC-S and record mean IoU per configuration.

## 2. TensorBoard logging, schedulers, and run management for training

- **From:** `boilerplate/learning/train.py` (single run, print-only, best-val checkpoint)
- **To:** `classifier/train.py` + YAML configs in `configs/classifier/`
- **Prompts:**
  > Refactor the training loop to be config-driven: load all hyperparameters,
  > data paths, and model settings from a YAML file instead of module constants,
  > and add a `make_model`/`make_loader` factory. Add a `--config` and
  > `--dry-run` flag.
  >
  > Add TensorBoard logging of train/val loss and per-level accuracy, log the
  > learning rate, and write the config into the run directory. Add a cosine LR
  > scheduler with warmup and optional mixed-precision training.
  >
  > Add a grid-search driver that takes a base config and a dict of
  > hyperparameter ranges (lr, weight decay, dropout, embed/hidden dims, batch
  > size), launches one run per combination into its own run directory, and
  > collects the best validation accuracy of each into a summary table.

## 3. Path-constrained inference and MC-dropout uncertainty

- **From:** `boilerplate/learning/models.py` (`HGNNPatchClassifier.forward`
  returns node logits; leaf prediction is a plain argmax over leaf positions)
- **To:** greedy / beam path decoding and MC-dropout uncertainty (as in the
  Beveridge `hgnn.py` `predict` / `_get_best_path` / `_get_best_path_beam`)
- **Prompt:**
  > Add hierarchy-constrained decoding to `HGNNPatchClassifier`. Given node
  > probabilities, walk the taxonomy from the root, at each step choosing the
  > highest-probability child via the adjacency matrix, to return a valid
  > root-to-leaf path. Add a beam-search variant with a configurable beam width.
  > Add an `enable_dropout()` + MC-dropout mode that runs several stochastic
  > forward passes to estimate per-node uncertainty, and let the path decoder
  > penalize uncertain nodes.

## 4. Dense-CRF refinement backend with taxonomy-aware compatibility

- **From:** `boilerplate/segmentation/refine.py` (`refine_slic` + a minimal
  `refine_crf` stub)
- **To:** `segmentation/crf.py` (backend selection, taxonomy-aware bilateral
  compatibility matrix, graceful fallbacks)
- **Prompt:**
  > Turn the refinement step into a selectable backend (`none`, `superpixel`,
  > `dense`) driven by the config. For the dense CRF, build a label-compatibility
  > matrix from the taxonomy: penalize confusions between materials that are far
  > apart in the tree less than confusions between nearby materials, by using the
  > normalized shortest-path distance between leaf nodes as the bilateral
  > compatibility. Add a Felzenszwalb option to the superpixel backend.

## 5. Stub predictor for testing the eval plumbing

- **From:** `boilerplate/segmentation/classify.py` (`PatchClassifier` only)
- **To:** `segmentation/classify.py` (`StubLeafPredictor`, `LeafProbPredictor`
  protocol, tqdm progress)
- **Prompt:**
  > Add a `StubLeafPredictor` that fakes leaf probabilities from the mean pixel
  > value of each patch, so we can run the whole segmentation + MINC-S evaluation
  > pipeline end-to-end without a trained checkpoint. Define a `LeafProbPredictor`
  > Protocol so the pipeline accepts either the real classifier or the stub, and
  > add a tqdm progress bar over patch batches. Wire a `--stub` flag into the
  > evaluation script.

## 6. Extra segmentation-agreement metrics

- **From:** `boilerplate/segmentation/metrics.py` (mean best IoU, Recall@0.50,
  components/image, sec/image, mapped semantic accuracy)
- **To:** `segmentation/metrics.py` (Adjusted Rand Index, variation of
  information, segmentation covering, boundary-F1, confusion-matrix mIoU)
- **Prompt:**
  > Add region-based partition-agreement metrics for comparing our label maps to
  > the ground truth: adjusted Rand index, variation of information, segmentation
  > covering in both directions, and a boundary-F1 at a pixel tolerance. Also add
  > a confusion-matrix-based per-class IoU / pixel accuracy path with a
  > crosswalk-aware shared label space between Matador-C1 and MINC-S. Keep the
  > reported metrics (mean best IoU, Recall@K, sec/image, mapped semantic
  > accuracy) as the headline summary.

## 7. SAM baseline and the full MINC-S evaluation harness

- **From:** `boilerplate/segmentation/metrics.py::summarize` +
  `boilerplate/segmentation/pipeline.py`
- **To:** `scripts/eval/evaluate_minc_s_components.py`
- **Prompts:**
  > Write a MINC-S evaluation script that reads the segments index, loads each
  > binary segment mask and its photo, runs the segmenter once per image, matches
  > every ground-truth segment to its highest-IoU predicted component, and writes
  > per-segment and per-image CSVs plus an aggregate metrics JSON. Add a
  > Matador-C1 -> MINC-S material crosswalk and compute mapped semantic accuracy.
  >
  > Add a Segment Anything (SAM ViT-B automatic mask generator) baseline that
  > produces class-agnostic proposals. Give SAM oracle credit by matching each
  > ground-truth segment to its best-IoU SAM proposal, and include SAM in the
  > geometric metrics only (it has no material labels).

## 8. Recutting, serialization, and visualization

- **From:** `boilerplate/segmentation/pipeline.py` (`segment()` returns a result)
- **To:** `segmentation/pipeline.py` (`MaterialMerger.recut`,
  `SegmentationResult.save`) + `segmentation/formats.py`, `segmentation/visualize.py`
- **Prompt:**
  > Cache the refined leaf-probability map on the result so we can re-project it
  > to a different taxonomy level without re-running the classifier (`recut`).
  > Add a `save()` that writes the label map, a color visualization, per-instance
  > PNGs, and a COCO-style instances JSON. Add a `from_patch_classifier`
  > constructor that loads a checkpoint or run directory.
