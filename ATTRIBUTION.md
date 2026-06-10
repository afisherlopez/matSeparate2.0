# Attribution

This document records, for every major component of MatSeparate, whether it is
our own production, adapted from the Beveridge et al. (2025) starter
code / paper, or provided by an existing Python package. Paths in
`boilerplate/` refer to the fixed boilerplate in this directory.

---

## 1. Adapted from Beveridge et al. (2025)

The hierarchical material-recognition (HMR) starter code Beveridge shared with
us consisted of `hgnn.py`, `loss.py`, `taxonomy/tree.py`, and a
`taxonomy-tree.json` asset. The following are derived from it:

- **Matador taxonomy and tree utilities** — `taxonomy/tree.py` and
  `taxonomy/assets/taxonomy-tree.json` are taken essentially verbatim from the
  starter code (`get_taxonomy`, `get_hierarchy_levels`, `get_hierarchy_mask`,
  `taxa_to_onehot`/`taxa_to_indices`, `all_paths_subgraph`,
  `level_complete_tree`, `write_network_text_with_color`, etc.). Our
  `matador-c1-taxonomy.json` is the Matador-C1 subset of that tree.
- **HGNN architecture** — `boilerplate/learning/models.py` (`GraphBackbone`,
  `HGNNPatchClassifier`) reimplements the design from the starter `hgnn.py`:
  a ResNet image embedding projected and inserted as a graph node alongside one
  prototype node per taxonomy node (initialized from average CNN embeddings,
  `init_prototypes`), a 2-layer GAT with skip connections (hidden 512, output
  256, 1 head), bidirectional imagetotaxonomy edges plus undirected taxonomy edges, global mean pooling, and a node-level linear classifier. The original
  `ImageEncoder`,
  `GraphBackbone`, `HGNN._init_graph`, and the greedy / beam best-path decoding
  are Beveridge's.
- **Hierarchical training objective** — `boilerplate/learning/losses.py`
  (`hgnn_loss`) is our condensed version of the starter `loss.py`
  (`greedy_loss` = path BCE combined with `hierarchical_softmax_loss`, a
  per-level cross-entropy, via a winner-take-all maximum).
- **Matador-C1 filtering protocol** — the choice to drop low-texture materials
  (thermoplastic, paint, glass, …) follows the Beveridge paper.

## 2. Our production

Everything specific to extending local hierarchical recognition into
scene-level material segmentation is ours:

- **Full segmentation pipeline** — `boilerplate/segmentation/*`:
  sliding-window sampling, coarse-grid assembly, bilinear upsampling, SLIC
  superpixel refinement, leaf -> frontier taxonomy projection, argmax label map,
  and 8-connected-component instance extraction
  (`sampler.py`, `upsample.py`, `refine.py`, `taxonomy_cut.py`, `classify.py`,
  `objects.py`, `pipeline.py`). The extended versions live in `segmentation/`.
- **Flat and global-context classifiers** — `ResNetPatchClassifier`,
  `GlobalResNetPatchClassifier`, and the global-context HGNN dual-encoder
  fusion (`use_context=True`) are our additions; the Beveridge starter only
  provided the single-image HGNN.
- **Data + training/eval glue** — Matador-C1 manifest/dataset loaders, the
  fixed training and per-level evaluation loops, run configs.
- **MINC-S evaluation harness** — the curated MINC-S subset protocol, the
  Matador-C1 to MINC-S material crosswalk, best-IoU matching, the SAM oracle
  baseline comparison, and the reported metrics
  (`boilerplate/segmentation/metrics.py`,
  `scripts/eval/evaluate_minc_s_components.py`).

## 3. Implemented with existing Python packages

- **`timm`** — pretrained ResNet50 backbones (`timm.create_model`).
- **`torch` / `torchvision`** — model training, transforms, bilinear
  interpolation (`F.interpolate`).
- **`torch_geometric`** — graph attention (`GATConv`), `global_mean_pool`,
  `Data`/`Batch` graph batching used by the HGNN.
- **`networkx`** — taxonomy graph representation, shortest paths, level/ancestor
  queries, leaf-to-frontier distances.
- **`scikit-image`** — SLIC (and Felzenszwalb) superpixels for refinement.
- **`pydensecrf`** — dense CRF refinement backend (extension only).
- **`scipy.ndimage`** — connected-component labeling and bounding boxes.
- **`segment-anything` (SAM, ViT-B)** — class-agnostic automatic mask baseline;
  used as-is, not fine-tuned.
- **`numpy`, `Pillow`, `tifffile`, `PyYAML`** — array ops, image and TIFF I/O,
  config parsing.

## References

- Beveridge et al. (2025), *Matador taxonomy / hierarchical material recognition*.
- Bell et al. (2015), *Materials in Context (MINC)* — sliding-CNN + dense-CRF
  segmentation that inspired the pipeline structure.
- Veličković et al. (2018), *Graph Attention Networks*.
- He et al. (2016), *Deep Residual Learning (ResNet)*.
- Kirillov et al. (2023), *Segment Anything*.
