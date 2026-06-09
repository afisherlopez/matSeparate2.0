# MatSeparate

Core implementations of the MatSeparate learning and
segmentation algorithms, separated from the agent-assisted experiment tooling.

## Layout

```
boilerplate/
  learning/                 # patch classification on Matador-C1
    taxonomy.py             #   Matador taxonomy wrapper (paths, levels, edges)
    data.py                 #   PatchDataset (local + optional context crops)
    models.py               #   ResNet / global ResNet / (global) HGNN
    losses.py               #   HGNN path + level-wise objective
    train.py                #   single fixed training run
    eval.py                 #   per-taxonomy-level accuracy
  segmentation/             # patch -> material-labeled components
    sampler.py              #   sliding window (128 px / 32 px)
    classify.py             #   patch classifier -> leaf-prob grid
    upsample.py             #   bilinear upsample to full resolution
    refine.py               #   SLIC superpixel refine (+ dense-CRF stub)
    taxonomy_cut.py         #   leaf -> taxonomy-level projection
    objects.py              #   8-connected-component instances
    pipeline.py             #   end-to-end MaterialSegmenter
    metrics.py              #   mean IoU, Recall@0.50, sec/img, semantic acc.
```

Two companion docs live at the **repository root**:

- `PROMPTS.md` — how each module was extended into agentic tooling.
- `ATTRIBUTION.md` — our work vs Beveridge vs Python packages.

## What is *not* here (agentic extensions)

Grid search, TensorBoard logging, LR schedulers, the dense-CRF backend with
taxonomy-aware compatibility, the stub predictor, region-agreement metrics
(ARI/VOI/boundary-F1), the SAM baseline, and result serialization all live in
the main-repo `classifier/` and `segmentation/` packages. `PROMPTS.md` records
the prompts that produced them from this boilerplate.
