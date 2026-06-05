All of the best and none of the rest. Shoutout MatBev.

## Components

- `patch_classifier/`: minimal ResNet50, HGNN, and global-context HGNN patch classifiers.
- `segmentation_pipeline/`: segmentation code and shared taxonomy assets.

Useful setup/evaluation scripts live under `segmentation_pipeline/scripts/`, including
Matador/MINC manifest builders, mask comparison tools, and MINC-S component evaluation.
SAM baseline evaluation is included as an optional eval path; it requires installing
Meta's `segment-anything` package and providing a SAM checkpoint.
