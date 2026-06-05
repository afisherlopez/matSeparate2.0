# Patch classifier

This folder contains the minimal patch classifier code for the project.

The taxonomy is intentionally a config argument. The same training code can use
the Matador taxonomy, a MINC taxonomy, or a mapped taxonomy file as long as the
CSV labels are node names in that taxonomy.

## Train

```bash
python -m patch_classifier.train --config patch_classifier/configs/resnet50.yaml
python -m patch_classifier.train --config patch_classifier/configs/hgnn.yaml
```

Override the taxonomy from the command line:

```bash
python -m patch_classifier.train \
  --config patch_classifier/configs/hgnn.yaml \
  --taxonomy segmentation_pipeline/taxonomy/assets/minc-taxonomy.json
```

## Expected manifest columns

By default the dataset expects:

- `image_path`: local patch path
- `c1_label`: leaf material label

Both names can be changed in the YAML file. For global-context training, add a
context path column and set `data.context_col`.

## Use in segmentation

The segmentation pipeline can load a patch-classifier checkpoint directly:

```bash
python segmentation_pipeline/scripts/segment_image.py \
  --classifier patch_classifier \
  --checkpoint runs/hgnn/best.pt \
  --image segmentation_pipeline/test_images/minc_1.jpg \
  --out out/segmentation_hgnn
```
