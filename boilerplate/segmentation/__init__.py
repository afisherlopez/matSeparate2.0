"""Patch-based material segmentation pipeline (fixed boilerplate).

Converts a trained patch classifier into material-labeled connected components:

    sliding window -> classify patches -> bilinear upsample
        -> SLIC superpixel refine -> taxonomy-level projection
        -> argmax label map -> 8-connected components

Fixed to the paper's main MINC-S configuration: window 128 px, stride 32 px,
SLIC refinement, min component area 64 px.
"""
