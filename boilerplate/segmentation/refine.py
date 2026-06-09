"""Spatial refinement of the dense probability map.

The image is segmented into superpixels and the probability vector
is averaged within each superpixel, encouraging local spatial consistency
without a trained dense segmentation model.

A dense-CRF hook (`refine_crf`) is also provided, but it was excluded from 
the paper because it was too slow at evaluation time.
"""

from __future__ import annotations
import numpy as np


def refine_slic(image_uint8: np.ndarray, p_dense: np.ndarray,
                n_segments: int = 400) -> np.ndarray:
    from skimage.segmentation import slic

    segments = slic(image_uint8, n_segments=n_segments, start_label=0, channel_axis=-1)
    refined = np.empty_like(p_dense)
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        refined[mask] = p_dense[mask].mean(axis=0)

    denom = refined.sum(axis=-1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return (refined / denom).astype(np.float32)


def refine_crf(image_uint8: np.ndarray, p_dense: np.ndarray,
               n_iterations: int = 7) -> np.ndarray:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w, num_labels = p_dense.shape
    probs = np.clip(p_dense, 1e-8, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    d = dcrf.DenseCRF2D(w, h, num_labels)
    d.setUnaryEnergy(
        np.ascontiguousarray(unary_from_softmax(probs.transpose(2, 0, 1)))
    )
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(
        sxy=60, srgb=13, rgbim=np.ascontiguousarray(image_uint8), compat=10
    )
    q = d.inference(n_iterations)
    return np.array(q).reshape(num_labels, h, w).transpose(1, 2, 0).astype(np.float32)
