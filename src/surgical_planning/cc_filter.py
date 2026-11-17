"""
cc_filter.py
============
Connected component post-processing for segmentation probability masks.

Removes disconnected small predicted regions (false positives outside the
main tumour mass) before volume counting and display. The largest component
plus any secondary component >= min_voxels are retained; everything else
is zeroed in the probability map.

CPU cost: negligible (~10 ms on 128^3 mask).
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_cc_filter(
    prob_mask: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 50,
) -> np.ndarray:
    """
    Remove disconnected small predicted regions.

    Args:
        prob_mask:   [D, H, W] float probability map from the model.
        threshold:   Binarisation threshold (must match inference threshold).
        min_voxels:  Secondary components smaller than this are discarded.

    Returns:
        Cleaned probability map — values outside kept components set to 0.
        Same dtype and shape as input.
    """
    try:
        from scipy.ndimage import label as scipy_label

        binary = (prob_mask > threshold)
        n_vox  = int(binary.sum())

        if n_vox == 0:
            return prob_mask

        labeled, n_comp = scipy_label(binary)
        if n_comp <= 1:
            return prob_mask   # single component — nothing to remove

        sizes = np.array([(labeled == i).sum() for i in range(1, n_comp + 1)])
        keep  = np.where(sizes >= min_voxels)[0] + 1   # 1-indexed labels

        if len(keep) == 0:
            keep = np.array([int(np.argmax(sizes)) + 1])   # always keep largest

        cleaned = np.where(np.isin(labeled, keep), prob_mask, 0.0)
        removed = n_vox - int((cleaned > threshold).sum())

        if removed > 0:
            logger.info(
                f"CC filter: removed {removed:,} isolated voxels "
                f"({n_comp - len(keep)} small component(s) discarded)"
            )

        return cleaned.astype(prob_mask.dtype)

    except Exception as exc:
        logger.warning(f"CC filter failed — returning original mask: {exc}")
        return prob_mask
