"""Per-frame segmentation of a binary phase mask into labelled bubbles."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Frame:
    """Labelled vapor regions in a single timestep.

    Labels are contiguous in ``1..count``; 0 is liquid. ``size`` and
    ``centroid`` are indexed by ``label - 1``.
    """

    labels: np.ndarray  # int32 [(Z,) Y, X], 0 = liquid
    count: int
    size: np.ndarray  # int64 [count], voxel counts
    centroid: np.ndarray  # float64 [count, ndim], in index space

    @property
    def ndim(self) -> int:
        return self.labels.ndim


def segment(mask, connectivity=1, min_size=0) -> Frame:
    """Label connected vapor regions in one frame.

    Args:
        mask: boolean array, True = vapor. 2D ``[Y, X]`` or 3D ``[Z, Y, X]``.
        connectivity: 1 links face-neighbours only (4-conn in 2D, 6-conn in
            3D); ``mask.ndim`` also links diagonals (8-conn / 26-conn).
            Face-only is the conservative choice -- full connectivity fuses
            bubbles that merely touch at a corner.
        min_size: drop regions smaller than this many voxels. Remaining
            labels are renumbered to stay contiguous.
    """
    mask = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labels, count = ndimage.label(mask, structure=structure)

    size = np.bincount(labels.ravel(), minlength=count + 1)[1:]

    if min_size > 0 and count:
        keep = size >= min_size
        # remap[old] -> new, with dropped regions folded back into liquid
        remap = np.zeros(count + 1, dtype=np.int32)
        remap[1:][keep] = np.arange(1, keep.sum() + 1, dtype=np.int32)
        labels = remap[labels]
        size = size[keep]
        count = int(keep.sum())

    if count:
        centroid = np.array(
            ndimage.center_of_mass(labels > 0, labels, np.arange(1, count + 1))
        )
    else:
        centroid = np.empty((0, mask.ndim))

    return Frame(labels.astype(np.int32), count, size.astype(np.int64), centroid)
