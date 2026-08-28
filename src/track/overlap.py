from enum import StrEnum

import numpy as np

from .segment import Frame


class Criterion(StrEnum):
    """Which overlap measure gates a candidate link."""

    CONTAINMENT = "containment"  #: shared voxels over the smaller region
    IOU = "iou"  #: shared voxels over the union


def intersection(a: Frame, b: Frame) -> np.ndarray:
    """Voxel counts shared by every pair of regions, as ``[a.count, b.count]``.

    Computed with a single ``bincount`` over the flattened label pair
    ``a * (b.count + 1) + b``, so the whole matrix costs one pass over the
    frame rather than a loop over the ``a.count * b.count`` region pairs.
    """
    if a.count == 0 or b.count == 0:
        return np.zeros((a.count, b.count), dtype=np.int64)

    stride = b.count + 1
    pairs = a.labels.ravel().astype(np.int64) * stride + b.labels.ravel()
    counts = np.bincount(pairs, minlength=(a.count + 1) * stride)
    # row 0 / column 0 are liquid, and are dropped
    return counts.reshape(a.count + 1, stride)[1:, 1:]


def iou(a: Frame, b: Frame, inter: np.ndarray | None = None) -> np.ndarray:
    """Intersection over union for every region pair."""
    if inter is None:
        inter = intersection(a, b)
    union = a.size[:, None] + b.size[None, :] - inter
    return np.divide(inter, union, out=np.zeros(inter.shape), where=union > 0)


def containment(a: Frame, b: Frame, inter: np.ndarray | None = None) -> np.ndarray:
    """Shared voxels as a fraction of the *smaller* region of each pair.

    Preferred over :func:`iou` for gating links across coalescence and
    breakup: when two bubbles merge, each parent is nearly contained in the
    child, so containment stays near 1 while IoU collapses toward the
    parent/child size ratio.
    """
    if inter is None:
        inter = intersection(a, b)
    smaller = np.minimum(a.size[:, None], b.size[None, :])
    return np.divide(inter, smaller, out=np.zeros(inter.shape), where=smaller > 0)


_MEASURES = {Criterion.CONTAINMENT: containment, Criterion.IOU: iou}


def score(a: Frame, b: Frame, criterion: Criterion = Criterion.CONTAINMENT):
    """Evaluate one of the overlap measures by name.

    Accepts a :class:`Criterion` member or its plain-string value.
    """
    try:
        measure = _MEASURES[Criterion(criterion)]
    except ValueError as exc:
        options = ", ".join(c.value for c in Criterion)
        raise ValueError(
            f"unknown criterion {criterion!r}; choose from {options}"
        ) from exc
    return measure(a, b)
