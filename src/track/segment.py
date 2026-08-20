"""Per-frame segmentation of a binary phase mask into labelled bubbles."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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
    periodic: tuple[bool, ...] = ()  # per axis; () means none

    @property
    def ndim(self) -> int:
        return self.labels.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.labels.shape

    @property
    def wraps(self) -> tuple[bool, ...]:
        """``periodic`` padded to ``ndim``, so it is always safe to index."""
        return self.periodic or (False,) * self.ndim


def as_periodic(periodic, ndim: int) -> tuple[bool, ...]:
    """Normalise a periodicity argument to one flag per axis."""
    if periodic is None or periodic is False:
        return (False,) * ndim
    if periodic is True:
        return (True,) * ndim
    flags = tuple(bool(p) for p in periodic)
    if len(flags) != ndim:
        raise ValueError(f"periodic has {len(flags)} flags but data is {ndim}D")
    return flags


def minimum_image(delta: np.ndarray, shape, periodic) -> np.ndarray:
    """Wrap displacement components onto ``[-L/2, L/2)`` for periodic axes.

    A bubble that leaves one side and re-enters the other has moved a short
    way, not the width of the domain. Everything measuring a separation --
    linking distances, track velocities -- has to go through this.
    """
    delta = np.array(delta, dtype=float, copy=True)
    for axis, (length, wrap) in enumerate(zip(shape, periodic)):
        if wrap:
            delta[..., axis] -= length * np.round(delta[..., axis] / length)
    return delta


def _wrap_labels(labels: np.ndarray, count: int, connectivity: int, periodic):
    """Fuse labels that meet across a periodic face.

    ``ndimage.label`` treats the array as a box, so a bubble straddling a
    periodic boundary comes back as two components. Opposite faces are
    compared and the label pairs found there are merged.
    """
    edges: list[tuple[int, int]] = []
    ndim = labels.ndim
    # the wrap already uses up one differing coordinate, so the neighbours it
    # may connect to diagonally are limited by the remaining connectivity
    free = max(connectivity - 1, 0)

    for axis, wrap in enumerate(periodic):
        if not wrap or labels.shape[axis] < 2:
            continue
        lo = np.take(labels, 0, axis=axis)
        hi = np.take(labels, -1, axis=axis)
        for shift in product((-1, 0, 1), repeat=ndim - 1):
            if sum(s != 0 for s in shift) > free:
                continue
            rolled = np.roll(hi, shift, axis=tuple(range(ndim - 1)))
            both = (lo > 0) & (rolled > 0)
            if both.any():
                edges.extend(zip(lo[both].tolist(), rolled[both].tolist()))

    if not edges:
        return labels, count

    rows, cols = zip(*edges)
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(count + 1, count + 1),
    )
    _, comp = connected_components(graph, directed=False)
    # renumber the merged components contiguously from 1, leaving 0 as liquid
    _, inverse = np.unique(comp[1:], return_inverse=True)
    remap = np.zeros(count + 1, dtype=np.int32)
    remap[1:] = inverse + 1
    return remap[labels], int(inverse.max()) + 1


def _centroids(labels: np.ndarray, count: int, size: np.ndarray, periodic):
    """Centre of mass per label, using a circular mean on periodic axes.

    The plain mean of a bubble sitting astride a periodic boundary lands in
    the middle of the domain, nowhere near the bubble. Averaging the angle
    instead puts it in the right place.
    """
    if not count:
        return np.empty((0, labels.ndim))

    flat = labels.ravel()
    out = np.empty((count, labels.ndim))
    for axis in range(labels.ndim):
        length = labels.shape[axis]
        line = np.arange(length)
        broadcast = [1] * labels.ndim
        broadcast[axis] = length
        coord = np.broadcast_to(line.reshape(broadcast), labels.shape).ravel()

        if periodic[axis]:
            angle = line * (2 * np.pi / length)
            cos = np.bincount(flat, np.cos(angle)[coord], count + 1)[1:]
            sin = np.bincount(flat, np.sin(angle)[coord], count + 1)[1:]
            pos = np.mod(np.arctan2(sin, cos) * (length / (2 * np.pi)), length)
            # a true 0 can come back as L - 1e-15; snap it rather than report L
            pos[np.isclose(pos, length)] = 0.0
            out[:, axis] = pos
        else:
            out[:, axis] = np.bincount(flat, coord, count + 1)[1:] / size
    return out


def segment(mask, connectivity=1, min_size=0, periodic=None) -> Frame:
    """Label connected vapor regions in one frame.

    Args:
        mask: boolean array, True = vapor. 2D ``[Y, X]`` or 3D ``[Z, Y, X]``.
        connectivity: 1 links face-neighbours only (4-conn in 2D, 6-conn in
            3D); ``mask.ndim`` also links diagonals (8-conn / 26-conn).
            Face-only is the conservative choice -- full connectivity fuses
            bubbles that merely touch at a corner.
        min_size: drop regions smaller than this many voxels. Remaining
            labels are renumbered to stay contiguous.
        periodic: one flag per axis, ``True`` where the domain wraps; or a
            single bool for all axes. For data laid out ``[(Z,) Y, X]``,
            left-right periodic is the last axis and top-bottom periodic is
            the one before it. Wrapped regions are fused into a single label
            and their centroids are computed as a circular mean.
    """
    mask = np.asarray(mask, dtype=bool)
    wraps = as_periodic(periodic, mask.ndim)
    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labels, count = ndimage.label(mask, structure=structure)

    if any(wraps):
        labels, count = _wrap_labels(labels, count, connectivity, wraps)

    size = np.bincount(labels.ravel(), minlength=count + 1)[1:]

    if min_size > 0 and count:
        keep = size >= min_size
        # remap[old] -> new, with dropped regions folded back into liquid
        remap = np.zeros(count + 1, dtype=np.int32)
        remap[1:][keep] = np.arange(1, keep.sum() + 1, dtype=np.int32)
        labels = remap[labels]
        size = size[keep]
        count = int(keep.sum())

    centroid = _centroids(labels, count, size, wraps)
    return Frame(
        labels.astype(np.int32), count, size.astype(np.int64), centroid, wraps
    )


def touches_wall(frame: Frame, axis: int = 0, side: int = 0) -> np.ndarray:
    """Which regions touch a domain boundary, as a bool array of ``count``.

    Defaults to the low face of axis 0, which is the heater in a boiling
    domain laid out ``[(Z,) Y, X]`` with the wall at ``Y = 0``. A bubble
    touching it is still attached to the nucleation site and being fed
    vapor, so it is not conserving volume.

    A periodic face is not a wall; asking about one raises rather than
    silently reporting attachment.
    """
    if frame.wraps[axis]:
        raise ValueError(f"axis {axis} is periodic, so it has no wall")
    face = np.take(frame.labels, side, axis=axis)
    hit = np.unique(face)
    attached = np.zeros(frame.count, dtype=bool)
    hit = hit[hit > 0]
    attached[hit - 1] = True
    return attached
