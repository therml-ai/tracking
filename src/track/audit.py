"""Physics consistency checks on a set of tracks.

No ground truth is needed: a coalescence has to conserve volume, so the
recorded events can be tested against the geometry they claim happened.

Only detached bubbles are checked. One still sitting on the heater is being
fed vapor by evaporation, so its volume is free to grow and conservation says
nothing; on subcooled data the wall-attached merges run ~5% heavy while
detached ones sit near 1.0.

The bound is two-sided rather than ``V_child <= sum(V_parents)``. Bubbles are
separate components while a thin liquid film still divides them, and
coalescence turns that film into vapor, so a product can legitimately exceed
the sum of its parents by about the neck volume.

Volume is not expected to hold exactly. A phase mask only locates the
interface to within a voxel, so the error is a *surface* effect and its
relative size falls off as ``V ** (-1/ndim)``. Measured on subcooled boiling
data, the frame-to-frame spread of untouched bubbles follows
``sigma ~ V ** -0.538`` against the -0.5 that surface-to-volume predicts,
with ``sigma ~= 0.55 / sqrt(V)`` in 2D. The tolerance therefore widens for
small bubbles instead of applying one fixed percentage everywhere.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .link import EventKind, Tracks
from .segment import Frame, touches_wall

TOPOLOGY = (EventKind.MERGE, EventKind.SPLIT, EventKind.COMPLEX)


@dataclass(frozen=True)
class Violation:
    """One event whose volume bookkeeping does not add up."""

    time: int
    kind: EventKind
    check: str
    ratio: float
    parents: tuple[int, ...]
    children: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"frame {self.time}: {self.kind} {list(self.parents)}"
            f" -> {list(self.children)} failed {self.check} (ratio {self.ratio:.3f})"
        )


def sizes_by_track(tracks: Tracks) -> list[dict[int, int]]:
    """Voxel count of every track, per frame.

    Read straight off the :class:`~track.link.Tracks`, so auditing a long run
    does not need every label image resident.
    """
    return [
        dict(zip(ids.tolist(), size.tolist()))
        for ids, size in zip(tracks.ids, tracks.size)
    ]


def attached_by_track(
    tracks: Tracks, frames: Sequence[Frame], axis: int = 0, side: int = 0
) -> list[dict[int, bool]]:
    """Whether each track touches the wall, per frame.

    This one does need the label images: attachment is a property of where the
    voxels are, which the per-region summaries do not capture.
    """
    if len(frames) != len(tracks.ids):
        raise ValueError(
            f"got {len(frames)} frames but {len(tracks.ids)} tracked steps; "
            "pass the same sequence that was linked"
        )
    return [
        dict(zip(ids.tolist(), touches_wall(frame, axis, side).tolist()))
        for ids, frame in zip(tracks.ids, frames)
    ]


def volume_consistency(
    tracks: Tracks,
    frames: Sequence[Frame] | None = None,
    lower: float = 0.95,
    upper: float = 1.15,
    noise: float = 0.55,
    sigmas: float = 3.0,
    min_volume: int = 25,
    detached_only: bool = True,
    wall_axis: int = 0,
    wall_side: int = 0,
) -> list[Violation]:
    """Check that volume is conserved across every topology change.

    Two tests per event. ``sum_ratio`` is the total volume after over the
    total before, which must sit in ``[lower, upper]`` -- below means volume
    vanished, above means far more vapor appeared than a neck can account
    for, the signature of two unrelated regions being fused. ``largest``
    checks that the biggest product is at least as big as the biggest input,
    which no genuine coalescence can fail.

    With ``detached_only`` (the default) an event is skipped when any of its
    bubbles touches the wall, since evaporation there adds volume that
    conservation cannot account for. Pass ``wall_axis``/``wall_side`` to say
    which boundary is the heater.

    ``lower`` and ``upper`` are the *physical* band, the deviation a real
    bubble may show once discretisation is accounted for: condensation in
    subcooled liquid pulls the ratio below 1, a coalescence neck pushes it
    above. On top of that sits a discretisation allowance
    ``sigmas * noise * V ** (-1/ndim)``, which is wide for a 10-voxel bubble
    and negligible for a 1000-voxel one. Set ``sigmas=0`` for a hard band.

    ``min_volume`` skips events too small to measure at all; below roughly 25
    voxels the ratio is dominated by single-voxel quantisation.

    ``frames`` is only needed for ``detached_only``, which has to look at where
    the voxels are; volumes come from ``tracks`` itself.
    """
    if detached_only and frames is None:
        raise ValueError(
            "detached_only needs frames to find wall-attached bubbles; "
            "pass the linked sequence, or detached_only=False"
        )
    size = sizes_by_track(tracks)
    ndim = tracks.centroid[0].shape[1] if tracks.centroid else 2
    wall = (
        attached_by_track(tracks, frames, wall_axis, wall_side)
        if detached_only
        else None
    )
    out: list[Violation] = []

    for e in tracks.events:
        if e.kind not in TOPOLOGY:
            continue
        if wall is not None and (
            any(wall[e.time][i] for i in e.parents)
            or any(wall[e.time + 1][i] for i in e.children)
        ):
            continue  # still on the heater, still growing
        before = [size[e.time][i] for i in e.parents]
        after = [size[e.time + 1][i] for i in e.children]
        if not before or not after or sum(before) < min_volume:
            continue  # too few voxels for the ratio to mean anything

        # a voxel of interface error matters less the bigger the bubble is
        slack = sigmas * noise * sum(before) ** (-1.0 / ndim)
        total = sum(after) / sum(before)
        if not (lower - slack) <= total <= (upper + slack):
            out.append(
                Violation(e.time, e.kind, "sum_ratio", total, e.parents, e.children)
            )

        largest = max(after) / max(before)
        if e.kind is EventKind.MERGE and largest < lower - slack:
            out.append(
                Violation(e.time, e.kind, "largest", largest, e.parents, e.children)
            )
    return out


def volume_ratios(tracks: Tracks) -> dict[str, np.ndarray]:
    """Raw ratios for calibrating the bounds, keyed by event kind.

    ``continue`` is the control: it is the same measurement on events where
    no topology changed, so it isolates how much of any merge surplus is
    just ambient growth between frames.
    """
    size = sizes_by_track(tracks)
    buckets: dict[str, list[float]] = {}
    for e in tracks.events:
        if e.kind in (EventKind.START, EventKind.END):
            continue
        before = sum(size[e.time][i] for i in e.parents)
        after = sum(size[e.time + 1][i] for i in e.children)
        if before:
            buckets.setdefault(str(e.kind), []).append(after / before)
    return {k: np.array(v) for k, v in buckets.items()}
