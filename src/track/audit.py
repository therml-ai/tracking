from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .link import EventKind, Tracks
from .segment import Frame, touches_wall

TOPOLOGY = (EventKind.MERGE, EventKind.SPLIT, EventKind.COMPLEX)


@dataclass(frozen=True)
class Violation:
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
    """
    return [
        dict(zip(ids.tolist(), size.tolist()))
        for ids, size in zip(tracks.ids, tracks.size)
    ]


def attached_by_track(
    tracks: Tracks, frames: Sequence[Frame], axis: int = 0, side: int = 0
) -> list[dict[int, bool]]:
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
