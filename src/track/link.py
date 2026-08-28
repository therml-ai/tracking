"""Frame-to-frame linking of labelled bubbles by voxel overlap."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .overlap import Criterion, score
from .segment import Connectivity, Frame, minimum_image, segment

class EventKind(StrEnum):
    """What happened to a bubble between one frame and the next.
    """
    START = "start"  #: nucleation, or entry into the domain
    END = "end"  #: condensation, or exit from the domain
    CONTINUE = "continue"  #: one bubble, still one bubble
    MERGE = "merge"  #: coalescence
    SPLIT = "split"  #: breakup, or departure from the wall film
    COMPLEX = "complex"  #: many-to-many, unresolved by this linker


@dataclass(frozen=True)
class Event:
    """A topology change between frame ``time`` and ``time + 1``.

    ``parents`` and ``children`` are track ids.
    """

    time: int
    kind: EventKind
    parents: tuple[int, ...]
    children: tuple[int, ...]


@dataclass(frozen=True)
class Trajectory:
    """One track's history: where it was and how big it was, frame by frame."""

    track: int
    frames: np.ndarray  # int, the frame indices where this track exists
    centroid: np.ndarray  # float [n, ndim], in index space
    volume: np.ndarray  # int [n], voxel counts
    shape: tuple[int, ...] = ()
    periodic: tuple[bool, ...] = ()

    def __len__(self) -> int:
        return int(self.frames.size)

    @property
    def centroid_displacement(self) -> np.ndarray:
        delta = np.diff(self.centroid, axis=0)
        if any(self.periodic):
            return minimum_image(delta, self.shape, self.periodic)
        return delta

    @property
    def volume_change(self) -> np.ndarray:
        return np.diff(self.volume) / self.volume[:-1]


@dataclass
class Tracks:
    """Per-frame label-to-track mapping, region properties and events.

    ``ids``, ``size`` and ``centroid`` share an indexing: entry ``[t][i]``
    describes label ``i + 1`` of frame ``t``.
    """

    ids: list[np.ndarray]  # ids[t][label - 1] -> track id
    size: list[np.ndarray]  # voxel count per region
    centroid: list[np.ndarray]  # [count, ndim] per frame
    events: list[Event]
    n_tracks: int
    shape: tuple[int, ...] = ()  # spatial shape of a frame
    periodic: tuple[bool, ...] = ()  # per axis, carried from the frames

    def trajectory(self, track_id: int) -> Trajectory:
        """Centroid and volume history of one track."""
        frames, cents, vols = [], [], []
        for t, row in enumerate(self.ids):
            hit = np.flatnonzero(row == track_id)
            if hit.size:
                i = int(hit[0])
                frames.append(t)
                cents.append(self.centroid[t][i])
                vols.append(self.size[t][i])
        if not frames:
            raise KeyError(f"no track {track_id}")
        return Trajectory(
            track_id,
            np.array(frames),
            np.array(cents),
            np.array(vols, dtype=np.int64),
            self.shape,
            self.periodic,
        )

    def trajectories(self) -> dict[int, Trajectory]:
        """Every track's history, in a single pass over the frames."""
        acc: dict[int, tuple[list, list, list]] = {}
        for t, row in enumerate(self.ids):
            for i, tid in enumerate(row.tolist()):
                frames, cents, vols = acc.setdefault(tid, ([], [], []))
                frames.append(t)
                cents.append(self.centroid[t][i])
                vols.append(self.size[t][i])
        return {
            tid: Trajectory(
                tid,
                np.array(f),
                np.array(c),
                np.array(v, dtype=np.int64),
                self.shape,
                self.periodic,
            )
            for tid, (f, c, v) in acc.items()
        }

    def of_kind(self, kind: EventKind) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def lifetimes(self) -> dict[int, tuple[int, int]]:
        """Map each track id to its ``(first_frame, last_frame)``, inclusive."""
        span: dict[int, tuple[int, int]] = {}
        for t, row in enumerate(self.ids):
            for tid in row.tolist():
                first, _ = span.get(tid, (t, t))
                span[tid] = (first, t)
        return span

    def relabel(self, frame_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recolour a frame's label image by track id (0 stays liquid)."""
        lut = np.zeros(len(frame_ids) + 1, dtype=np.int64)
        lut[1:] = frame_ids
        return lut[labels]


def _augment_by_distance(
    adj: np.ndarray, a: Frame, b: Frame, max_distance: float
) -> np.ndarray:
    """Link leftover regions by a global nearest-centroid assignment.

    Only regions that do NOT overlap are eligible, so this can add 1:1
    continuations but can never disturb a coalescence or breakup already
    found by overlap.
    """
    free_a = np.flatnonzero(~adj.any(axis=1))
    free_b = np.flatnonzero(~adj.any(axis=0))
    if free_a.size == 0 or free_b.size == 0:
        return adj

    delta = a.centroid[free_a][:, None, :] - b.centroid[free_b][None, :, :]
    if any(a.wraps):
        delta = minimum_image(delta, a.shape, a.wraps)
    dist = np.linalg.norm(delta, axis=-1)
    # linear_sum_assignment needs a feasible matrix, so gate with a large
    # finite cost and discard over-gate pairs after solving
    barrier = max_distance * 1e3 + 1.0
    rows, cols = linear_sum_assignment(np.where(dist <= max_distance, dist, barrier))
    ok = dist[rows, cols] <= max_distance
    adj[free_a[rows[ok]], free_b[cols[ok]]] = True
    return adj


def _groups(adj: np.ndarray, n1: int, n2: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(parent_rows, child_cols)`` for each component of the graph."""
    rows, cols = np.nonzero(adj)
    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, n1 + cols)),
        shape=(n1 + n2, n1 + n2),
    )
    _, comp = connected_components(graph, directed=False)
    order = np.argsort(comp, kind="stable")
    bounds = np.flatnonzero(np.diff(comp[order])) + 1
    for nodes in np.split(order, bounds):
        yield nodes[nodes < n1], nodes[nodes >= n1] - n1


def link_by_voxel_overlap(
    frames: Iterable[Frame],
    min_overlap: float = 0.5,
    criterion: Criterion = Criterion.CONTAINMENT,
    max_distance: float = 0.0,
) -> Tracks:
    """Link labelled frames into tracks by voxel overlap. This assumes frames are
    close enough in time for bubbles to overlap themselves.

    - A parent and child are candidates when their overlap ``criterion`` reaches ``min_overlap``.
    - IDs follow the largest region: in a merge the child inherits the biggest parent's ID,
      through a split the biggest child keeps the ID, and the other children get new IDs.
    - ``max_distance`` adds a second pass that matches whatever overlap left
      unpaired by centroid proximity. This may be needed for small, fast bubbles.
    """
    stream = iter(frames)
    prev = next(stream, None)
    if prev is None:
        return Tracks([], [], [], [], 0)

    first = prev
    ids = [np.arange(1, prev.count + 1, dtype=np.int64)]
    sizes = [prev.size]
    cents = [prev.centroid]
    next_id = prev.count + 1
    events: list[Event] = []

    for t, cur in enumerate(stream):
        adj = score(prev, cur, criterion) >= min_overlap
        if max_distance > 0:
            adj = _augment_by_distance(adj, prev, cur, max_distance)
        cur_ids = np.zeros(cur.count, dtype=np.int64)

        for prows, ccols in _groups(adj, prev.count, cur.count):
            parents = tuple(int(i) for i in ids[t][prows])

            if ccols.size == 0:
                events.append(Event(t, EventKind.END, parents, ()))
                continue

            if prows.size == 0:
                kind = EventKind.START
                heir = -1  # nothing to inherit from
            elif prows.size == 1 and ccols.size == 1:
                # the one case where identity carries over
                kind = EventKind.CONTINUE
                heir = int(ccols[0])
                cur_ids[heir] = ids[t][prows[0]]
            else:
                kind = (
                    EventKind.SPLIT
                    if prows.size == 1
                    else EventKind.MERGE
                    if ccols.size == 1
                    else EventKind.COMPLEX
                )
                heir = -1  # every product of a topology change is a new bubble

            for c in ccols:
                if int(c) != heir:
                    cur_ids[c] = next_id
                    next_id += 1

            events.append(
                Event(t, kind, parents, tuple(int(c) for c in cur_ids[ccols]))
            )

        ids.append(cur_ids)
        sizes.append(cur.size)
        cents.append(cur.centroid)
        prev = cur

    return Tracks(
        ids, sizes, cents, events, next_id - 1, first.shape, first.wraps
    )


def track(
    masks: Iterable[np.ndarray],
    connectivity: Connectivity = Connectivity.FACE,
    min_size: int = 0,
    min_overlap: float = 0.5,
    criterion: Criterion = Criterion.CONTAINMENT,
    max_distance: float = 0.0,
    periodic=None,
) -> Tracks:
    """Segment and link a sequence of phase masks (True = vapor).
    ``masks`` is consumed lazily, so only two frames are labelled at a time.
    """
    frames = (segment(m, connectivity, min_size, periodic) for m in masks)
    return link_by_voxel_overlap(
        frames,
        min_overlap=min_overlap,
        criterion=criterion,
        max_distance=max_distance,
    )
