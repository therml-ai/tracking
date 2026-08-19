"""Frame-to-frame linking of labelled bubbles by voxel overlap."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .overlap import Criterion, score
from .segment import Frame, segment

class EventKind(StrEnum):
    """What happened to a bubble between one frame and the next.

    A ``StrEnum``, so members compare equal to their value and format as
    plain text. Note that libraries dispatching on ``type(value)`` rather
    than ``isinstance`` still need a real ``str`` -- see
    :func:`track.graph.write_graphml`.
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


@dataclass
class Tracks:
    """Per-frame label-to-track mapping plus the topology events."""

    ids: list[np.ndarray]  # ids[t][label - 1] -> track id
    events: list[Event]
    n_tracks: int

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

    Overlap fails for bubbles that travel a large fraction of their own
    radius per frame -- a small bubble simply does not intersect itself
    between frames, however permissive the threshold. Those regions are
    matched here instead, by minimising total centroid distance under a
    ``max_distance`` gate.

    Only regions that overlap *nothing* are eligible, so this can add 1:1
    continuations but can never disturb a coalescence or breakup already
    found by overlap.
    """
    free_a = np.flatnonzero(~adj.any(axis=1))
    free_b = np.flatnonzero(~adj.any(axis=0))
    if free_a.size == 0 or free_b.size == 0:
        return adj

    dist = np.linalg.norm(
        a.centroid[free_a][:, None, :] - b.centroid[free_b][None, :, :], axis=-1
    )
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


def link(
    frames: Iterable[Frame],
    min_overlap: float = 0.5,
    criterion: Criterion = Criterion.CONTAINMENT,
    max_distance: float = 0.0,
) -> Tracks:
    """Link labelled frames into tracks by voxel overlap.

    A parent and child are candidates when their overlap ``criterion``
    reaches ``min_overlap``; each connected component of the resulting
    bipartite graph becomes one event. Identity follows the largest region:
    through a merge the child inherits the biggest parent's id, through a
    split the biggest child keeps it, and every other child starts a new
    track. This is a baseline -- it assumes frames are close enough in time
    that bubbles overlap themselves, which holds for simulation output but
    not for sparsely sampled data.

    ``max_distance`` adds a second pass that matches whatever overlap left
    unpaired by centroid proximity, which is what keeps small fast-moving
    bubbles on one track. Set it to 0 to use overlap alone. Choose it from
    the data: a little above the per-frame displacement of a typical bubble.
    """
    stream = iter(frames)
    prev = next(stream, None)
    if prev is None:
        return Tracks([], [], 0)

    ids = [np.arange(1, prev.count + 1, dtype=np.int64)]
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
            else:
                kind = (
                    EventKind.CONTINUE
                    if prows.size == 1 and ccols.size == 1
                    else EventKind.SPLIT
                    if prows.size == 1
                    else EventKind.MERGE
                    if ccols.size == 1
                    else EventKind.COMPLEX
                )
                heir = int(ccols[np.argmax(cur.size[ccols])])
                cur_ids[heir] = ids[t][prows[np.argmax(prev.size[prows])]]

            for c in ccols:
                if int(c) != heir:
                    cur_ids[c] = next_id
                    next_id += 1

            events.append(
                Event(t, kind, parents, tuple(int(c) for c in cur_ids[ccols]))
            )

        ids.append(cur_ids)
        prev = cur

    return Tracks(ids, events, next_id - 1)


def track(
    masks: Iterable[np.ndarray],
    connectivity: int = 1,
    min_size: int = 0,
    min_overlap: float = 0.5,
    criterion: Criterion = Criterion.CONTAINMENT,
    max_distance: float = 0.0,
) -> Tracks:
    """Segment and link a sequence of phase masks (True = vapor).

    ``masks`` is consumed lazily, so only two frames are labelled at a time.
    """
    frames = (segment(m, connectivity, min_size) for m in masks)
    return link(
        frames,
        min_overlap=min_overlap,
        criterion=criterion,
        max_distance=max_distance,
    )
