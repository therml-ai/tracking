"""The track lineage as a graph: which bubble came from which.

The movie shows where bubbles are; this shows how they are related. Nodes are
tracks, edges are coalescence and breakup events, and a weakly connected
component is one family descended from a common ancestor.

``networkx`` is an optional dependency, imported lazily.
"""

from .link import EventKind, Tracks

#: events that create a parent -> child edge; the rest concern one track only
LINEAGE_KINDS = frozenset({EventKind.MERGE, EventKind.SPLIT, EventKind.COMPLEX})


def _networkx():
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "networkx is required for track.graph; install the viz group: "
            "uv sync --group viz"
        ) from exc
    return nx


def to_networkx(tracks: Tracks):
    """Build a ``DiGraph`` of track lineage.

    Nodes carry ``start``, ``end`` and ``duration`` (frame indices into the
    sequence that was linked). Edges carry ``kind`` and ``time``, and run from
    a parent track to a child track.

    Identity-preserving links leave no edge: through a merge the largest
    parent keeps its id, so only the *absorbed* parents get an edge to it, and
    a track that merely continues stays a single node.
    """
    nx = _networkx()
    graph = nx.DiGraph()
    for tid, (start, end) in tracks.lifetimes().items():
        graph.add_node(tid, start=start, end=end, duration=end - start + 1)
    for event in tracks.events:
        if event.kind not in LINEAGE_KINDS:
            continue
        for parent in event.parents:
            for child in event.children:
                if parent != child:  # the heir keeps its id; that is not an edge
                    graph.add_edge(parent, child, kind=event.kind, time=event.time)
    return graph


def write_graphml(graph, path) -> None:
    """Write the lineage graph for Gephi, Cytoscape or igraph.

    The GraphML writer dispatches on ``type(value)`` exactly, so it rejects
    the :class:`~track.link.EventKind` members even though they are strings.
    They are flattened on the way out, leaving the in-memory graph untouched.
    """
    nx = _networkx()
    out = graph.copy()
    for _, _, data in out.edges(data=True):
        data["kind"] = str(data["kind"])
    nx.write_graphml(out, path)


def families(graph) -> list[set[int]]:
    """Weakly connected components -- one bubble family each, largest first."""
    nx = _networkx()
    return sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
