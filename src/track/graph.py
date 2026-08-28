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

    Every parent of a coalescence or breakup gets an edge to every product,
    because no id survives a topology change. A track that merely continues
    stays a single node with no edge. Since ids are only ever minted fresh,
    edges always run forward in time and the graph is acyclic.
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
                graph.add_edge(parent, child, kind=event.kind, time=event.time)
    return graph


def write_graphml(graph, path) -> None:
    """Write the lineage graph for Gephi, Cytoscape or igraph.
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
