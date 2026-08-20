"""Draw the bubble lineage graph recovered by tracking.

    uv run --group viz scripts/lineage.py files/phase_bits.npy \
        --shape 2001 512 512 --start 500 --stop 700 --out lineage.png

Each track is a horizontal bar spanning its lifetime, coloured by the same id
palette as the movie, so a bar's colour matches the bubble on screen. Vertical
connectors are events: a coalescence pulls two bars into one, a breakup sends
one bar into two. Rows are grouped into families -- weakly connected
components of the lineage graph -- separated by blank rows.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from track import segment
from track import EventKind
from track.graph import families, to_networkx, write_graphml
from track.link import link_by_voxel_overlap
from track.palette import colours

EVENT_STYLE = {
    EventKind.MERGE: ("#ff6b4a", "coalescence"),
    EventKind.SPLIT: ("#4ad9ff", "breakup"),
    EventKind.COMPLEX: ("#e05cff", "complex"),
}


def load(path, shape):
    raw = np.load(path)
    if raw.dtype == np.uint8 and raw.ndim == 1:
        raw = np.unpackbits(raw)
    if shape:
        n = int(np.prod(shape))
        if raw.size < n:
            sys.exit(f"file holds {raw.size} values, too few for shape {shape}")
        raw = raw[:n].reshape(shape)
    return raw.astype(bool)


def periodic_flags(ndim: int, left_right: bool, top_bottom: bool) -> tuple[bool, ...]:
    """Map the two named boundaries onto per-axis flags.

    Data is laid out ``[(Z,) Y, X]``, so left-right is the last axis and
    top-bottom the one before it.
    """
    flags = [False] * ndim
    if left_right:
        flags[-1] = True
    if top_bottom:
        flags[-2] = True
    return tuple(flags)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--shape", type=int, nargs="+")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--out", default="lineage.png")
    p.add_argument("--connectivity", type=int, default=1)
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--left-right-periodic", action="store_true",
                   help="the domain wraps in X; bubbles crossing it stay one bubble")
    p.add_argument("--top-bottom-periodic", action="store_true",
                   help="the domain wraps in Y")
    p.add_argument("--min-overlap", type=float, default=0.25)
    p.add_argument("--max-distance", type=float, default=8.0)
    p.add_argument("--min-duration", type=int, default=3,
                   help="hide tracks shorter than this many frames")
    p.add_argument("--top", type=int, default=0,
                   help="draw only the N largest families (0 = all)")
    p.add_argument("--graphml", help="also write the graph for Gephi/Cytoscape")
    args = p.parse_args()

    vapor = load(args.input, tuple(args.shape) if args.shape else None)
    wraps = periodic_flags(
        vapor.ndim - 1, args.left_right_periodic, args.top_bottom_periodic
    )
    stop = args.stop if args.stop is not None else len(vapor)
    sel = list(range(args.start, stop))

    tracks = link_by_voxel_overlap(
        (segment(vapor[t], args.connectivity, args.min_size, wraps) for t in sel),
        min_overlap=args.min_overlap,
        max_distance=args.max_distance,
    )
    graph = to_networkx(tracks)
    print(f"{graph.number_of_nodes()} tracks, {graph.number_of_edges()} lineage edges")

    if args.graphml:
        write_graphml(graph, args.graphml)
        print(f"wrote {args.graphml}")

    groups = families(graph)
    # a family is worth a row only if something in it persists
    keep = [
        g for g in groups
        if max(graph.nodes[n]["duration"] for n in g) >= args.min_duration
    ]
    if args.top:
        keep = keep[: args.top]
    if not keep:
        sys.exit("nothing to draw; lower --min-duration")

    row, ypos = 0, {}
    for group in keep:
        for node in sorted(group, key=lambda n: (graph.nodes[n]["start"], n)):
            if graph.nodes[node]["duration"] >= args.min_duration or graph.degree(node):
                ypos[node] = row
                row += 1
        row += 1  # blank row between families

    height = max(3.0, min(26.0, 0.13 * row))
    fig, ax = plt.subplots(figsize=(13, height), constrained_layout=True)

    for node, y in ypos.items():
        d = graph.nodes[node]
        ax.plot(
            [args.start + d["start"], args.start + d["end"]], [y, y],
            color=colours(np.array(node)), linewidth=2.4, solid_capstyle="butt",
        )

    for u, v, d in graph.edges(data=True):
        if u in ypos and v in ypos:
            x = args.start + d["time"] + 0.5
            ax.plot([x, x], [ypos[u], ypos[v]],
                    color=EVENT_STYLE[d["kind"]][0], linewidth=0.9, alpha=0.85)

    counts = {k: sum(1 for *_, d in graph.edges(data=True) if d["kind"] == k)
              for k in EVENT_STYLE}
    ax.legend(
        handles=[Line2D([], [], color=c, lw=2, label=f"{lab} ({counts[k]})")
                 for k, (c, lab) in EVENT_STYLE.items()],
        loc="lower right", fontsize=9, framealpha=0.9,
    )
    ax.set_xlabel("frame")
    ax.set_ylabel(f"track  ({len(keep)} families, {len(ypos)} shown)")
    ax.set_yticks([])
    ax.set_ylim(-1, row)
    ax.invert_yaxis()  # largest families first, reading top-down
    ax.set_title("bubble lineage: bars are tracks, connectors are events")
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
