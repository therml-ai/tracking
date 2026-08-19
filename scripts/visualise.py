"""Animate a phase mask alongside the bubble ids recovered by tracking.

    uv run --group viz scripts/visualise.py files/phase_bits.npy \
        --shape 2001 512 512 --stop 400 --out bubbles.mp4

The left panel is the raw vapor mask; the right panel colours every bubble by
its track id, so a colour that persists means identity was held across frames
and a colour that changes marks a coalescence or breakup event.
"""

import argparse
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from track import Criterion, segment
from track.link import link
from track.graph import LINEAGE_KINDS
from track.palette import colours as id_colours


def ffmpeg_works() -> bool:
    """matplotlib only checks that the binary exists; a broken Homebrew ffmpeg
    (missing codec dylibs) passes that check and then dies mid-encode."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def pick_writer(out: str, fps: int) -> tuple[object, str]:
    if not out.endswith(".gif"):
        if ffmpeg_works():
            return FFMpegWriter(fps=fps, bitrate=3200), out
        out = out.rsplit(".", 1)[0] + ".gif"
        print("ffmpeg unusable, falling back to", out, file=sys.stderr)
    return PillowWriter(fps=fps), out


def load(path: str, shape: tuple[int, ...] | None) -> np.ndarray:
    raw = np.load(path)
    if raw.dtype == np.uint8 and raw.ndim == 1:
        raw = np.unpackbits(raw)
    if shape:
        n = int(np.prod(shape))
        if raw.size < n:
            sys.exit(f"file holds {raw.size} values, too few for shape {shape}")
        raw = raw[:n].reshape(shape)  # trailing bits are packbits padding
    if raw.ndim != 3:
        sys.exit(f"expected [T, Y, X] after reshape, got {raw.shape}; pass --shape")
    return raw.astype(bool)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--shape", type=int, nargs="+", help="e.g. --shape 2001 512 512")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="subsample frames; note tracking runs on the subsampled "
                        "sequence, so a large stride weakens overlap linking")
    p.add_argument("--out", default="bubbles.mp4")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--connectivity", type=int, default=1)
    p.add_argument("--min-size", type=int, default=4,
                   help="drop regions below this many pixels; keep it small -- "
                        "this discards bubbles outright rather than merging them")
    p.add_argument("--min-overlap", type=float, default=0.25)
    p.add_argument("--criterion", type=Criterion, choices=list(Criterion),
                   default=Criterion.CONTAINMENT)
    p.add_argument("--max-distance", type=float, default=8.0,
                   help="centroid gate (px) for the fallback matching stage that "
                        "keeps small fast bubbles on one track; 0 disables it")
    p.add_argument("--label-min-size", type=int, default=150,
                   help="annotate ids on regions at least this large (0 disables)")
    args = p.parse_args()

    vapor = load(args.input, tuple(args.shape) if args.shape else None)
    sel = range(args.start, args.stop if args.stop is not None else len(vapor), args.stride)
    sel = list(sel)
    print(f"loaded {vapor.shape}, animating {len(sel)} frames")

    def frames_of(indices):
        for t in indices:
            yield segment(vapor[t], args.connectivity, args.min_size)

    tracks = link(frames_of(sel), min_overlap=args.min_overlap,
                  criterion=args.criterion, max_distance=args.max_distance)
    print(f"{tracks.n_tracks} tracks; rendering to {args.out}")

    # events keyed by the frame they land in, for the on-screen annotation
    notes: dict[int, list[str]] = {}
    for e in tracks.events:
        if e.kind in LINEAGE_KINDS:
            notes.setdefault(e.time + 1, []).append(
                f"{e.kind}: {'+'.join(map(str, e.parents))}"
                f" -> {'+'.join(map(str, e.children))}"
            )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)
    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
    ax0.set_title("phase mask (vapor)")
    ax1.set_title("bubble id")

    first = segment(vapor[sel[0]], args.connectivity, args.min_size)
    im0 = ax0.imshow(vapor[sel[0]], origin="lower", cmap="bone", vmin=0, vmax=1,
                     interpolation="nearest")
    im1 = ax1.imshow(id_colours(tracks.relabel(tracks.ids[0], first.labels)),
                     origin="lower", interpolation="nearest")
    stamp = ax0.text(0.02, 0.97, "", transform=ax0.transAxes, va="top", color="w",
                     fontsize=9, family="monospace")
    note = ax1.text(0.02, 0.97, "", transform=ax1.transAxes, va="top", color="w",
                    fontsize=8, family="monospace")
    texts: list[plt.Text] = []

    def draw(i: int):
        t = sel[i]
        f = segment(vapor[t], args.connectivity, args.min_size)
        ids = tracks.ids[i]
        im0.set_data(vapor[t])
        im1.set_data(id_colours(tracks.relabel(ids, f.labels)))
        stamp.set_text(f"frame {t:>5d}\nbubbles {f.count:>3d}")
        note.set_text("\n".join(notes.get(i, [])[:4]))

        while texts:
            texts.pop().remove()
        if args.label_min_size:
            big = np.flatnonzero(f.size >= args.label_min_size)
            for lab in big:
                y, x = f.centroid[lab][:2]
                texts.append(ax1.text(x, y, str(int(ids[lab])), color="k", fontsize=7,
                                      ha="center", va="center", fontweight="bold"))
        return [im0, im1, stamp, note, *texts]

    anim = FuncAnimation(fig, draw, frames=len(sel), blit=False)
    writer, out = pick_writer(args.out, args.fps)
    anim.save(out, writer=writer, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
