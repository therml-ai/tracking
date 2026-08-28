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

from track import Connectivity, Criterion, segment, volume_consistency
from track.link import link_by_voxel_overlap
from track.graph import LINEAGE_KINDS
from track.palette import colours as id_colours


MIN_SIDE = 3.0  # inches; below this, titles and annotations stop being legible
MAX_SIDE = 40.0


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


def _stream(masks, args, chunk: int = 64):
    """Segment in chunks, so a long animation never holds every label image."""
    wraps = periodic_flags(
        masks.ndim - 1, args.left_right_periodic, args.top_bottom_periodic
    )
    for start in range(0, len(masks), chunk):
        yield from segment(
            masks[start : start + chunk], args.connectivity, args.min_size, wraps
        )


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
    p.add_argument("--shape", type=int, nargs="+", help="e.g. --shape 2001 512 512")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="subsample frames; note tracking runs on the subsampled "
                        "sequence, so a large stride weakens overlap linking")
    p.add_argument("--out", default="bubbles.mp4")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--connectivity", type=Connectivity, choices=list(Connectivity),
                   default=Connectivity.FACE)
    p.add_argument("--min-size", type=int, default=4,
                   help="drop regions below this many pixels; keep it small -- "
                        "this discards bubbles outright rather than merging them")
    p.add_argument("--left-right-periodic", action="store_true",
                   help="the domain wraps in X; bubbles crossing it stay one bubble")
    p.add_argument("--top-bottom-periodic", action="store_true",
                   help="the domain wraps in Y")
    p.add_argument("--min-overlap", type=float, default=0.25)
    p.add_argument("--criterion", type=Criterion, choices=list(Criterion),
                   default=Criterion.CONTAINMENT)
    p.add_argument("--max-distance", type=float, default=8.0,
                   help="centroid gate (px) for the fallback matching stage that "
                        "keeps small fast bubbles on one track; 0 disables it")
    p.add_argument("--label-min-size", type=int, default=150,
                   help="annotate ids on regions at least this large (0 disables)")
    p.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True,
                   help="ring bubbles whose volume bookkeeping fails; needs all "
                        "frames resident, so watch memory on long spans")
    p.add_argument("--audit-detached-only", action=argparse.BooleanOptionalAction,
                   default=True, help="only check bubbles clear of the heater")
    p.add_argument("--audit-lower", type=float, default=0.95)
    p.add_argument("--audit-upper", type=float, default=1.15)
    p.add_argument("--audit-sigmas", type=float, default=3.0,
                   help="how many sigma of interface-discretisation slack to "
                        "allow on top of the physical band; 0 for a hard band")
    p.add_argument("--audit-min-volume", type=int, default=25,
                   help="skip events on bubbles too small to measure; below "
                        "~25 voxels the ratio is quantisation noise")
    p.add_argument("--wall-axis", type=int, default=0)
    p.add_argument("--wall-side", type=int, default=0)
    p.add_argument("--layout", choices=["auto", "rows", "columns"], default="auto",
                   help="stack the two panels vertically (rows) or side by side "
                        "(columns); auto picks whichever keeps the figure closer "
                        "to square, so wide domains stack")
    p.add_argument("--fig-width", type=float, default=12.0)
    p.add_argument("--flag-hold", type=int, default=6,
                   help="frames to keep a flagged bubble ringed, so it is visible")
    args = p.parse_args()

    vapor = load(args.input, tuple(args.shape) if args.shape else None)
    wraps = periodic_flags(
        vapor.ndim - 1, args.left_right_periodic, args.top_bottom_periodic
    )
    stop = args.stop if args.stop is not None else len(vapor)
    time_range = list(range(args.start, stop, args.stride))
    print(f"loaded {vapor.shape}, animating {len(time_range)} frames")

    # the audit needs every frame at once; when it is off, stay streaming
    frames = (
        segment(vapor[time_range], args.connectivity, args.min_size, wraps)
        if args.audit
        else None
    )
    tracks = link_by_voxel_overlap(
        frames if frames is not None else _stream(vapor[time_range], args),
        min_overlap=args.min_overlap,
        criterion=args.criterion,
        max_distance=args.max_distance,
    )
    print(f"{tracks.n_tracks} tracks; rendering to {args.out}")

    flagged: dict[int, set[int]] = {}
    alerts: dict[int, list[str]] = {}
    if frames is not None:
        bad = volume_consistency(
            tracks, frames,
            lower=args.audit_lower, upper=args.audit_upper,
            min_volume=args.audit_min_volume, sigmas=args.audit_sigmas,
            detached_only=args.audit_detached_only,
            wall_axis=args.wall_axis, wall_side=args.wall_side,
        )
        print(f"{len(bad)} volume violations")
        for v in bad:
            print(f"  {v}")
            involved = set(v.parents) | set(v.children)
            for k in range(v.time, min(v.time + args.flag_hold, len(time_range))):
                flagged.setdefault(k, set()).update(involved)
                alerts.setdefault(k, []).append(
                    f"!! {v.kind} {v.check} {v.ratio:.2f}"
                )

    # events keyed by the frame they land in, for the on-screen annotation
    notes: dict[int, list[str]] = {}
    for e in tracks.events:
        if e.kind in LINEAGE_KINDS:
            notes.setdefault(e.time + 1, []).append(
                f"{e.kind}: {'+'.join(map(str, e.parents))}"
                f" -> {'+'.join(map(str, e.children))}"
            )

    # a panel is as wide as the domain; two of them side by side make a figure
    # of aspect 2a, stacked they make a/2. Stacking wins once a > 1, nudged up
    # a little so square data keeps the familiar side-by-side.
    panel = vapor.shape[2] / vapor.shape[1]
    stacked = args.layout == "rows" or (args.layout == "auto" and panel > 1.2)
    grid = (2, 1) if stacked else (1, 2)
    combined = panel / 2 if stacked else panel * 2
    width, height = args.fig_width, args.fig_width / combined
    if height < MIN_SIDE:
        # very wide data: grow the figure sideways rather than letterbox it
        width = min(MIN_SIDE * combined, MAX_SIDE)
        height = width / combined
    height = min(height, MAX_SIDE)
    fig, (ax0, ax1) = plt.subplots(
        *grid, figsize=(width, height), constrained_layout=True
    )
    print(
        f"panel aspect {panel:.2f} -> {'stacked' if stacked else 'side by side'}"
        f", figure {width:.1f}x{height:.1f} in"
    )
    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
    ax0.set_title("phase mask (vapor)")
    ax1.set_title("bubble id")

    first = segment(vapor[time_range[:1]], args.connectivity, args.min_size, wraps)[0]
    im0 = ax0.imshow(vapor[time_range[0]], origin="lower", cmap="bone", vmin=0, vmax=1,
                     interpolation="nearest")
    im1 = ax1.imshow(id_colours(tracks.relabel(tracks.ids[0], first.labels)),
                     origin="lower", interpolation="nearest")
    stamp = ax0.text(0.02, 0.97, "", transform=ax0.transAxes, va="top", color="w",
                     fontsize=9, family="monospace")
    note = ax1.text(0.02, 0.97, "", transform=ax1.transAxes, va="top", color="w",
                    fontsize=8, family="monospace")
    alert = ax1.text(0.02, 0.03, "", transform=ax1.transAxes, va="bottom",
                     color="#ff2d55", fontsize=9, family="monospace",
                     fontweight="bold")
    texts: list[plt.Text] = []
    rings: list = []

    def draw(i: int):
        t = time_range[i]
        f = (
            frames[i]
            if frames is not None
            else segment(vapor[t : t + 1], args.connectivity, args.min_size, wraps)[0]
        )
        ids = tracks.ids[i]
        im0.set_data(vapor[t])
        im1.set_data(id_colours(tracks.relabel(ids, f.labels)))
        stamp.set_text(f"frame {t:>5d}\nbubbles {f.count:>3d}")
        note.set_text("\n".join(notes.get(i, [])[:4]))
        alert.set_text("\n".join(dict.fromkeys(alerts.get(i, [])))[:120])

        while rings:
            rings.pop().remove()
        if i in flagged:
            lut = np.zeros(len(ids) + 1, dtype=bool)
            lut[1:] = np.isin(ids, list(flagged[i]))
            hot = lut[f.labels]
            if hot.any():
                for ax in (ax0, ax1):
                    rings.append(
                        ax.contour(hot, levels=[0.5], colors="#ff2d55",
                                   linewidths=1.8)
                    )

        while texts:
            texts.pop().remove()
        if args.label_min_size:
            big = np.flatnonzero(f.size >= args.label_min_size)
            for lab in big:
                y, x = f.centroid[lab][:2]
                texts.append(ax1.text(x, y, str(int(ids[lab])), color="k", fontsize=7,
                                      ha="center", va="center", fontweight="bold"))
        return [im0, im1, stamp, note, alert, *texts]

    anim = FuncAnimation(fig, draw, frames=len(time_range), blit=False)
    writer, out = pick_writer(args.out, args.fps)
    anim.save(out, writer=writer, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
