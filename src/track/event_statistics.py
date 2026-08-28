from dataclasses import dataclass

import numpy as np

from .link import EventKind, Tracks

COALESCENCE = EventKind.MERGE
BREAKUP = EventKind.SPLIT


def _check_timestep(timestep: float) -> float:
    assert timestep > 0, f"timestep must be positive, got {timestep!r}"
    return float(timestep)


@dataclass(frozen=True)
class EventRates:
    """Summary of how often bubbles coalesce and break up.

    Different Event statistics use different units
      - ``coalescence`` and ``breakup`` are events per unit time.
      - The ``_interval`` properties inert to "one event" every N time units.
      - The ``_per_bubble`` forms divide by the mean bubble count. This is
        useful for comparing cases where (say) there's very different vapor volumes.
    """

    transitions: int
    timestep: float
    mean_bubbles: float
    coalescence: float
    breakup: float
    coalescence_per_bubble: float
    breakup_per_bubble: float

    @property
    def duration(self) -> float:
        """Time spanned by the transitions this was measured over."""
        return self.transitions * self.timestep

    @property
    def coalescence_interval(self) -> float:
        """Mean time between coalescences; ``inf`` if there were none."""
        return 1.0 / self.coalescence if self.coalescence else float("inf")

    @property
    def breakup_interval(self) -> float:
        """Mean time between breakups; ``inf`` if there were none."""
        return 1.0 / self.breakup if self.breakup else float("inf")

    @property
    def breakup_ratio(self) -> float:
        """Breakups per coalescence; ``inf`` if bubbles only ever break up."""
        if self.coalescence == 0:
            return float("inf") if self.breakup else float("nan")
        return self.breakup / self.coalescence

    def __str__(self) -> str:
        def line(name, rate, per_bubble, gap):
            every = "never" if rate == 0 else f"one every {gap:.4g}"
            return (
                f"  {name:<12}{rate:.4g}/time ({per_bubble:.4g} per bubble)"
                f"  -- {every}"
            )

        return "\n".join(
            [
                f"{self.transitions} transitions x timestep {self.timestep:.4g}"
                f" = {self.duration:.4g}, {self.mean_bubbles:.1f} bubbles/frame",
                line(
                    "coalescence",
                    self.coalescence,
                    self.coalescence_per_bubble,
                    self.coalescence_interval,
                ),
                line(
                    "breakup", self.breakup, self.breakup_per_bubble, self.breakup_interval
                ),
            ]
        )


def transitions(tracks: Tracks) -> int:
    """Number of frame-to-frame transitions, the denominator for a rate."""
    return max(len(tracks.ids) - 1, 0)


def duration(tracks: Tracks, timestep: float = 1.0) -> float:
    """Time spanned by a run: transitions times ``timestep``."""
    return transitions(tracks) * _check_timestep(timestep)


def bubble_counts(tracks: Tracks) -> np.ndarray:
    """Regions present in each frame, ``[n_frames]``."""
    return np.array([len(row) for row in tracks.ids], dtype=np.int64)


def event_counts(tracks: Tracks, kind: EventKind) -> np.ndarray:
    """How many events of ``kind`` occur at each transition, ``[n - 1]``.

    Indexed by transition, so entry ``t`` covers frame ``t`` to ``t + 1``.
    Several events can share a transition, which is why this counts rather
    than flags. Counts are unitless -- ``timestep`` does not enter here.
    """
    assert kind in EventKind
    counts = np.zeros(transitions(tracks), dtype=np.int64)
    for event in tracks.events:
        if event.kind is kind and event.time < counts.size:
            counts[event.time] += 1
    return counts


def rates(tracks: Tracks, timestep: float = 1.0) -> EventRates:
    """Overall coalescence and breakup rates, per unit of simulation time."""
    dt = _check_timestep(timestep)
    n = transitions(tracks)
    bubbles = bubble_counts(tracks)
    mean_bubbles = float(bubbles.mean()) if bubbles.size else 0.0

    if n == 0:
        return EventRates(0, dt, mean_bubbles, 0.0, 0.0, 0.0, 0.0)

    span = n * dt
    merges = float(event_counts(tracks, COALESCENCE).sum()) / span
    splits = float(event_counts(tracks, BREAKUP).sum()) / span
    scale = mean_bubbles if mean_bubbles else float("nan")
    return EventRates(
        transitions=n,
        timestep=dt,
        mean_bubbles=mean_bubbles,
        coalescence=merges,
        breakup=splits,
        coalescence_per_bubble=merges / scale,
        breakup_per_bubble=splits / scale,
    )


def rate_over_time(
    tracks: Tracks,
    kind: EventKind,
    window: int = 25,
    per_bubble: bool = False,
    timestep: float = 1.0,
) -> np.ndarray:
    """Rate of ``kind`` smoothed over a centred window, ``[n - 1]``.

    Raw per-transition counts are mostly 0 and 1, so a single one tells you
    little; averaging over ``window`` transitions shows how the rate evolves
    as boiling develops. ``window`` is a count of transitions, not a duration.
    The window is shortened at the ends rather than padded with zeros, so
    early and late values are not biased low.

    With ``per_bubble`` the result is divided by the bubble count in each
    window too, separating "more coalescence" from "more bubbles".
    """
    dt = _check_timestep(timestep)
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    counts = event_counts(tracks, kind).astype(float)
    if counts.size == 0:
        return counts

    kernel = np.ones(min(window, counts.size))
    total = np.convolve(counts, kernel, mode="same")
    samples = np.convolve(np.ones_like(counts), kernel, mode="same")
    rate = total / (samples * dt)

    if per_bubble:
        bubbles = bubble_counts(tracks)[: counts.size].astype(float)
        mean = np.convolve(bubbles, kernel, mode="same") / samples
        rate = np.divide(rate, mean, out=np.zeros_like(rate), where=mean > 0)
    return rate


def intervals(tracks: Tracks, kind: EventKind, timestep: float = 1.0) -> np.ndarray:
    """Elapsed time between successive events of ``kind``.

    The measured counterpart to :attr:`EventRates.coalescence_interval`, which
    is only the reciprocal of a mean rate. Simultaneous events give gaps of
    zero, since several can share one transition.

    Useful as a randomness check: for events arriving independently at a
    steady rate the gaps are exponential, so their standard deviation should
    be close to their mean. Much less spread means the events are pinned to a
    cycle; much more means they arrive in bursts.
    """
    dt = _check_timestep(timestep)
    times = sorted(e.time for e in tracks.events if e.kind is kind)
    if len(times) < 2:
        return np.empty(0)
    return np.diff(np.array(times, dtype=np.int64)) * dt
