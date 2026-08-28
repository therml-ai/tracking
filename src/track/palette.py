"""Stable colours for track ids, used by the visualisation scripts.
"""

import numpy as np

# irrational stride round the hue circle: consecutive ids land far apart, and
# the mapping is stateless, so a track id keeps its colour for its whole life
GOLDEN = 0.6180339887498949
LIQUID = np.array([0.04, 0.05, 0.10])


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i %= 6
    pick = [i == k for k in range(6)]
    return np.stack(
        [
            np.select(pick, [v, q, p, p, t, v]),
            np.select(pick, [t, v, v, q, p, p]),
            np.select(pick, [p, p, t, v, v, q]),
        ],
        axis=-1,
    )


def colours(ids) -> np.ndarray:
    """Map track ids to RGB in ``[0, 1]``; id 0 is liquid background."""
    ids = np.asarray(ids)
    rgb = _hsv_to_rgb(
        (ids * GOLDEN) % 1.0,
        np.where(ids % 2 == 0, 0.62, 0.85),
        np.where(ids % 3 == 0, 0.99, 0.82),
    )
    return np.where((ids == 0)[..., None], LIQUID, rgb)
