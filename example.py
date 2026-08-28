from typing import List

import numpy as np

from track import (
    segment,
    link_by_voxel_overlap,
    Frame,
    Tracks,
    event_counts,
    EventKind
)

mask = np.load("files/saturated_phase_bits.npy")
mask = np.reshape(np.unpackbits(mask), (2001, 512, 512))

frames: List[Frame] = segment(mask, "face", min_size=0)
tracks: Tracks = link_by_voxel_overlap(
    frames,
    min_overlap=0.25,
    criterion="containment",
    max_distance=8.0 # voxels
)

# number of bubbles that split / broke up
merge_count = event_counts(tracks, EventKind.MERGE)
print(f"{sum(merge_count)} mergers")
split_count = event_counts(tracks, EventKind.SPLIT)
print(f"{sum(split_count)} splits")
