import numpy as np
import pytest

from track import Criterion, EventKind, intersection, link_by_voxel_overlap, segment, track


def disc(shape, center, radius):
    idx = np.indices(shape)
    return sum((i - c) ** 2 for i, c in zip(idx, center)) <= radius**2


def test_segment_counts_and_sizes():
    m = np.zeros((8, 8), bool)
    m[1:3, 1:3] = True
    m[5:8, 4:7] = True
    f = segment(m)
    assert f.count == 2
    assert sorted(f.size) == [4, 9]


def test_min_size_drops_and_renumbers():
    m = np.zeros((8, 8), bool)
    m[0, 0] = True  # speck
    m[4:7, 4:7] = True
    f = segment(m, min_size=2)
    assert f.count == 1
    assert f.labels.max() == 1  # contiguous after the drop
    assert f.size.tolist() == [9]


def test_corner_touch_split_by_connectivity():
    m = np.zeros((6, 6), bool)
    m[1:3, 1:3] = True
    m[3:5, 3:5] = True  # touches the first only at a corner
    assert segment(m, connectivity=1).count == 2
    assert segment(m, connectivity=2).count == 1


def test_intersection_is_exact():
    a = np.zeros((6, 6), bool)
    a[1:4, 1:4] = True
    b = np.zeros((6, 6), bool)
    b[2:5, 2:5] = True
    inter = intersection(segment(a), segment(b))
    assert inter.shape == (1, 1)
    assert inter[0, 0] == 4  # the 2x2 corner they share


def test_translation_keeps_one_track():
    shape = (40, 40)
    masks = [disc(shape, (20, 8 + t), 6) for t in range(10)]
    tr = track(masks)
    assert tr.n_tracks == 1
    assert all(row.tolist() == [1] for row in tr.ids)
    assert {e.kind for e in tr.events} == {EventKind.CONTINUE}


def test_merge_mints_a_new_id_and_keeps_both_parents():
    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), 14)
    tr = track([before, joined], min_overlap=0.25)
    merges = tr.of_kind(EventKind.MERGE)
    assert len(merges) == 1
    assert sorted(merges[0].parents) == [1, 2]
    assert len(merges[0].children) == 1
    # the coalesced bubble is a new track, not either parent
    child, = merges[0].children
    assert child not in merges[0].parents
    assert tr.n_tracks == 3


def test_split_is_detected():
    shape = (40, 60)
    whole = disc(shape, (20, 30), 11)
    # a pinch-off: both fragments still lie inside the parent's footprint
    parted = disc(shape, (20, 24), 5) | disc(shape, (20, 36), 5)
    tr = track([whole, parted])
    splits = tr.of_kind(EventKind.SPLIT)
    assert len(splits) == 1
    assert splits[0].parents == (1,)
    assert len(splits[0].children) == 2
    # neither fragment continues the parent's identity
    assert 1 not in splits[0].children
    assert tr.n_tracks == 3


def test_nucleation_and_disappearance():
    shape = (30, 30)
    empty = np.zeros(shape, bool)
    blob = disc(shape, (15, 15), 5)
    tr = track([empty, blob, blob, empty])
    assert len(tr.of_kind(EventKind.START)) == 1
    assert len(tr.of_kind(EventKind.END)) == 1
    assert tr.n_tracks == 1


def test_disjoint_bubbles_get_distinct_tracks():
    shape = (40, 40)
    m = disc(shape, (10, 10), 5) | disc(shape, (30, 30), 5)
    tr = track([m, m])
    assert tr.n_tracks == 2
    assert sorted(tr.ids[-1].tolist()) == [1, 2]


def test_containment_survives_merge_where_iou_fails():
    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), 14)
    assert len(track([before, joined], criterion=Criterion.CONTAINMENT).of_kind(EventKind.MERGE)) == 1
    # IoU at the same threshold cannot hold both parents to the child
    assert track([before, joined], criterion=Criterion.IOU).of_kind(EventKind.MERGE) == []


def test_empty_input():
    tr = track([])
    assert tr.n_tracks == 0 and tr.ids == []


def test_unknown_criterion_rejected():
    with pytest.raises(ValueError, match="unknown criterion"):
        track([np.ones((4, 4), bool)] * 2, criterion="nope")


def test_works_in_3d():
    shape = (20, 20, 20)
    masks = [disc(shape, (10, 10, 6 + t), 4) for t in range(5)]
    tr = track(masks)
    assert tr.n_tracks == 1
    assert all(row.tolist() == [1] for row in tr.ids)


def test_small_fast_bubble_needs_the_distance_stage():
    """A bubble moving ~its own radius per frame never overlaps itself."""
    shape = (40, 60)
    masks = [disc(shape, (20, 8 + 3 * t), 2) for t in range(8)]  # r=2, step=3

    overlap_only = track(masks, min_overlap=0.25)
    assert overlap_only.n_tracks == 8  # a brand new id every single frame

    with_distance = track(masks, min_overlap=0.25, max_distance=8.0)
    assert with_distance.n_tracks == 1
    assert all(row.tolist() == [1] for row in with_distance.ids)


def test_distance_gate_is_respected():
    shape = (40, 80)
    far = [disc(shape, (20, 10), 3), disc(shape, (20, 60), 3)]  # 50 px jump
    assert track(far, max_distance=8.0).n_tracks == 2  # too far, not linked
    assert track(far, max_distance=60.0).n_tracks == 1


def test_distance_stage_leaves_merges_alone():
    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), 14)
    plain = track([before, joined], min_overlap=0.25)
    gated = track([before, joined], min_overlap=0.25, max_distance=8.0)
    assert len(plain.of_kind(EventKind.MERGE)) == len(gated.of_kind(EventKind.MERGE)) == 1
    assert plain.n_tracks == gated.n_tracks


def test_distance_matching_stays_one_to_one():
    """Two parents competing for one child may only claim it once."""
    shape = (40, 40)
    two = disc(shape, (14, 20), 2) | disc(shape, (26, 20), 2)
    one = disc(shape, (20, 20), 2)  # within the gate of both parents
    tr = track([two, one], min_overlap=0.99, max_distance=10.0)
    assert sum(len(e.children) for e in tr.events if e.kind == EventKind.CONTINUE) == 1
    assert len(tr.of_kind(EventKind.END)) == 1  # the loser ends, it is not co-assigned


def test_distance_stage_preserves_identity_of_neighbours():
    """Adjacent small bubbles must not swap ids as they drift."""
    shape = (40, 60)
    masks = [
        disc(shape, (14, 10 + 3 * t), 2) | disc(shape, (26, 10 + 3 * t), 2)
        for t in range(6)
    ]
    tr = track(masks, min_overlap=0.25, max_distance=8.0)
    assert tr.n_tracks == 2
    # each frame carries both original ids, in a stable order
    assert all(sorted(row.tolist()) == [1, 2] for row in tr.ids)
    assert all(row.tolist() == tr.ids[0].tolist() for row in tr.ids)


def test_lineage_graph_links_every_parent_to_the_product():
    nx = pytest.importorskip("networkx")
    from track.graph import families, to_networkx

    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), 14)
    g = to_networkx(track([before, joined], min_overlap=0.25))

    # both parents point at the new bubble; no id survives the merge
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    assert all(d["kind"] == EventKind.MERGE for *_, d in g.edges(data=True))
    assert sorted(n for n, deg in g.out_degree() if deg == 0) == [3]
    assert nx.is_directed_acyclic_graph(g)
    assert len(families(g)) == 1  # the three tracks form one family


def test_lineage_graph_has_no_edges_without_events():
    pytest.importorskip("networkx")
    from track.graph import to_networkx

    shape = (40, 40)
    masks = [disc(shape, (20, 8 + t), 6) for t in range(6)]
    g = to_networkx(track(masks))
    assert g.number_of_nodes() == 1
    assert g.number_of_edges() == 0


def test_lineage_node_attributes_match_lifetimes():
    pytest.importorskip("networkx")
    from track.graph import to_networkx

    shape = (30, 30)
    blob = disc(shape, (15, 15), 5)
    tr = track([np.zeros(shape, bool), blob, blob, blob])
    g = to_networkx(tr)
    assert g.nodes[1] == {"start": 1, "end": 3, "duration": 3}
    assert tr.lifetimes()[1] == (1, 3)


def test_split_produces_a_lineage_edge():
    pytest.importorskip("networkx")
    from track.graph import to_networkx

    shape = (40, 60)
    whole = disc(shape, (20, 30), 11)
    parted = disc(shape, (20, 24), 5) | disc(shape, (20, 36), 5)
    g = to_networkx(track([whole, parted]))
    assert [d["kind"] for *_, d in g.edges(data=True)] == [EventKind.SPLIT] * 2
    assert g.in_degree(1) == 0 and g.out_degree(1) == 2  # parent feeds both


def test_event_kind_is_a_string_enum():
    """Members must stay usable wherever the old bare strings were."""
    assert EventKind.MERGE == "merge"
    assert f"{EventKind.SPLIT}" == "split"
    assert EventKind("complex") is EventKind.COMPLEX
    assert {EventKind.END: 1}["end"] == 1  # hashes like its value
    with pytest.raises(ValueError):
        EventKind("not-a-kind")


def test_every_emitted_kind_is_an_enum_member():
    shape = (40, 60)
    masks = [
        np.zeros(shape, bool),
        disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5),
        disc(shape, (20, 30), 14),
        np.zeros(shape, bool),
    ]
    tr = track(masks, min_overlap=0.25)
    assert tr.events
    for e in tr.events:
        assert isinstance(e.kind, EventKind)


def test_graphml_export_flattens_the_enum():
    pytest.importorskip("networkx")
    import networkx as nx

    from track.graph import to_networkx, write_graphml

    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    g = to_networkx(track([before, disc(shape, (20, 30), 14)], min_overlap=0.25))
    path = "/tmp/_lineage_test.graphml"
    write_graphml(g, path)
    back = nx.read_graphml(path)
    assert [d["kind"] for *_, d in back.edges(data=True)] == ["merge"] * 2
    # the in-memory graph still holds real enum members
    assert all(isinstance(d["kind"], EventKind) for *_, d in g.edges(data=True))


def test_criterion_is_a_string_enum():
    assert Criterion.IOU == "iou"
    assert Criterion("containment") is Criterion.CONTAINMENT
    assert [c.value for c in Criterion] == ["containment", "iou"]


def test_criterion_accepts_member_or_plain_string():
    """Existing string call sites must keep working."""
    shape = (40, 40)
    masks = [disc(shape, (20, 8 + t), 6) for t in range(5)]
    by_member = track(masks, criterion=Criterion.IOU)
    by_string = track(masks, criterion="iou")
    assert by_member.n_tracks == by_string.n_tracks == 1


def test_score_dispatches_to_the_named_measure():
    from track import containment, iou, score

    shape = (40, 60)
    a = segment(disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5))
    b = segment(disc(shape, (20, 30), 14))
    assert np.array_equal(score(a, b, Criterion.CONTAINMENT), containment(a, b))
    assert np.array_equal(score(a, b, Criterion.IOU), iou(a, b))


def test_unknown_criterion_names_the_valid_options():
    with pytest.raises(ValueError, match="unknown criterion .*containment, iou"):
        track([np.ones((4, 4), bool)] * 2, criterion="nope")


def test_link_by_voxel_overlap_takes_pre_segmented_frames():
    """The lower-level entry point the scripts use: Frames in, Tracks out."""
    shape = (40, 40)
    frames = [segment(disc(shape, (20, 8 + t), 6)) for t in range(6)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    assert tr.n_tracks == 1
    assert [e.kind for e in tr.events] == [EventKind.CONTINUE] * 5
    # identical to running the mask-level convenience wrapper
    masks = [disc(shape, (20, 8 + t), 6) for t in range(6)]
    assert tr.n_tracks == track(masks, min_overlap=0.25).n_tracks


def test_no_id_survives_a_topology_change():
    """An id must denote one physical bubble for its whole life."""
    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), 14)
    parted = disc(shape, (20, 24), 5) | disc(shape, (20, 36), 5)
    tr = track([before, joined, joined, parted], min_overlap=0.25)

    for e in tr.events:
        if e.kind in (EventKind.MERGE, EventKind.SPLIT, EventKind.COMPLEX):
            assert not set(e.parents) & set(e.children), e
    # a merge then a split: 2 parents -> 1 product -> 2 fragments = 5 ids
    assert tr.n_tracks == 5


def test_continuation_still_preserves_identity():
    """Only 1:1 links carry an id; that must not have been broken."""
    shape = (40, 40)
    masks = [disc(shape, (20, 8 + t), 6) for t in range(10)]
    tr = track(masks)
    assert tr.n_tracks == 1
    assert all(row.tolist() == [1] for row in tr.ids)


def test_lineage_is_acyclic_by_construction():
    """Ids are only ever minted fresh, so edges cannot point backwards."""
    nx = pytest.importorskip("networkx")
    from track.graph import to_networkx

    shape = (40, 60)
    whole = disc(shape, (20, 30), 11)
    parted = disc(shape, (20, 24), 5) | disc(shape, (20, 36), 5)
    tr = track([whole, parted, parted, whole, whole, parted], min_overlap=0.25)
    g = to_networkx(tr)
    assert g.number_of_edges() > 0
    assert nx.is_directed_acyclic_graph(g)
    for u, v in g.edges():
        assert u < v  # a child id is always newer than its parents


def test_volume_consistency_passes_a_clean_merge():
    from track import volume_consistency

    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    # area-conserving product: sqrt(8**2 + 5**2) keeps the two discs' total
    joined = disc(shape, (20, 30), np.hypot(8, 5))
    frames = [segment(m) for m in (before, joined)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    assert tr.of_kind(EventKind.MERGE)
    assert volume_consistency(tr, frames) == []


def test_volume_consistency_flags_a_merge_that_invents_vapor():
    from track import volume_consistency

    shape = (60, 60)
    before = disc(shape, (30, 24), 4) | disc(shape, (30, 36), 4)
    huge = disc(shape, (30, 30), 22)  # far more vapor than the parents held
    frames = [segment(m) for m in (before, huge)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    bad = volume_consistency(tr, frames)
    assert [v.check for v in bad] == ["sum_ratio"]
    assert bad[0].ratio > 1.5
    assert bad[0].kind is EventKind.MERGE


def test_volume_consistency_bounds_are_configurable():
    from track import volume_consistency

    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), np.hypot(8, 5))
    frames = [segment(m) for m in (before, joined)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    # an impossibly tight band must flag the very event the default accepts,
    # once the discretisation allowance is switched off
    assert volume_consistency(tr, frames, lower=0.999, upper=1.001, sigmas=0)


def test_attached_by_track_rejects_a_mismatched_sequence():
    from track.audit import attached_by_track

    shape = (30, 30)
    frames = [segment(disc(shape, (15, 15), 5)) for _ in range(3)]
    tr = link_by_voxel_overlap(frames)
    with pytest.raises(ValueError, match="same sequence that was linked"):
        attached_by_track(tr, frames[:2])


def test_volume_consistency_needs_frames_only_for_attachment():
    from track import volume_consistency

    tr, frames = merging_blocks(24, 1.25)
    # volumes come from the Tracks, so no frames are needed here
    assert volume_consistency(tr, detached_only=False)
    with pytest.raises(ValueError, match="detached_only needs frames"):
        volume_consistency(tr)


def test_volume_ratios_reports_the_continue_control():
    from track import volume_ratios

    shape = (40, 40)
    frames = [segment(disc(shape, (20, 8 + t), 6)) for t in range(5)]
    tr = link_by_voxel_overlap(frames)
    ratios = volume_ratios(tr)
    assert set(ratios) == {"continue"}
    # a rigid translating disc neither grows nor shrinks
    assert np.allclose(ratios["continue"], 1.0, atol=0.05)


def test_touches_wall_finds_regions_on_the_heater():
    from track import touches_wall

    m = np.zeros((20, 20), bool)
    m[0:3, 2:5] = True  # sitting on the Y=0 wall
    m[10:13, 10:13] = True  # detached
    f = segment(m)
    assert f.count == 2
    assert touches_wall(f).tolist() == [True, False]
    assert touches_wall(f, axis=0, side=-1).tolist() == [False, False]
    assert touches_wall(f, axis=1, side=0).tolist() == [False, False]


def test_wall_attached_merges_are_exempt_from_conservation():
    """A bubble on the heater is fed vapor, so it may gain volume freely."""
    from track import volume_consistency

    shape = (60, 60)
    # both parents sit on the wall, and the product balloons
    before = disc(shape, (2, 24), 5) | disc(shape, (2, 36), 5)
    grown = disc(shape, (2, 30), 20)
    frames = [segment(m) for m in (before, grown)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    assert tr.of_kind(EventKind.MERGE)

    assert volume_consistency(tr, frames) == []  # skipped: attached
    flagged = volume_consistency(tr, frames, detached_only=False)
    assert [v.check for v in flagged] == ["sum_ratio"]


def test_detached_merges_are_still_checked():
    from track import volume_consistency

    shape = (60, 60)
    before = disc(shape, (30, 24), 4) | disc(shape, (30, 36), 4)
    huge = disc(shape, (30, 30), 22)
    frames = [segment(m) for m in (before, huge)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    assert volume_consistency(tr, frames)  # nowhere near the wall


def test_wall_side_is_configurable():
    from track import volume_consistency

    shape = (60, 60)
    # the same scenario, mirrored onto the far face of axis 0
    before = disc(shape, (57, 24), 5) | disc(shape, (57, 36), 5)
    grown = disc(shape, (57, 30), 20)
    frames = [segment(m) for m in (before, grown)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    assert volume_consistency(tr, frames)  # wall assumed at Y=0, so checked
    assert volume_consistency(tr, frames, wall_side=-1) == []  # heater at Y=-1


def test_tiny_bubbles_are_below_the_measurable_floor():
    """A voxel or two on a tiny bubble is a huge ratio swing, not physics."""
    from track import volume_consistency

    shape = (40, 40)
    before = disc(shape, (20, 18), 1) | disc(shape, (20, 22), 1)  # 5 px each
    after = disc(shape, (20, 20), 3)
    frames = [segment(m) for m in (before, after)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    assert tr.of_kind(EventKind.MERGE)
    assert volume_consistency(tr, frames) == []  # skipped by min_volume
    assert volume_consistency(tr, frames, min_volume=0)  # flagged without it


def test_min_volume_does_not_hide_real_errors():
    from track import volume_consistency

    shape = (60, 60)
    before = disc(shape, (30, 24), 4) | disc(shape, (30, 36), 4)  # ~98 px
    huge = disc(shape, (30, 30), 22)
    frames = [segment(m) for m in (before, huge)]
    tr = link_by_voxel_overlap(frames, min_overlap=0.25)
    assert volume_consistency(tr, frames, min_volume=25)


def merging_blocks(side, gain, gap=2):
    """Two square bubbles that coalesce into one of ``gain`` times the area.

    Squares rather than discs so the areas are exact and the two parents stay
    disjoint whatever their size.
    """
    width = 2 * side + gap
    height = round(gain * 2 * side * side / width)
    shape = (height + side + 20, width + 20)
    before = np.zeros(shape, bool)
    before[5 : 5 + side, 5 : 5 + side] = True
    before[5 : 5 + side, 5 + side + gap : 5 + 2 * side + gap] = True
    after = np.zeros(shape, bool)
    after[5 : 5 + height, 5 : 5 + width] = True
    frames = [segment(before), segment(after)]
    assert frames[0].count == 2 and frames[1].count == 1
    return link_by_voxel_overlap(frames, min_overlap=0.25), frames


def test_tolerance_widens_for_small_bubbles():
    """Interface error is a surface effect, so slack scales with 1/sqrt(V)."""
    from track import volume_consistency

    # the same ~25% surplus: forgiven on small bubbles, flagged on large ones
    small_tr, small_f = merging_blocks(8, 1.25)
    large_tr, large_f = merging_blocks(24, 1.25)
    assert small_tr.of_kind(EventKind.MERGE) and large_tr.of_kind(EventKind.MERGE)
    assert volume_consistency(small_tr, small_f) == []
    assert volume_consistency(large_tr, large_f)


def test_sigmas_zero_gives_a_hard_band():
    from track import volume_consistency

    tr, frames = merging_blocks(8, 1.25)
    assert volume_consistency(tr, frames) == []
    assert volume_consistency(tr, frames, sigmas=0)


def test_trajectory_records_centroid_and_volume():
    shape = (30, 40)
    masks = [np.zeros(shape, bool) for _ in range(5)]
    for t in range(5):
        masks[t][10:16, 5 + 2 * t : 11 + 2 * t] = True  # 6x6 block, 2 px/frame
    tr = track(masks)
    tj = tr.trajectory(1)

    assert len(tj) == 5
    assert tj.frames.tolist() == [0, 1, 2, 3, 4]
    assert tj.volume.tolist() == [36] * 5
    assert np.allclose(tj.velocity, [[0, 2]] * 4)
    assert np.allclose(tj.growth, 0.0)


def test_trajectory_follows_a_growing_bubble():
    shape = (60, 60)
    radii = [6, 7, 8, 9]
    tr = track([disc(shape, (30, 30), r) for r in radii])
    tj = tr.trajectory(1)
    assert np.all(np.diff(tj.volume) > 0)
    assert np.all(tj.growth > 0)
    # area of a disc, so volume should track r**2
    assert np.allclose(tj.volume / np.array(radii) ** 2, np.pi, atol=0.4)


def test_trajectory_skips_frames_where_the_track_is_absent():
    shape = (30, 30)
    blob = disc(shape, (15, 15), 5)
    tr = track([np.zeros(shape, bool), blob, blob, np.zeros(shape, bool)])
    tj = tr.trajectory(1)
    assert tj.frames.tolist() == [1, 2]
    assert len(tj) == 2


def test_trajectories_matches_trajectory_for_every_track():
    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), 14)
    tr = track([before, before, joined], min_overlap=0.25)
    everything = tr.trajectories()
    assert set(everything) == set(tr.lifetimes())
    for tid, tj in everything.items():
        one = tr.trajectory(tid)
        assert tj.frames.tolist() == one.frames.tolist()
        assert tj.volume.tolist() == one.volume.tolist()
        assert np.allclose(tj.centroid, one.centroid)


def test_trajectory_rejects_an_unknown_track():
    shape = (30, 30)
    tr = track([disc(shape, (15, 15), 5)] * 2)
    with pytest.raises(KeyError, match="no track 999"):
        tr.trajectory(999)


def test_trajectory_is_consistent_with_lifetimes():
    shape = (40, 40)
    masks = [disc(shape, (20, 8 + t), 6) for t in range(7)]
    tr = track(masks)
    for tid, (first, last) in tr.lifetimes().items():
        tj = tr.trajectory(tid)
        assert tj.frames[0] == first and tj.frames[-1] == last
