import numpy as np
import pytest

from track import Criterion, EventKind, intersection, link, segment, track


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


def test_merge_is_detected_and_id_follows_larger():
    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    after = disc(shape, (20, 22), 8) | disc(shape, (20, 36), 5)
    joined = disc(shape, (20, 26), 9) | disc(shape, (20, 33), 7)
    tr = track([before, after, joined])
    merges = tr.of_kind(EventKind.MERGE)
    assert len(merges) == 1
    assert len(merges[0].parents) == 2
    assert len(merges[0].children) == 1
    # the child keeps the bigger parent's identity
    assert merges[0].children[0] == 1


def test_split_is_detected():
    shape = (40, 60)
    whole = disc(shape, (20, 30), 11)
    # a pinch-off: both fragments still lie inside the parent's footprint
    parted = disc(shape, (20, 24), 5) | disc(shape, (20, 36), 5)
    tr = track([whole, parted])
    splits = tr.of_kind(EventKind.SPLIT)
    assert len(splits) == 1
    assert len(splits[0].children) == 2
    assert tr.n_tracks == 2  # one child inherits, one is new


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


def test_lineage_graph_records_absorption_not_survival():
    nx = pytest.importorskip("networkx")
    from track.graph import families, to_networkx

    shape = (40, 60)
    before = disc(shape, (20, 20), 8) | disc(shape, (20, 40), 5)
    joined = disc(shape, (20, 30), 14)
    g = to_networkx(track([before, joined], min_overlap=0.25))

    # the surviving track keeps its id, so only the absorbed one gets an edge
    assert g.number_of_edges() == 1
    (u, v), = g.edges()
    assert g[u][v]["kind"] == EventKind.MERGE
    assert u != v  # never a self-loop on the heir
    assert nx.is_directed_acyclic_graph(g)
    assert len(families(g)) == 1  # the two tracks form one family


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
    assert [d["kind"] for *_, d in g.edges(data=True)] == [EventKind.SPLIT]


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
    assert [d["kind"] for *_, d in back.edges(data=True)] == ["merge"]
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
