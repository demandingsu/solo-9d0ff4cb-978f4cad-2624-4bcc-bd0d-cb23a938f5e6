"""归组与稳定划分的纯逻辑测试。"""
from datetime import datetime, timedelta, timezone

from app.grouping import Photo, assign_split, compute_groups

T0 = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)


def p(no, cam, sec, species="boar"):
    return Photo(no, cam, T0 + timedelta(seconds=sec), species, f"{cam}/{no}.jpg")


def test_transitive_chaining_merges_via_bridge():
    # A..B 间隔 50s，B..C 间隔 50s；A..C 相隔 100s 仍同组（传递归组）
    photos = [p("P1", "C1", 0), p("P2", "C1", 50), p("P3", "C1", 100)]
    groups = compute_groups(photos)
    assert len(groups) == 1
    assert [x.photo_no for x in groups[0].photos] == ["P1", "P2", "P3"]


def test_gap_over_60s_splits():
    photos = [p("P1", "C1", 0), p("P2", "C1", 61)]
    assert len(compute_groups(photos)) == 2


def test_exactly_60s_stays_together():
    photos = [p("P1", "C1", 0), p("P2", "C1", 60)]
    assert len(compute_groups(photos)) == 1


def test_fractional_gap_over_60s_splits():
    # 60.8 秒间隔必须分组（回归：整秒截断曾把它当成 60 秒而错误并组）
    photos = [p("P1", "C1", 0), p("P2", "C1", 60.8)]
    assert len(compute_groups(photos)) == 2


def test_fractional_gap_exactly_60s_stays_together():
    # 带小数秒恰好 60.0 秒：仍在阈值内，同组
    photos = [p("P1", "C1", 0.5), p("P2", "C1", 60.5)]
    assert len(compute_groups(photos)) == 1


def test_fractional_gap_just_under_60s_stays_together():
    # 59.9 秒：同组
    photos = [p("P1", "C1", 0.2), p("P2", "C1", 60.1)]
    assert len(compute_groups(photos)) == 1


def test_mixed_whole_and_fractional_seconds_grouping():
    # 旧库整秒照片与新增小数秒照片混用：按真实时刻归组
    # 0 -> 30.5（间隔 30.5）同组；30.5 -> 60.8（间隔 30.3）同组，链式合并为一组
    photos = [p("P1", "C1", 0), p("P2", "C1", 30.5), p("P3", "C1", 60.8)]
    groups = compute_groups(photos)
    assert len(groups) == 1
    assert [x.photo_no for x in groups[0].photos] == ["P1", "P2", "P3"]
    # 整秒照片与 60.8 秒后的小数秒照片：必须分组
    assert len(compute_groups([p("P1", "C1", 0), p("P3", "C1", 60.8)])) == 2


def test_subsecond_ordering_within_group():
    # 同一秒内的小数时刻按真实先后排序，而非照片号
    photos = [p("P9", "C1", 0.5), p("P1", "C1", 0.2)]
    (g,) = compute_groups(photos)
    assert [x.photo_no for x in g.photos] == ["P1", "P9"]
    assert g.group_id == "P1"


def test_cameras_are_independent():
    photos = [p("P1", "C1", 0), p("P2", "C2", 10)]
    groups = compute_groups(photos)
    assert len(groups) == 2
    assert {g.camera_no for g in groups} == {"C1", "C2"}


def test_group_id_is_earliest_photo():
    photos = [p("P9", "C1", 30), p("P3", "C1", 0)]
    (g,) = compute_groups(photos)
    assert g.group_id == "P3"


def test_backfill_bridge_merges_and_keeps_earlier_group_id():
    base = [p("P1", "C1", 0), p("P2", "C1", 30), p("P3", "C1", 100), p("P4", "C1", 130)]
    assert len(compute_groups(base)) == 2  # 30s -> 100s 间隔 70s，分为两组
    merged = compute_groups(base + [p("P5", "C1", 65)])  # 桥接：30->65=35s, 65->100=35s
    assert len(merged) == 1
    assert merged[0].group_id == "P1"  # 沿用更早组的标识


def test_split_is_deterministic_and_group_consistent():
    photos = [p(f"P{i}", "C1", i * 300) for i in range(1, 21)]  # 20 个独立组
    groups1 = compute_groups(photos)
    groups2 = compute_groups(list(reversed(photos)))
    s1 = [assign_split(g.group_id, 20, "salt") for g in groups1]
    s2 = [assign_split(g.group_id, 20, "salt") for g in groups2]
    assert s1 == s2  # 与输入顺序无关
    assert "val" in s1 and "train" in s1  # 20 个组应两侧都有
