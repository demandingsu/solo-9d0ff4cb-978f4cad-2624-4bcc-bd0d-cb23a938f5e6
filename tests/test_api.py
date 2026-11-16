"""端到端测试：版本发布、跨批补传、幂等、冲突拒绝、并发与发布失败。"""
import csv
import io
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import Settings, create_app


def make_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["photo_no", "camera_no", "captured_at", "species", "rel_path"])
    w.writerows(rows)
    return buf.getvalue().encode()


BATCH1 = make_csv([
    ["P0002", "CAM01", "2026-03-01T08:00:00Z", "wild_boar", "cam01/P0002.jpg"],
    ["P0003", "CAM01", "2026-03-01T08:00:30Z", "wild_boar", "cam01/P0003.jpg"],
    ["P0005", "CAM01", "2026-03-01T08:01:40Z", "roe_deer", "cam01/P0005.jpg"],
    ["P0006", "CAM01", "2026-03-01T08:02:10Z", "roe_deer", "cam01/P0006.jpg"],
    ["P0001", "CAM02", "2026-03-01T08:00:10Z", "red_fox", "cam02/P0001.jpg"],
    ["P0004", "CAM02", "2026-03-01T09:30:00Z", "red_fox", "cam02/P0004.jpg"],
])

BACKFILL = make_csv([
    ["P0007", "CAM01", "2026-03-01T08:01:05Z", "wild_boar", "cam01/P0007.jpg"],  # 桥接 CAM01 两旧组
    ["P0008", "CAM02", "2026-03-01T09:30:40Z", "red_fox", "cam02/P0008.jpg"],     # 加入 CAM02 旧组
    ["P0009", "CAM03", "2026-03-01T12:00:00Z", "sika_deer", "cam03/P0009.jpg"],   # 新相机新组
])


@pytest.fixture()
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", val_percent=20, gap_seconds=60)
    app = create_app(settings)
    client = TestClient(app, follow_redirects=True)
    return client, settings


def upload(client, data, name="batch.csv"):
    return client.post("/upload", files={"file": (name, data, "text/csv")})


def fetch_index(client, vid, name):
    r = client.get(f"/versions/{vid}/files/{name}")
    assert r.status_code == 200, r.text
    return list(csv.DictReader(io.StringIO(r.text)))


def db_count(settings, table):
    conn = sqlite3.connect(settings.db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_first_upload_publishes_v1(env):
    client, settings = env
    r = upload(client, BATCH1, "b1.csv")
    assert r.status_code == 200
    assert "已发布版本 v1" in r.text

    train = fetch_index(client, 1, "train_index.csv")
    val = fetch_index(client, 1, "val_index.csv")
    assert len(train) + len(val) == 6
    train_nos = {r["photo_no"] for r in train}
    val_nos = {r["photo_no"] for r in val}
    assert not (train_nos & val_nos)  # 无交集
    # 同一组不跨两侧（无泄漏）
    train_groups = {r["group_id"] for r in train}
    val_groups = {r["group_id"] for r in val}
    assert not (train_groups & val_groups)
    # 间隔 70 秒 -> CAM01 两组；共 4 组
    assert len(train_groups | val_groups) == 4


def test_backfill_merges_groups_and_old_links_keep_content(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    v1_train = client.get("/versions/1/files/train_index.csv").text
    v1_val = client.get("/versions/1/files/val_index.csv").text

    r = upload(client, BACKFILL, "b2.csv")
    assert "已发布版本 v2" in r.text

    # v2：CAM01 两组合并为一组（group_id 沿用更早的 P0002），共 4 组
    train = fetch_index(client, 2, "train_index.csv")
    val = fetch_index(client, 2, "val_index.csv")
    all_rows = train + val
    assert len(all_rows) == 9
    cam01_groups = {r["group_id"] for r in all_rows if r["camera_no"] == "CAM01"}
    assert cam01_groups == {"P0002"}
    cam01_members = {r["photo_no"] for r in all_rows if r["camera_no"] == "CAM01"}
    assert cam01_members == {"P0002", "P0003", "P0005", "P0006", "P0007"}
    # P0008 加入 CAM02 旧组（group_id 仍为 P0004）
    g_of = {r["photo_no"]: r["group_id"] for r in all_rows}
    assert g_of["P0008"] == g_of["P0004"] == "P0004"

    # 旧版本链接仍下载原内容
    assert client.get("/versions/1/files/train_index.csv").text == v1_train
    assert client.get("/versions/1/files/val_index.csv").text == v1_val
    # 首页现役版本为 v2
    assert "v2" in client.get("/").text


def test_duplicate_batch_creates_no_version(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    r = upload(client, BATCH1, "b1-again.csv")
    assert "未产生新版本" in r.text
    assert db_count(settings, "versions") == 1
    assert db_count(settings, "photos") == 6
    # 批次被记录为 noop
    conn = sqlite3.connect(settings.db_path)
    statuses = [row[0] for row in conn.execute("SELECT status FROM batches ORDER BY id")]
    conn.close()
    assert statuses == ["applied", "noop"]


def test_conflicting_photo_no_rejects_whole_batch(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    conflict = make_csv([
        ["P0001", "CAM02", "2026-03-01T08:00:10Z", "gray_wolf", "cam02/P0001.jpg"],  # 物种被改
        ["P0011", "CAM03", "2026-03-01T12:05:00Z", "sika_deer", "cam03/P0011.jpg"],  # 全新照片
    ])
    r = upload(client, conflict, "bad.csv")
    assert "整批拒绝" in r.text
    assert db_count(settings, "versions") == 1
    assert db_count(settings, "photos") == 6  # P0011 也未入库
    assert "v1" in client.get("/").text


def test_invalid_csv_rejected(env):
    client, settings = env
    bad = b"photo_no,camera_no,captured_at,species,rel_path\nP1,C1,not-a-time,fox,a.jpg\n"
    r = upload(client, bad, "bad.csv")
    assert "拒绝" in r.text
    assert db_count(settings, "versions") == 0
    assert db_count(settings, "photos") == 0


def test_concurrent_uploads_serialize_and_stay_consistent(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    batch_a = make_csv([["P0101", "CAM09", "2026-03-02T10:00:00Z", "boar", "c9/P0101.jpg"]])
    batch_b = make_csv([["P0201", "CAM10", "2026-03-02T11:00:00Z", "deer", "c10/P0201.jpg"]])

    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(upload, client, batch_a, "a.csv")
        fb = pool.submit(upload, client, batch_b, "b.csv")
        ra, rb = fa.result(), fb.result()
    assert ra.status_code == 200 and rb.status_code == 200

    assert db_count(settings, "versions") == 3  # 两个并发批次各产一个版本
    assert db_count(settings, "photos") == 8
    # 现役版本（v3）的两份下载合起来正好是完整照片集，且无交集
    train = fetch_index(client, 3, "train_index.csv")
    val = fetch_index(client, 3, "val_index.csv")
    nos = [r["photo_no"] for r in train] + [r["photo_no"] for r in val]
    assert sorted(nos) == sorted(["P0001", "P0002", "P0003", "P0004", "P0005", "P0006", "P0101", "P0201"])
    # 首页展示的现役版本与下载一致
    assert "v3" in client.get("/").text


def test_publish_failure_keeps_active_version(env, monkeypatch):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    v1_train = client.get("/versions/1/files/train_index.csv").text

    import app.main as main

    def boom(*args, **kwargs):
        raise RuntimeError("磁盘写入失败（模拟）")

    monkeypatch.setattr(main, "write_version_files", boom)
    r = upload(client, BACKFILL, "b2.csv")
    assert "发布失败" in r.text
    assert "现役版本未变" in r.text

    # 现役版本不变、数据回滚、下载仍正常
    assert db_count(settings, "versions") == 1
    assert db_count(settings, "photos") == 6
    assert client.get("/versions/1/files/train_index.csv").text == v1_train
    home = client.get("/").text
    assert "v1" in home and "v2" not in home

    # 故障恢复后重投同一批可正常发布 v2
    monkeypatch.undo()
    r2 = upload(client, BACKFILL, "b2.csv")
    assert "已发布版本 v2" in r2.text
    assert db_count(settings, "photos") == 9


def test_version_detail_shows_groups(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    r = client.get("/versions/1")
    assert r.status_code == 200
    assert "P0002" in r.text and "出没事件分组" in r.text


# ---------------------------------------------------------------- 小数秒精度链路

FRAC_BATCH = make_csv([
    ["P1001", "CAM01", "2026-03-01T08:00:00Z", "wild_boar", "cam01/P1001.jpg"],
    ["P1002", "CAM01", "2026-03-01T08:01:00.800Z", "wild_boar", "cam01/P1002.jpg"],  # 距 P1001 60.8 秒 -> 必须分组
    ["P1003", "CAM01", "2026-03-01T08:01:30.500Z", "wild_boar", "cam01/P1003.jpg"],  # 距 P1002 29.7 秒 -> 同组
])


def fetch_all_rows(client, vid):
    return fetch_index(client, vid, "train_index.csv") + fetch_index(client, vid, "val_index.csv")


def test_fractional_boundary_grouping_and_storage(env):
    """60.8 秒间隔必须分组（修复前被截断为 60 秒而错误并组）；索引保留小数秒。"""
    client, settings = env
    r = upload(client, FRAC_BATCH, "frac.csv")
    assert "已发布版本 v1" in r.text
    rows = fetch_all_rows(client, 1)
    assert len(rows) == 3
    g = {r["photo_no"]: r["group_id"] for r in rows}
    assert g["P1001"] != g["P1002"]      # 60.8s > 60s：两组
    assert g["P1002"] == g["P1003"]      # 29.7s：同组
    at = {r["photo_no"]: r["captured_at"] for r in rows}
    assert at["P1001"] == "2026-03-01T08:00:00Z"        # 整秒保持旧格式
    assert at["P1002"] == "2026-03-01T08:01:00.800Z"    # 小数秒原样保留
    assert at["P1003"] == "2026-03-01T08:01:30.500Z"


def test_subsecond_change_same_photo_no_rejects_batch(env):
    """同号照片只改小数秒 -> 内容不同 -> 整批拒绝（修复前误判为原件重投）。"""
    client, settings = env
    upload(client, FRAC_BATCH, "frac.csv")
    changed = make_csv([
        ["P1002", "CAM01", "2026-03-01T08:01:00.900Z", "wild_boar", "cam01/P1002.jpg"],  # 仅小数秒被改
        ["P1004", "CAM01", "2026-03-01T08:02:00Z", "wild_boar", "cam01/P1004.jpg"],      # 全新照片也不许入库
    ])
    r = upload(client, changed, "changed.csv")
    assert "整批拒绝" in r.text
    assert db_count(settings, "photos") == 3
    assert db_count(settings, "versions") == 1
    assert "v1" in client.get("/").text


def test_equivalent_timezone_treated_as_same_instant(env):
    """等价时区写法（+08:00 与 Z）视为同一时刻 -> 判为重投，不产生新版本。"""
    client, settings = env
    upload(client, FRAC_BATCH, "frac.csv")
    same = make_csv([
        ["P1002", "CAM01", "2026-03-01T16:01:00.800+08:00", "wild_boar", "cam01/P1002.jpg"],
        ["P1003", "CAM01", "2026-03-01T08:01:30.500000+00:00", "wild_boar", "cam01/P1003.jpg"],
    ])
    r = upload(client, same, "tz.csv")
    assert "未产生新版本" in r.text
    assert db_count(settings, "versions") == 1
    assert db_count(settings, "photos") == 3


def test_fractional_reupload_is_noop(env):
    """带小数秒的批次原样重投 -> noop，不产生新版本。"""
    client, settings = env
    upload(client, FRAC_BATCH, "frac.csv")
    r = upload(client, FRAC_BATCH, "frac-again.csv")
    assert "未产生新版本" in r.text
    assert db_count(settings, "versions") == 1
    assert db_count(settings, "photos") == 3


def test_mixed_whole_second_and_fractional_across_batches(env):
    """旧库整秒照片与新增小数秒照片混用：跨批归组按真实时刻计算。"""
    client, settings = env
    upload(client, BATCH1, "b1.csv")  # 整秒旧数据
    backfill = make_csv([
        ["P0007", "CAM01", "2026-03-01T08:01:05.500Z", "wild_boar", "cam01/P0007.jpg"],  # 桥接 CAM01 两旧组
        ["P0010", "CAM03", "2026-03-01T12:00:00Z", "sika_deer", "cam03/P0010.jpg"],
    ])
    r = upload(client, backfill, "b2.csv")
    assert "已发布版本 v2" in r.text
    later = make_csv([
        ["P0011", "CAM03", "2026-03-01T12:01:00.800Z", "sika_deer", "cam03/P0011.jpg"],  # 距 P0010 60.8 秒 -> 新组
    ])
    r = upload(client, later, "b3.csv")
    assert "已发布版本 v3" in r.text

    rows = fetch_all_rows(client, 3)
    g = {r["photo_no"]: r["group_id"] for r in rows}
    # 小数秒桥接照片把 CAM01 旧两组合并，沿用最早组标识
    assert g["P0002"] == g["P0005"] == g["P0007"] == "P0002"
    # 60.8 秒间隔必须分组
    assert g["P0011"] != g["P0010"]


def test_index_sorted_by_true_instant_with_mixed_precision(env):
    """整秒与小数秒混排时索引按真实时刻排序（字符串序会把 .500Z 排到 00Z 之前）。"""
    client, settings = env
    upload(client, make_csv([
        ["P2001", "CAM01", "2026-03-01T08:00:00Z", "boar", "c1/P2001.jpg"],
        ["P2003", "CAM01", "2026-03-01T08:00:30Z", "boar", "c1/P2003.jpg"],
    ]), "b1.csv")
    upload(client, make_csv([
        ["P2002", "CAM01", "2026-03-01T08:00:00.500Z", "boar", "c1/P2002.jpg"],
    ]), "b2.csv")
    rows = fetch_all_rows(client, 2)
    cam01 = [r["photo_no"] for r in rows if r["camera_no"] == "CAM01"]
    assert cam01 == ["P2001", "P2002", "P2003"]
