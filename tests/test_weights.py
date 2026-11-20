"""Tests for paired replicate-weight generation (POST /api/weight-jobs)."""
from __future__ import annotations

import csv as csv_mod
import hashlib
import io
import zipfile
from pathlib import Path

import duckdb
import pytest

from app import weights


# ---------------------------------------------------------------- fixtures
def write_sample_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.writer(f)
        w.writerow(["sample_id", "district", "bus", "main_weight"])
        w.writerows(rows)


def write_crosswalk_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.writer(f)
        w.writerow(["round", "sample_id", "area_id"])
        w.writerows(rows)


def make_simple_inputs(tmp: Path):
    """Two rounds: district A areas a1/a2 (2 samples each), district B area b1."""
    r1 = tmp / "r1.csv"
    write_sample_csv(r1, [
        ("s1", "A", 1, 4.0), ("s2", "A", 0, 6.0),
        ("s3", "A", 1, 3.0), ("s4", "A", 0, 5.0),
        ("s5", "B", 1, 2.0), ("s6", "B", 1, 8.0),
    ])
    r2 = tmp / "r2.csv"
    write_sample_csv(r2, [
        ("t1", "A", 1, 9.0), ("t2", "A", 1, 1.0),
        ("t3", "A", 0, 2.0), ("t4", "A", 1, 7.0),
        ("t5", "B", 0, 5.0), ("t6", "B", 1, 5.0),
    ])
    xw = tmp / "xw.csv"
    write_crosswalk_csv(xw, [
        (1, "s1", "a1"), (1, "s2", "a1"), (1, "s3", "a2"), (1, "s4", "a2"),
        (1, "s5", "b1"), (1, "s6", "b1"),
        (2, "t1", "a1"), (2, "t2", "a1"), (2, "t3", "a2"), (2, "t4", "a2"),
        (2, "t5", "b1"), (2, "t6", "b1"),
    ])
    return r1, r2, xw


def post_job(client, r1: Path, r2: Path, xw: Path, reps: int = 100, seed: int = 42):
    with open(r1, "rb") as f1, open(r2, "rb") as f2, open(xw, "rb") as fx:
        return client.post(
            "/api/weight-jobs",
            files={
                "round1_sample_csv": (r1.name, f1.read(), "text/csv"),
                "round2_sample_csv": (f2.name, f2.read(), "text/csv"),
                "crosswalk_csv": (fx.name, fx.read(), "text/csv"),
            },
            data={"n_replicates": str(reps), "seed": str(seed)},
        )


def read_zip(client, resp) -> zipfile.ZipFile:
    assert resp.status_code == 200, resp.text
    download = client.get(resp.json()["download_url"])
    assert download.status_code == 200, download.text
    return zipfile.ZipFile(io.BytesIO(download.content))


def parquet_to_rows(data: bytes) -> list[tuple]:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tf:
        tf.write(data)
        tf.flush()
        con = duckdb.connect()
        try:
            rows = con.execute(
                "SELECT sample_id, rep_id, weight FROM read_parquet(?)"
                " ORDER BY sample_id, rep_id",
                [tf.name],
            ).fetchall()
        finally:
            con.close()
    return rows


def expected_hits(district: str, area_ids: list[str], n_reps: int, seed: int) -> dict:
    out = {}
    for k in range(n_reps):
        rng = weights._replicate_rng(seed, district, k)
        counts = [0] * len(area_ids)
        for _ in range(len(area_ids)):
            counts[rng.randrange(len(area_ids))] += 1
        out[weights._rep_id(k)] = dict(zip(area_ids, counts))
    return out


# ---------------------------------------------------------------- happy path
def test_generation_happy_path_zip_structure(app_env, tmp_path):
    client, data_dir = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    resp = post_job(client, r1, r2, xw)
    zf = read_zip(client, resp)
    assert set(zf.namelist()) == {
        "round1/samples.csv", "round1/replicates.parquet",
        "round2/samples.csv", "round2/replicates.parquet",
        "manifest.json",
    }
    payload = resp.json()
    assert payload["n_replicates"] == 100
    assert payload["round1"] == {"n_samples": 6, "n_rep_rows": 600}
    assert payload["round2"] == {"n_samples": 6, "n_rep_rows": 600}
    assert payload["districts"] == ["A", "B"]
    assert payload["download_url"].endswith("/download")
    # Original sample CSV bytes are preserved inside the ZIP.
    assert zf.read("round1/samples.csv") == r1.read_bytes()
    assert zf.read("round2/samples.csv") == r2.read_bytes()


def test_generated_weights_match_main_weight_times_hits(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    reps = 50
    resp = post_job(client, r1, r2, xw, reps=reps, seed=7)
    zf = read_zip(client, resp)
    rows = parquet_to_rows(zf.read("round1/replicates.parquet"))
    assert len(rows) == 6 * reps

    main_w = {"s1": 4.0, "s2": 6.0, "s3": 3.0, "s4": 5.0, "s5": 2.0, "s6": 8.0}
    sample_area = {"s1": "a1", "s2": "a1", "s3": "a2", "s4": "a2",
                   "s5": "b1", "s6": "b1"}
    hits_a = expected_hits("A", ["a1", "a2"], reps, 7)
    hits_b = expected_hits("B", ["b1"], reps, 7)
    for sample_id, rep_id, weight in rows:
        area = sample_area[sample_id]
        expected_h = (hits_a if area in ("a1", "a2") else hits_b)[rep_id][area]
        assert weight == pytest.approx(main_w[sample_id] * expected_h)


def test_zero_weights_are_explicit_rows(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    resp = post_job(client, r1, r2, xw, reps=200, seed=99)
    zf = read_zip(client, resp)
    rows = parquet_to_rows(zf.read("round1/replicates.parquet"))
    # Full coverage: exactly one row per sample per replicate.
    pairs = {(s, k) for s, k, _ in rows}
    assert len(pairs) == 6 * 200
    zero_rows = [(s, k) for s, k, w in rows if w == 0.0]
    assert zero_rows, "with replacement some areas must go undrawn sometimes"
    # Each zero must coincide with an area whose hit count is zero.
    hits_a = expected_hits("A", ["a1", "a2"], 200, 99)
    sample_area = {"s1": "a1", "s2": "a1", "s3": "a2", "s4": "a2"}
    for sample_id, rep_id in zero_rows:
        if sample_id in sample_area:
            assert hits_a[rep_id][sample_area[sample_id]] == 0


def test_paired_rounds_share_area_hit_counts(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    reps = 80
    resp = post_job(client, r1, r2, xw, reps=reps, seed=123)
    zf = read_zip(client, resp)
    d1 = parquet_to_rows(zf.read("round1/replicates.parquet"))
    d2 = parquet_to_rows(zf.read("round2/replicates.parquet"))
    # In single-area district B, every draw hits b1 once -> every row is the
    # unchanged main weight; the pairing guarantee is checked structurally:
    # round-2 t1..t4 zero/nonzero pattern must match the shared area draws
    # implied by round 1's weights.
    w1 = {(s, k): w for s, k, w in d1}
    w2 = {(s, k): w for s, k, w in d2}
    main1 = {"s1": 4.0, "s2": 6.0, "s3": 3.0, "s4": 5.0}
    main2 = {"t1": 9.0, "t2": 1.0, "t3": 2.0, "t4": 7.0}
    area1 = {"s1": "a1", "s2": "a1", "s3": "a2", "s4": "a2"}
    area2 = {"t1": "a1", "t2": "a1", "t3": "a2", "t4": "a2"}
    for k in {rep for _, rep, _ in d1}:
        for area in ("a1", "a2"):
            r1_zero = all(w1[(s, k)] == 0.0 for s, a in area1.items() if a == area)
            r2_zero = all(w2[(t, k)] == 0.0 for t, a in area2.items() if a == area)
            assert r1_zero == r2_zero
            if not r1_zero:
                hit1 = {s: w1[(s, k)] / main1[s] for s, a in area1.items() if a == area}
                hit2 = {t: w2[(t, k)] / main2[t] for t, a in area2.items() if a == area}
                assert set(hit1.values()) == set(hit2.values()) == {
                    list(hit1.values())[0]
                }
    # District B: n_areas = 1 so hits are always 1.
    for k in {rep for _, rep, _ in d1}:
        assert w1[("s5", k)] == 2.0 and w1[("s6", k)] == 8.0
        assert w2[("t5", k)] == 5.0 and w2[("t6", k)] == 5.0


def test_draw_counts_sum_to_number_of_areas(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    reps = 30
    resp = post_job(client, r1, r2, xw, reps=reps, seed=5)
    zf = read_zip(client, resp)
    hits_a = expected_hits("A", ["a1", "a2"], reps, 5)
    hits_b = expected_hits("B", ["b1"], reps, 5)
    for k, by_area in hits_a.items():
        assert sum(by_area.values()) == 2
    for k, by_area in hits_b.items():
        assert by_area["b1"] == 1
    # Every replicate must have positive total weight in every round so the
    # generated Parquet can pass the existing ingestion denominator checks.
    rows1 = parquet_to_rows(zf.read("round1/replicates.parquet"))
    totals: dict[str, float] = {}
    for _, rep_id, weight in rows1:
        totals[rep_id] = totals.get(rep_id, 0.0) + weight
    assert all(t > 0 for t in totals.values())


# ---------------------------------------------------------------- determinism
def test_same_seed_and_reps_are_byte_identical(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    a = read_zip(client, post_job(client, r1, r2, xw, reps=40, seed=2026)).read(
        "round1/replicates.parquet"
    )
    b = read_zip(client, post_job(client, r1, r2, xw, reps=40, seed=2026)).read(
        "round1/replicates.parquet"
    )
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_row_reorder_does_not_change_weights(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    base = read_zip(client, post_job(client, r1, r2, xw, reps=40, seed=77))

    r1b = tmp_path / "r1_shuf.csv"
    r2b = tmp_path / "r2_shuf.csv"
    xwb = tmp_path / "xw_shuf.csv"
    for src, dst in ((r1, r1b), (r2, r2b), (xw, xwb)):
        lines = src.read_text(encoding="utf-8").splitlines()
        header, body = lines[0], lines[1:]
        body.reverse()
        dst.write_text(header + "\n" + "\n".join(body) + "\n", encoding="utf-8")

    shuf = read_zip(client, post_job(client, r1b, r2b, xwb, reps=40, seed=77))
    for name in ("round1/replicates.parquet", "round2/replicates.parquet"):
        assert parquet_to_rows(base.read(name)) == parquet_to_rows(shuf.read(name))


def test_extending_reps_keeps_existing_numbering(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    small = read_zip(client, post_job(client, r1, r2, xw, reps=25, seed=888))
    big = read_zip(client, post_job(client, r1, r2, xw, reps=60, seed=888))
    small_rows = parquet_to_rows(small.read("round1/replicates.parquet"))
    big_rows = parquet_to_rows(big.read("round1/replicates.parquet"))
    kept = {(s, k): w for s, k, w in big_rows if int(k[3:]) <= 25}
    assert kept == {(s, k): w for s, k, w in small_rows}
    assert len(big_rows) == 6 * 60


def test_different_seeds_usually_differ(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    a = parquet_to_rows(
        read_zip(client, post_job(client, r1, r2, xw, reps=50, seed=1)).read(
            "round1/replicates.parquet"
        )
    )
    b = parquet_to_rows(
        read_zip(client, post_job(client, r1, r2, xw, reps=50, seed=2)).read(
            "round1/replicates.parquet"
        )
    )
    assert a != b


# ---------------------------------------------------------------- round trip
def test_generated_zip_ingests_through_original_entry(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    zf = read_zip(client, post_job(client, r1, r2, xw, reps=30, seed=11))
    out = tmp_path / "unzipped"
    zf.extractall(out)
    with open(out / "round1" / "samples.csv", "rb") as fc, \
         open(out / "round1" / "replicates.parquet", "rb") as fp:
        resp = client.post(
            "/api/rounds/1",
            files={
                "sample_csv": ("samples.csv", fc.read(), "text/csv"),
                "replicate_parquet": ("replicates.parquet", fp.read(),
                                      "application/octet-stream"),
            },
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_rep_rows"] == 6 * 30
    # And the analysis endpoint still works on the generated weights.
    d = client.get("/api/result", params={"district": "A"}).json()
    # Round 2 missing -> unavailable with a Chinese reason, but the call works.
    assert d["available"] is False and "第二轮" in d["reason"]


# ---------------------------------------------------------------- validation
def _expect_422(resp, fragment):
    assert resp.status_code == 422, resp.text
    assert fragment in resp.json()["detail"], resp.json()["detail"]


def test_bad_extensions_rejected(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    with open(r1, "rb") as f1, open(r2, "rb") as f2, open(xw, "rb") as fx:
        resp = client.post(
            "/api/weight-jobs",
            files={
                "round1_sample_csv": ("x.txt", f1.read(), "text/plain"),
                "round2_sample_csv": (r2.name, f2.read(), "text/csv"),
                "crosswalk_csv": (fx.name, fx.read(), "text/csv"),
            },
            data={"n_replicates": "10", "seed": "1"},
        )
    _expect_422(resp, ".csv")


def test_bad_n_replicates_and_seed(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    for reps, seed, frag in [
        ("0", "1", "正整数"),
        ("-3", "1", "正整数"),
        ("abc", "1", "整数"),
        ("10", "", "随机种子"),
        ("10", "1.5", "整数"),
        ("1000000", "1", "不能超过"),
    ]:
        with open(r1, "rb") as f1, open(r2, "rb") as f2, open(xw, "rb") as fx:
            resp = client.post(
                "/api/weight-jobs",
                files={
                    "round1_sample_csv": (r1.name, f1.read(), "text/csv"),
                    "round2_sample_csv": (r2.name, f2.read(), "text/csv"),
                    "crosswalk_csv": (fx.name, fx.read(), "text/csv"),
                },
                data={"n_replicates": reps, "seed": seed},
            )
        _expect_422(resp, frag)


def test_bad_sample_columns_rejected(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad = tmp_path / "badcols.csv"
    bad.write_text(
        "sample_id,district,bus,main_weight,extra\ns1,A,1,4,9\n", encoding="utf-8"
    )
    _expect_422(post_job(client, bad, r2, xw), "列结构")


def test_bad_bus_and_weight_rejected(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad_bus = tmp_path / "badbus.csv"
    rows = list(csv_mod.reader(open(r1, encoding="utf-8")))
    rows[1][2] = "2"
    csv_mod.writer(open(bad_bus, "w", newline="", encoding="utf-8")).writerows(rows)
    _expect_422(post_job(client, bad_bus, r2, xw), "0 或 1")

    bad_w = tmp_path / "badw.csv"
    rows = list(csv_mod.reader(open(r1, encoding="utf-8")))
    rows[1][3] = "-1"
    csv_mod.writer(open(bad_w, "w", newline="", encoding="utf-8")).writerows(rows)
    _expect_422(post_job(client, bad_w, r2, xw), "非负")


def test_duplicate_sample_id_rejected(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad = tmp_path / "dupid.csv"
    write_sample_csv(bad, [
        ("s1", "A", 1, 4.0), ("s1", "A", 0, 6.0),
        ("s3", "A", 1, 3.0), ("s4", "A", 0, 5.0),
        ("s5", "B", 1, 2.0), ("s6", "B", 1, 8.0),
    ])
    _expect_422(post_job(client, bad, r2, xw), "样本号重复")


def test_crosswalk_bad_round_value(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad = tmp_path / "badround.csv"
    lines = xw.read_text(encoding="utf-8").splitlines()
    lines[1] = "3,s1,a1"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _expect_422(post_job(client, r1, r2, bad), "轮次只能是 1 或 2")


def test_crosswalk_duplicate_area_membership(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad = tmp_path / "duparea.csv"
    write_crosswalk_csv(bad, [
        (1, "s1", "a1"), (1, "s1", "a2"),  # s1 mapped twice
        (1, "s2", "a1"), (1, "s3", "a2"), (1, "s4", "a2"),
        (1, "s5", "b1"), (1, "s6", "b1"),
        (2, "t1", "a1"), (2, "t2", "a1"), (2, "t3", "a2"), (2, "t4", "a2"),
        (2, "t5", "b1"), (2, "t6", "b1"),
    ])
    _expect_422(post_job(client, r1, r2, bad), "只能归一个片区")


def test_crosswalk_foreign_and_missing_sample(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    foreign = tmp_path / "foreign.csv"
    with open(xw, encoding="utf-8", newline="") as f:
        xw_rows = list(csv_mod.reader(f))[1:]
    rows = [(1, "ghost", "a1")] + [
        row for row in xw_rows
        if not (row[0] == "1" and row[1] == "s1")
    ]
    write_crosswalk_csv(foreign, rows)
    _expect_422(post_job(client, r1, r2, foreign), "不存在的样本号")

    missing = tmp_path / "missing.csv"
    write_crosswalk_csv(missing, [
        row for row in xw_rows
        if not (row[0] == "2" and row[1] == "t6")
    ])
    _expect_422(post_job(client, r1, r2, missing), "缺少片区归属")


def test_area_set_mismatch_between_rounds(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad = tmp_path / "areamismatch.csv"
    # Round 2 uses a3 instead of a2 -> district A area sets differ.
    write_crosswalk_csv(bad, [
        (1, "s1", "a1"), (1, "s2", "a1"), (1, "s3", "a2"), (1, "s4", "a2"),
        (1, "s5", "b1"), (1, "s6", "b1"),
        (2, "t1", "a1"), (2, "t2", "a1"), (2, "t3", "a3"), (2, "t4", "a3"),
        (2, "t5", "b1"), (2, "t6", "b1"),
    ])
    _expect_422(post_job(client, r1, r2, bad), "片区集合须相同")


def test_district_set_mismatch_between_rounds(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad_r2 = tmp_path / "r2_district.csv"
    write_sample_csv(bad_r2, [
        ("t1", "C", 1, 9.0), ("t2", "C", 1, 1.0),
        ("t3", "C", 0, 2.0), ("t4", "C", 1, 7.0),
        ("t5", "B", 0, 5.0), ("t6", "B", 1, 5.0),
    ])
    _expect_422(post_job(client, r1, bad_r2, xw), "区集合不一致")


def test_area_spanning_districts_rejected(app_env, tmp_path):
    client, _ = app_env
    r1, r2, _ = make_simple_inputs(tmp_path)
    # Area a1 maps samples in both district A and district B within round 1.
    bad_r1 = tmp_path / "r1_span.csv"
    write_sample_csv(bad_r1, [
        ("s1", "A", 1, 4.0), ("s2", "A", 0, 6.0),
        ("s3", "A", 1, 3.0), ("s4", "A", 0, 5.0),
        ("s5", "B", 1, 2.0), ("s6", "B", 1, 8.0),
    ])
    bad_xw = tmp_path / "xw_span.csv"
    write_crosswalk_csv(bad_xw, [
        (1, "s1", "a1"), (1, "s2", "a1"), (1, "s3", "a2"), (1, "s4", "a2"),
        (1, "s5", "a1"), (1, "s6", "b1"),
        (2, "t1", "a1"), (2, "t2", "a1"), (2, "t3", "a2"), (2, "t4", "a2"),
        (2, "t5", "b1"), (2, "t6", "b1"),
    ])
    _expect_422(post_job(client, bad_r1, r2, bad_xw), "只能属于一个区")


def test_zero_area_main_weight_rejected(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    bad_r1 = tmp_path / "r1_zeroarea.csv"
    write_sample_csv(bad_r1, [
        ("s1", "A", 1, 4.0), ("s2", "A", 0, 6.0),
        ("s3", "A", 1, 0.0), ("s4", "A", 0, 0.0),  # area a2 sums to zero
        ("s5", "B", 1, 2.0), ("s6", "B", 1, 8.0),
    ])
    _expect_422(post_job(client, bad_r1, r2, xw), "主权重合计须为正")


def test_failed_generation_leaves_published_data_intact(app_env, tmp_path):
    client, _ = app_env
    # Publish valid data through the original entry first.
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    from tests.factories import hand_built, upload
    r1_csv, r1_pq, r2_csv, r2_pq = hand_built(demo_dir)
    assert upload(client, 1, r1_csv, r1_pq).status_code == 200
    assert upload(client, 2, r2_csv, r2_pq).status_code == 200
    before = client.get("/api/result", params={"district": "A"}).json()

    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    r1, r2, xw = make_simple_inputs(ok_dir)
    empty_r1 = tmp_path / "empty.csv"
    empty_r1.write_text("sample_id,district,bus,main_weight\n", encoding="utf-8")
    resp = post_job(client, empty_r1, r2, xw)
    assert resp.status_code == 422 and "为空" in resp.json()["detail"]

    after = client.get("/api/result", params={"district": "A"}).json()
    assert after["point_estimate"] == before["point_estimate"]
    assert after["ci_low"] == before["ci_low"]
    status = client.get("/api/status").json()
    assert status["rounds"]["1"]["loaded"] is True
    # No leftover job directory for failed jobs.
    jobs_root = tmp_path / "data" / "jobs"
    jobs = list(jobs_root.glob("*")) if jobs_root.exists() else []
    assert jobs == []


def test_download_unknown_job_404_and_bad_id_422(app_env, tmp_path):
    client, _ = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    resp = post_job(client, r1, r2, xw)
    job_id = resp.json()["job_id"]
    got = client.get(f"/api/weight-jobs/{job_id}/download")
    assert got.status_code == 200
    assert got.headers["content-type"] == "application/zip"
    assert client.get("/api/weight-jobs/deadbeefdeadbeef/download").status_code == 404
    bad = client.get("/api/weight-jobs/../etc/download")
    assert bad.status_code in (404, 422)
    assert client.get("/api/weight-jobs/zzzz/download").status_code == 422


def test_rng_helpers_are_stable():
    # Pin the exact draw stream so a future Python/library change cannot
    # silently renumber weights.
    rng = weights._replicate_rng(42, "A", 0)
    assert [rng.randrange(10) for _ in range(10)] == [
        9, 1, 9, 1, 8, 0, 8, 4, 7, 6
    ]
    assert weights._rep_id(0) == "rep000001"
    assert weights._rep_id(499) == "rep000500"
    # Independent keys -> independent streams.
    assert weights._replicate_rng(42, "A", 0).getrandbits(64) != \
        weights._replicate_rng(42, "A", 1).getrandbits(64)


def test_prune_old_jobs_bounds_retention(app_env, tmp_path):
    client, data_dir = app_env
    r1, r2, xw = make_simple_inputs(tmp_path)
    for _ in range(weights.JOBS_TO_KEEP + 2):
        assert post_job(client, r1, r2, xw, reps=2).status_code == 200
    jobs = list((data_dir / "jobs").iterdir())
    assert len(jobs) <= weights.JOBS_TO_KEEP
    # Remaining ZIPs still downloadable.
    for job in jobs:
        assert (job / "paired_replicate_weights.zip").is_file()
