"""End-to-end scale test for paired replicate-weight generation.

Spawns the real uvicorn server in a subprocess (optionally inside a cgroup
memory cap emulating the Docker Compose 384 MiB limit), then:

1. builds two rounds of sample CSVs plus a crosswalk CSV whose area sets
   match between rounds (n_samples * n_reps >= 5,000,000 weight rows each),
2. POSTs them to /api/weight-jobs with an explicit replicate count and seed,
3. downloads the resulting ZIP,
4. independently recomputes the paired hit counts and checks every weight in
   both Parquet files, the explicit zero rows and round-1/round-2 pairing,
5. re-POSTs with shuffled rows and with an extended replicate count to check
   determinism and stable numbering,
6. posts an invalid job and confirms the service stays healthy afterwards,
7. measures the service process peak RSS.

Run:
  python scripts/weight_scale_test.py --rows 5000000 --mem-mb 384 --out /tmp/wscale
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from app import weights  # noqa: E402


def post_job(base: str, files: dict, fields: dict) -> tuple[int, dict]:
    boundary = "----wscaleboundary" + os.urandom(8).hex()
    parts = []
    for field, path in files.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{Path(path).name}"\r\n\r\n'.encode()
        )
        parts.append(Path(path).read_bytes())
        parts.append(b"\r\n")
    for field, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        base + "/api/weight-jobs",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=1800) as resp:
        return resp.read()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=300) as resp:
        return json.loads(resp.read().decode())


def wait_ready(base: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            get_json(base + "/api/status")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def rss_mib(pid: int) -> float:
    statm = Path(f"/proc/{pid}/statm").read_text().split()
    page_kb = resource.getpagesize() // 1024
    return int(statm[1]) * page_kb / 1024


def make_inputs(out: Path, n_samples: int, n_areas: int, seed: int = 7):
    """Two rounds with identical district/area structure, distinct sample ids."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE TABLE base AS
            SELECT i,
                   'D' || (i % 8) AS district,
                   i % ? AS area_idx,
                   (abs(hash(i + ?)) % 2) AS bus,
                   round(1 + (abs(hash(i * 7 + ?)) % 1000) / 100.0, 4) AS main_weight
            FROM range(?) t(i)
            """,
            [n_areas, seed, seed, n_samples],
        )
        r1_csv = out / "round1_samples.csv"
        r2_csv = out / "round2_samples.csv"
        con.execute(
            "COPY (SELECT 's' || i AS sample_id, district, bus, main_weight "
            "FROM base ORDER BY i) TO ? (HEADER, DELIMITER ',')",
            [str(r1_csv)],
        )
        con.execute(
            "COPY (SELECT 't' || i AS sample_id, district, bus, "
            "round(main_weight * 1.1, 4) AS main_weight FROM base ORDER BY i) "
            "TO ? (HEADER, DELIMITER ',')",
            [str(r2_csv)],
        )
        pairs = con.execute(
            "SELECT DISTINCT district, area_idx FROM base ORDER BY 1, 2"
        ).fetchall()
    finally:
        con.close()

    xw_csv = out / "crosswalk.csv"
    with open(xw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "sample_id", "area_id"])
        for district, idx in pairs:
            area_id = f"{district}_a{idx:04d}"
            for i in range(n_samples):
                if f"D{i % 8}" == district and i % n_areas == idx:
                    w.writerow((1, f"s{i}", area_id))
                    w.writerow((2, f"t{i}", area_id))
    return r1_csv, r2_csv, xw_csv


def shuffle_csv(src: Path, dst: Path) -> None:
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    body.reverse()
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--areas", type=int, default=200)
    ap.add_argument("--mem-mb", type=int, default=384)
    ap.add_argument("--out", default="/tmp/wscale")
    ap.add_argument("--no-cgroup", action="store_true")
    ap.add_argument("--duckdb-mem", default="180MB")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    n_samples = max(args.areas, args.rows // args.reps)
    actual_rows = n_samples * args.reps
    print(f"[plan] {n_samples} samples/round x {args.reps} reps = "
          f"{actual_rows:,} weight rows/round; {args.areas} areas")

    print("[gen] building input CSVs ...")
    t0 = time.time()
    r1_csv, r2_csv, xw_csv = make_inputs(out / "inputs", n_samples, args.areas)
    print(f"      done in {time.time()-t0:.1f}s; csv sizes "
          f"r1={r1_csv.stat().st_size/1e6:.1f} MiB xw={xw_csv.stat().st_size/1e6:.1f} MiB")

    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(data_dir),
        "UPLOAD_TMP_DIR": str(data_dir / "tmp"),
        "DUCKDB_MEMORY_LIMIT": args.duckdb_mem,
        "DUCKDB_THREADS": "2",
        "PYTHONPATH": str(ROOT),
    })
    cmd = [
        sys.executable, "-I", "-m", "uvicorn",
        "app.main:app", "--host", "127.0.0.1", "--port", "8124", "--workers", "1",
    ]
    cgroup_path = None
    if not args.no_cgroup and os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        cgroup_path = Path("/sys/fs/cgroup/wscale")
        try:
            cgroup_path.mkdir(exist_ok=True)
            (cgroup_path / "memory.max").write_text(str(args.mem_mb * 1024 * 1024))
            (cgroup_path / "memory.swap.max").write_text("0")
            print(f"[cg] memory.max = {args.mem_mb} MiB")
        except (PermissionError, OSError) as exc:
            print(f"[cg] unavailable ({exc}); measuring peak RSS instead")
            cgroup_path = None

    log = open(out / "server.log", "wb")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT)
    )
    try:
        if cgroup_path is not None:
            try:
                (cgroup_path / "cgroup.procs").write_text(str(proc.pid))
            except PermissionError:
                cgroup_path = None
        base = "http://127.0.0.1:8124"
        wait_ready(base)
        print(f"[srv] pid={proc.pid} ready; baseline RSS={rss_mib(proc.pid):.1f} MiB")

        files = {
            "round1_sample_csv": r1_csv,
            "round2_sample_csv": r2_csv,
            "crosswalk_csv": xw_csv,
        }
        t0 = time.time()
        code, payload = post_job(
            base, files, {"n_replicates": args.reps, "seed": 20261120}
        )
        print(f"[gen-http] HTTP {code} in {time.time()-t0:.1f}s -> "
              f"{json.dumps(payload, ensure_ascii=False)[:240]}")
        if code != 200:
            return 2
        assert payload["round1"]["n_rep_rows"] >= args.rows
        assert payload["round2"]["n_rep_rows"] >= args.rows

        t0 = time.time()
        blob = get_bytes(base + payload["download_url"])
        print(f"[zip] {len(blob)/1e6:.1f} MiB in {time.time()-t0:.1f}s "
              f"sha256={hashlib.sha256(blob).hexdigest()[:16]}")
        zf = zipfile.ZipFile(io.BytesIO(blob))

        # ---- reference material ---------------------------------------------
        manifest = json.loads(zf.read("manifest.json"))
        district_of_area = {
            a: d["district"] for d in manifest["districts"] for a in d["areas"]
        }
        areas_of_district = {d["district"]: d["areas"] for d in manifest["districts"]}

        # Reference hits replayed independently, spilled to CSV exactly like
        # the server does internally (this validates the RNG seeding scheme,
        # not the production code path which builds the same file itself).
        ref_hits_csv = out / "ref_hits.csv"
        t0 = time.time()
        with open(ref_hits_csv, "w", newline="") as f:
            w = csv.writer(f)
            for district in sorted(areas_of_district):
                areas = areas_of_district[district]
                for k in range(args.reps):
                    rng = weights._replicate_rng(20261120, district, k)
                    counts = [0] * len(areas)
                    for _ in range(len(areas)):
                        counts[rng.randrange(len(areas))] += 1
                    rep_id = weights._rep_id(k)
                    for area, hits in zip(areas, counts):
                        w.writerow((district, area, rep_id, hits))
        print(f"[verify] reference hits replayed in {time.time()-t0:.1f}s")

        main_w = {1: {}, 2: {}}
        for round_no, p in ((1, r1_csv), (2, r2_csv)):
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    main_w[round_no][row["sample_id"]] = float(row["main_weight"])
        membership = {}
        with open(xw_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                membership[(int(row["round"]), row["sample_id"])] = row["area_id"]

        # Spill the bundled Parquets so every check runs inside DuckDB without
        # materialising 5 million rows in the test process.
        pq1 = out / "verify_r1.parquet"
        pq2 = out / "verify_r2.parquet"
        pq1.write_bytes(zf.read("round1/replicates.parquet"))
        pq2.write_bytes(zf.read("round2/replicates.parquet"))
        xw_spill = out / "verify_xw.csv"
        xw_spill.write_bytes(xw_csv.read_bytes())
        r1_spill = out / "verify_r1.csv"
        r2_spill = out / "verify_r2.csv"
        r1_spill.write_bytes(r1_csv.read_bytes())
        r2_spill.write_bytes(r2_csv.read_bytes())

        con = duckdb.connect()
        con.execute("SET memory_limit='256MB'")
        con.execute(f"SET temp_directory='{out / 'verify_tmp'}'")
        (out / "verify_tmp").mkdir(exist_ok=True)
        try:
            con.execute(
                "CREATE TABLE ref_hits AS SELECT column0 AS district, "
                "column1 AS area_id, column2 AS rep_id, "
                "CAST(column3 AS INTEGER) AS hits "
                "FROM read_csv_auto(?, header=false)",
                [str(ref_hits_csv)],
            )
            con.execute(
                "CREATE TABLE xw AS SELECT CAST(round AS INTEGER) AS round, "
                "sample_id, area_id "
                "FROM read_csv_auto(?, header=true)",
                [str(xw_spill)],
            )

            # ---- structural checks ------------------------------------------
            for member, pq, n_smp in (
                ("round1", pq1, n_samples), ("round2", pq2, n_samples)
            ):
                n_rows, n_rep, n_smp_db, n_zero, n_dupe, n_bad_total = con.execute(
                    f"""
                    WITH w AS (SELECT * FROM read_parquet('{pq.as_posix()}'))
                    SELECT (SELECT count(*) FROM w),
                           (SELECT count(DISTINCT rep_id) FROM w),
                           (SELECT count(DISTINCT sample_id) FROM w),
                           (SELECT count(*) FROM w WHERE weight = 0),
                           (SELECT count(*) FROM (
                               SELECT sample_id, rep_id FROM w
                               GROUP BY 1, 2 HAVING count(*) > 1)),
                           (SELECT count(*) FROM (
                               SELECT rep_id FROM w GROUP BY 1 HAVING sum(weight) <= 0))
                    """
                ).fetchone()
                assert (n_rows, n_rep, n_smp_db) == (
                    n_smp * args.reps, args.reps, n_smp
                ), member
                assert n_zero > 0 and n_dupe == 0 and n_bad_total == 0, member
            print("[verify] structure: full coverage, explicit zeros, "
                  "positive totals, no duplicate pairs")

            # ---- exact value check: weight must equal main_weight x hits ----
            for round_no, sample_csv, pq in (
                (1, r1_spill, pq1), (2, r2_spill, pq2)
            ):
                mismatches = con.execute(
                    f"""
                    WITH s AS (
                        SELECT sample_id, district,
                               CAST(main_weight AS DOUBLE) AS main_weight
                        FROM read_csv_auto('{sample_csv.as_posix()}', header=true)
                    ), w AS (SELECT * FROM read_parquet('{pq.as_posix()}')),
                    m AS (
                        SELECT x.sample_id, x.area_id, s.district, s.main_weight
                        FROM xw x JOIN s ON s.sample_id = x.sample_id
                        WHERE x.round = {round_no}
                    )
                    SELECT count(*) FROM w
                    JOIN m USING (sample_id)
                    JOIN ref_hits h
                      ON h.district = m.district AND h.area_id = m.area_id
                     AND h.rep_id = w.rep_id
                    WHERE abs(w.weight - m.main_weight * h.hits) > 1e-12
                    """
                ).fetchone()[0]
                assert mismatches == 0, (round_no, mismatches)
            print("[verify] every weight equals main_weight x replayed hits "
                  "(both rounds, exact)")

            # ---- pairing: s_i and t_i in the same area share hit factors ----
            pair_bad = con.execute(
                f"""
                WITH w1 AS (SELECT * FROM read_parquet('{pq1.as_posix()}')),
                     w2 AS (SELECT * FROM read_parquet('{pq2.as_posix()}')),
                     s1 AS (
                        SELECT sample_id, CAST(main_weight AS DOUBLE) AS mw
                        FROM read_csv_auto('{r1_spill.as_posix()}', header=true)),
                     s2 AS (
                        SELECT sample_id, CAST(main_weight AS DOUBLE) AS mw
                        FROM read_csv_auto('{r2_spill.as_posix()}', header=true)),
                     x1 AS (SELECT sample_id, area_id FROM xw WHERE round = 1),
                     x2 AS (SELECT sample_id, area_id FROM xw WHERE round = 2),
                     f1 AS (
                        SELECT w1.rep_id, x1.area_id,
                               w1.weight / s1.mw AS factor
                        FROM w1 JOIN x1 USING (sample_id) JOIN s1 USING (sample_id)),
                     f2 AS (
                        SELECT w2.rep_id, x2.area_id,
                               w2.weight / s2.mw AS factor
                        FROM w2 JOIN x2 USING (sample_id) JOIN s2 USING (sample_id))
                    SELECT count(*) FROM f1 JOIN f2 USING (rep_id, area_id)
                    WHERE abs(f1.factor - f2.factor) > 1e-12
                    """
            ).fetchone()[0]
            assert pair_bad == 0, pair_bad
            print("[verify] paired hit factors identical across rounds")

            # ---- original sample CSV bytes preserved -----------------------
            assert zf.read("round1/samples.csv") == r1_csv.read_bytes()
            assert zf.read("round2/samples.csv") == r2_csv.read_bytes()
            assert manifest["n_replicates"] == args.reps
            assert manifest["seed"] == 20261120
        finally:
            con.close()
            for p in (ref_hits_csv, xw_spill, r1_spill, r2_spill, pq1, pq2):
                p.unlink(missing_ok=True)
            shutil.rmtree(out / "verify_tmp", ignore_errors=True)

        # ---- determinism: shuffled rows -> byte-identical -------------------
        shuf = out / "shuf"
        shuf.mkdir(exist_ok=True)
        shuffle_csv(r1_csv, shuf / "r1.csv")
        shuffle_csv(r2_csv, shuf / "r2.csv")
        shuffle_csv(xw_csv, shuf / "xw.csv")
        code2, payload2 = post_job(
            base,
            {"round1_sample_csv": shuf / "r1.csv",
             "round2_sample_csv": shuf / "r2.csv",
             "crosswalk_csv": shuf / "xw.csv"},
            {"n_replicates": args.reps, "seed": 20261120},
        )
        assert code2 == 200, payload2
        zf2 = zipfile.ZipFile(io.BytesIO(get_bytes(base + payload2["download_url"])))
        for name in ("round1/replicates.parquet", "round2/replicates.parquet"):
            assert hashlib.sha256(zf.read(name)).hexdigest() == \
                hashlib.sha256(zf2.read(name)).hexdigest(), name
        print("[determinism] shuffled rows produce byte-identical Parquet")

        # ---- same seed, more reps: existing numbering unchanged -------------
        code3, payload3 = post_job(
            base, files,
            {"n_replicates": args.reps + 10, "seed": 20261120},
        )
        assert code3 == 200, payload3
        zf3 = zipfile.ZipFile(io.BytesIO(get_bytes(base + payload3["download_url"])))
        a_csv, b_csv = out / "_a.parquet", out / "_b.parquet"
        a_csv.write_bytes(zf.read("round1/replicates.parquet"))
        b_csv.write_bytes(zf3.read("round1/replicates.parquet"))
        con = duckdb.connect()
        try:
            kept, extended_rows = con.execute(
                f"""
                WITH a AS (SELECT * FROM read_parquet('{a_csv.as_posix()}')),
                     b AS (SELECT * FROM read_parquet('{b_csv.as_posix()}'))
                SELECT (
                    SELECT count(*) FROM a JOIN b USING (sample_id, rep_id, weight)
                ), (SELECT count(*) FROM b)
                """
            ).fetchone()
        finally:
            con.close()
            a_csv.unlink(missing_ok=True)
            b_csv.unlink(missing_ok=True)
        assert kept == n_samples * args.reps
        assert extended_rows == n_samples * (args.reps + 10)
        print(f"[determinism] first {args.reps} reps unchanged when extending "
              f"to {args.reps + 10}")

        # ---- invalid job leaves the service healthy -------------------------
        bad_csv = out / "empty.csv"
        bad_csv.write_text("sample_id,district,bus,main_weight\n", encoding="utf-8")
        code4, payload4 = post_job(
            base,
            {"round1_sample_csv": bad_csv, "round2_sample_csv": r2_csv,
             "crosswalk_csv": xw_csv},
            {"n_replicates": "10", "seed": "1"},
        )
        assert code4 == 422 and "为空" in payload4["detail"], payload4
        assert get_json(base + "/api/status")["rounds"]["1"]["loaded"] is False
        print("[isolation] invalid generation rejected with reason; service healthy")

        peak = rss_mib(proc.pid)
        cg_peak = None
        if cgroup_path is not None:
            p = cgroup_path / "memory.peak"
            if p.exists():
                cg_peak = int(p.read_text().strip()) / 1024 / 1024
        print(f"[mem] process RSS at finish: {peak:.1f} MiB; "
              f"cgroup peak: {f'{cg_peak:.1f} MiB' if cg_peak else 'n/a'} "
              f"(limit {args.mem_mb} MiB)")
        print("[OK] all weight-generation scale checks passed")
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
