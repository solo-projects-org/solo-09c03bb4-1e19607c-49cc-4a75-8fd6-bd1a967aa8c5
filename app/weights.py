"""Paired replicate-weight generation for two survey rounds.

Inputs (all original-format CSV files with headers):

* round 1 / round 2 sample CSVs: ``sample_id, district, bus, main_weight``
  (the same format the ingestion endpoint already accepts)
* crosswalk CSV: ``round, sample_id, area_id``

Rules implemented here (see README, section "配对重抽权重生成"):

* every sample maps to exactly one area in exactly one round;
* the two rounds use the same set of areas inside every district;
* every area's main-weight total is positive in every round;
* per replicate, each district independently resamples *as many areas as it
  originally contains*, with replacement, and both rounds share the same
  per-area hit counts (paired bootstrap);
* a household's replicate weight is its main weight times its area's hit
  count, including explicit zero-weight rows for areas never drawn;
* draws are deterministic: a keyed PRNG seeded from
  ``(seed, district, replicate_index)`` drives lexicographically sorted area
  ids, so reordering input rows never changes the result and extending the
  replicate count with the same seed keeps every previously numbered weight.

Outputs are two Parquet files (schema ``sample_id, rep_id, weight``) plus
byte-for-byte copies of the uploaded sample CSVs, bundled into one ZIP that
the original ingestion entry can consume after extraction.  All heavy work
runs in a scratch DuckDB database on the data volume so the 384 MiB service
cap is respected; the published survey database is never touched, which is
why a failed generation leaves current data queryable.
"""
from __future__ import annotations

import csv as csv_mod
import hashlib
import json
import os
import random
import shutil
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from . import db

SAMPLE_COLUMNS = ["sample_id", "district", "bus", "main_weight"]
CROSSWALK_COLUMNS = ["round", "sample_id", "area_id"]
PARQUET_COLUMNS = ["sample_id", "rep_id", "weight"]

MAX_REPLICATES = 999_999  # rep ids are zero-padded to six digits
REP_ID_WIDTH = 6
# DuckDB executemany INSERT is pathologically slow in this build (one implicit
# round trip per row); draws are streamed into a scratch CSV instead and bulk
# loaded once via read_csv_auto, keeping only one district's state in memory.
JOBS_TO_KEEP = 10

# Generation is CPU- and memory-heavy; serialise jobs so one service process
# cannot run two scratch databases at once.
_generation_lock = threading.Lock()


class WeightGenerationError(Exception):
    """A validation or generation failure; maps to HTTP 422."""


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    job_dir: Path
    scratch_db: Path
    hits_csv: Path
    zip_path: Path
    parquet_round1: Path
    parquet_round2: Path


def _sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _column_names(con: duckdb.DuckDBPyConnection, reader_sql: str) -> list[str]:
    try:
        return [d[0].lower() for d in con.execute(f"SELECT * FROM {reader_sql} LIMIT 0").description]
    except Exception as exc:
        raise WeightGenerationError(f"无法读取文件: {exc}") from exc


def _check_columns(actual: list[str], expected: list[str], label: str) -> None:
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    problems = []
    if missing:
        problems.append(f"缺少列: {', '.join(missing)}")
    if extra:
        problems.append(f"多余列: {', '.join(extra)}")
    if problems:
        raise WeightGenerationError(
            f"{label}列结构不匹配（要求 {', '.join(expected)}）；" + "；".join(problems)
        )


def parse_positive_int(raw: str | None, label: str, maximum: int | None = None) -> int:
    if raw is None or str(raw).strip() == "":
        raise WeightGenerationError(f"请填写{label}")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise WeightGenerationError(f"{label}必须是整数") from exc
    if value <= 0:
        raise WeightGenerationError(f"{label}必须为正整数")
    if maximum is not None and value > maximum:
        raise WeightGenerationError(f"{label}不能超过 {maximum}")
    return value


def parse_seed(raw: str | None) -> int:
    if raw is None or str(raw).strip() == "":
        raise WeightGenerationError("请填写随机种子")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise WeightGenerationError("随机种子必须是整数") from exc
    if not (-(2**63) <= value <= 2**63 - 1):
        raise WeightGenerationError("随机种子须在 -2^63 到 2^63-1 之间")
    return value


def _examples(values: list[str], limit: int = 10) -> str:
    shown = values[:limit]
    tail = " …" if len(values) > limit else ""
    return ", ".join(shown) + tail


def create_job_dir() -> JobPaths:
    root = Path(db._data_dir()) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    job_id = os.urandom(12).hex()
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    return JobPaths(
        job_id=job_id,
        job_dir=job_dir,
        scratch_db=job_dir / "scratch.duckdb",
        hits_csv=job_dir / "hits.csv",
        zip_path=job_dir / "paired_replicate_weights.zip",
        parquet_round1=job_dir / "round1_replicates.parquet",
        parquet_round2=job_dir / "round2_replicates.parquet",
    )


def prune_old_jobs() -> None:
    """Bound on-disk retention: keep the newest few successful job folders."""
    root = Path(db._data_dir()) / "jobs"
    try:
        jobs = [p for p in root.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return
    jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in jobs[JOBS_TO_KEEP:]:
        shutil.rmtree(old, ignore_errors=True)


def generate(
    csv1_path: str,
    csv2_path: str,
    crosswalk_path: str,
    n_replicates: int,
    seed: int,
    paths: JobPaths,
) -> dict:
    """Validate inputs, draw paired replicates and assemble the ZIP.

    Raises :class:`WeightGenerationError` with a Chinese reason on any
    validation failure; the scratch directory is removed by the caller.
    """
    for label, p in (
        ("第一轮样本 CSV", csv1_path),
        ("第二轮样本 CSV", csv2_path),
        ("对照 CSV", crosswalk_path),
    ):
        if not Path(p).is_file():
            raise WeightGenerationError(f"{label}文件不存在")

    with _generation_lock:
        con = duckdb.connect(str(paths.scratch_db))
        try:
            con.execute(f"SET memory_limit = '{db.MEMORY_LIMIT}'")
            con.execute(f"SET threads = {db.THREADS}")
            con.execute(f"SET temp_directory = '{paths.job_dir / 'tmp'}'")
            (paths.job_dir / "tmp").mkdir(parents=True, exist_ok=True)

            _load_inputs(con, csv1_path, csv2_path, crosswalk_path)
            _validate_samples(con)
            _validate_crosswalk(con)
            district_areas = _validate_area_structure(con)
            _validate_positive_area_weights(con)
            n1, n2 = _sample_counts(con)

            _create_hits_table(con)
            _draw_hits_csv(district_areas, n_replicates, seed, paths.hits_csv)
            _load_hits_csv(con, paths.hits_csv)
            _verify_hits(con, district_areas, n_replicates)
            _write_round_parquet(con, 1, n1, n_replicates, paths.parquet_round1)
            _write_round_parquet(con, 2, n2, n_replicates, paths.parquet_round2)
        finally:
            con.close()

        manifest = _build_manifest(
            n_replicates, seed, n1, n2, district_areas
        )
        _assemble_zip(paths, csv1_path, csv2_path, manifest)

    return {
        "job_id": paths.job_id,
        "seed": seed,
        "n_replicates": n_replicates,
        "districts": [d for d, _ in district_areas],
        "round1": {"n_samples": n1, "n_rep_rows": n1 * n_replicates},
        "round2": {"n_samples": n2, "n_rep_rows": n2 * n_replicates},
    }


# ---------------------------------------------------------------- loading
def _load_inputs(
    con: duckdb.DuckDBPyConnection,
    csv1_path: str,
    csv2_path: str,
    crosswalk_path: str,
) -> None:
    readers = {
        1: f"read_csv_auto({_sql_string(csv1_path)})",
        2: f"read_csv_auto({_sql_string(csv2_path)})",
    }
    for round_no, reader in readers.items():
        cols = _column_names(con, reader)
        _check_columns(cols, SAMPLE_COLUMNS, f"第{round_no}轮样本 CSV")
    xw_reader = f"read_csv_auto({_sql_string(crosswalk_path)})"
    _check_columns(_column_names(con, xw_reader), CROSSWALK_COLUMNS, "对照 CSV")

    for round_no, reader in readers.items():
        try:
            con.execute(
                f"""
                CREATE TABLE s{round_no} AS
                SELECT CAST(sample_id AS VARCHAR)   AS sample_id,
                       CAST(district AS VARCHAR)    AS district,
                       CAST(bus AS INTEGER)         AS bus,
                       CAST(main_weight AS DOUBLE)  AS main_weight
                FROM {reader}
                """
            )
        except Exception as exc:
            raise WeightGenerationError(
                f"第{round_no}轮样本 CSV 类型转换失败"
                "（bus 须为 0/1，main_weight 须为数值）: "
                f"{exc}"
            ) from exc
    try:
        con.execute(
            f"""
            CREATE TABLE xwalk AS
            SELECT CAST(round AS INTEGER)     AS round,
                   CAST(sample_id AS VARCHAR) AS sample_id,
                   CAST(area_id AS VARCHAR)   AS area_id
            FROM {xw_reader}
            """
        )
    except Exception as exc:
        raise WeightGenerationError(
            f"对照 CSV 类型转换失败（round 须为 1 或 2 的整数）: {exc}"
        ) from exc


def _validate_samples(con: duckdb.DuckDBPyConnection) -> None:
    for round_no in (1, 2):
        n = con.execute(f"SELECT count(*) FROM s{round_no}").fetchone()[0]
        if n == 0:
            raise WeightGenerationError(f"第{round_no}轮样本 CSV 为空，至少需要一个样本")
        empty = con.execute(
            f"""SELECT count(*) FROM s{round_no}
                WHERE sample_id IS NULL OR trim(sample_id) = ''
                   OR district IS NULL OR trim(district) = ''"""
        ).fetchone()[0]
        if empty:
            raise WeightGenerationError(f"第{round_no}轮存在样本号或区为空的记录")
        bad_bus = con.execute(
            f"SELECT count(*) FROM s{round_no} WHERE bus NOT IN (0, 1) OR bus IS NULL"
        ).fetchone()[0]
        if bad_bus:
            raise WeightGenerationError(
                f"第{round_no}轮是否乘公交(bus) 只能取 0 或 1，发现 {bad_bus} 条违规记录"
            )
        bad_w = con.execute(
            f"""SELECT count(*) FROM s{round_no}
                WHERE main_weight IS NULL OR NOT isfinite(main_weight) OR main_weight < 0"""
        ).fetchone()[0]
        if bad_w:
            raise WeightGenerationError(
                f"第{round_no}轮主权重必须为有限非负数，发现 {bad_w} 条违规记录"
            )
        dup = con.execute(
            f"SELECT count(*) FROM "
            f"(SELECT sample_id FROM s{round_no} GROUP BY 1 HAVING count(*) > 1)"
        ).fetchone()[0]
        if dup:
            raise WeightGenerationError(
                f"第{round_no}轮样本号重复，涉及 {dup} 个样本号"
            )


def _validate_crosswalk(con: duckdb.DuckDBPyConnection) -> None:
    n = con.execute("SELECT count(*) FROM xwalk").fetchone()[0]
    if n == 0:
        raise WeightGenerationError("对照 CSV 为空，至少需要一条对照记录")

    bad_round = con.execute(
        "SELECT count(*) FROM xwalk WHERE round IS NULL OR round NOT IN (1, 2)"
    ).fetchone()[0]
    if bad_round:
        raise WeightGenerationError(
            f"对照 CSV 的轮次只能是 1 或 2，发现 {bad_round} 条违规记录"
        )
    empty = con.execute(
        """SELECT count(*) FROM xwalk
           WHERE sample_id IS NULL OR trim(sample_id) = ''
              OR area_id IS NULL OR trim(area_id) = ''"""
    ).fetchone()[0]
    if empty:
        raise WeightGenerationError("对照 CSV 存在样本号或片区号为空的记录")

    # One sample belongs to exactly one area: no duplicate (round, sample_id).
    dup = con.execute(
        "SELECT count(*) FROM "
        "(SELECT round, sample_id FROM xwalk GROUP BY 1, 2 HAVING count(*) > 1)"
    ).fetchone()[0]
    if dup:
        raise WeightGenerationError(
            f"每个样本在同一轮只能归一个片区，发现 {dup} 个样本号被重复对照"
        )

    # Every crosswalk row must point at an existing sample of its own round.
    foreign = con.execute(
        """
        SELECT count(*) FROM xwalk x
        WHERE NOT EXISTS (
            SELECT 1 FROM s1 WHERE x.round = 1 AND s1.sample_id = x.sample_id
        ) AND NOT EXISTS (
            SELECT 1 FROM s2 WHERE x.round = 2 AND s2.sample_id = x.sample_id
        )
        """
    ).fetchone()[0]
    if foreign:
        raise WeightGenerationError(
            f"对照 CSV 引用了对应轮次样本 CSV 中不存在的样本号，涉及 {foreign} 条记录"
        )

    # Every sample of every round must be covered exactly once.
    for round_no in (1, 2):
        missing = con.execute(
            f"""
            SELECT count(*) FROM s{round_no} s
            WHERE NOT EXISTS (
                SELECT 1 FROM xwalk x
                WHERE x.round = {round_no} AND x.sample_id = s.sample_id
            )
            """
        ).fetchone()[0]
        if missing:
            raise WeightGenerationError(
                f"第{round_no}轮有 {missing} 个样本在对照 CSV 中缺少片区归属"
            )

    # Materialise the joined sample/area map used by every later step.
    con.execute(
        """
        CREATE TABLE map AS
        SELECT x.round, x.area_id, s.sample_id, s.district, s.main_weight
        FROM xwalk x JOIN s1 s ON x.round = 1 AND x.sample_id = s.sample_id
        UNION ALL
        SELECT x.round, x.area_id, s.sample_id, s.district, s.main_weight
        FROM xwalk x JOIN s2 s ON x.round = 2 AND x.sample_id = s.sample_id
        """
    )


def _validate_area_structure(con: duckdb.DuckDBPyConnection) -> list[tuple[str, list[str]]]:
    # An area may not span multiple districts within one round.
    bad = con.execute(
        """
        SELECT round, area_id FROM map
        GROUP BY round, area_id HAVING count(DISTINCT district) > 1
        """
    ).fetchall()
    if bad:
        detail = _examples([f"第{r}轮 {a}" for r, a in bad])
        raise WeightGenerationError(f"同一片区在同一轮只能属于一个区，违规: {detail}")

    districts1 = {
        r[0] for r in con.execute(
            "SELECT DISTINCT district FROM map WHERE round = 1 ORDER BY 1"
        ).fetchall()
    }
    districts2 = {
        r[0] for r in con.execute(
            "SELECT DISTINCT district FROM map WHERE round = 2 ORDER BY 1"
        ).fetchall()
    }
    if districts1 != districts2:
        only1 = sorted(districts1 - districts2)
        only2 = sorted(districts2 - districts1)
        detail = []
        if only1:
            detail.append("仅第一轮有: " + _examples(only1))
        if only2:
            detail.append("仅第二轮有: " + _examples(only2))
        raise WeightGenerationError("两轮的区集合不一致（" + "；".join(detail) + "）")

    # Equal area sets within each district.
    rows = con.execute(
        "SELECT DISTINCT round, district, area_id FROM map"
    ).fetchall()
    areas: dict[int, dict[str, set[str]]] = {1: {}, 2: {}}
    for round_no, district, area_id in rows:
        areas[round_no].setdefault(district, set()).add(area_id)

    mismatched: list[str] = []
    district_areas: list[tuple[str, list[str]]] = []
    for district in sorted(districts1):
        a1 = areas[1].get(district, set())
        a2 = areas[2].get(district, set())
        if a1 != a2:
            parts = []
            only1 = sorted(a1 - a2)
            only2 = sorted(a2 - a1)
            if only1:
                parts.append("仅第一轮片区: " + _examples(only1))
            if only2:
                parts.append("仅第二轮片区: " + _examples(only2))
            mismatched.append(f"区 {district}（" + "；".join(parts) + "）")
        else:
            district_areas.append((district, sorted(a1)))
    if mismatched:
        raise WeightGenerationError(
            "两轮在各区的片区集合须相同，违规: " + "；".join(mismatched[:10])
            + (" …" if len(mismatched) > 10 else "")
        )
    return district_areas


def _validate_positive_area_weights(con: duckdb.DuckDBPyConnection) -> None:
    bad = con.execute(
        """
        SELECT round, area_id, sum(main_weight) AS w
        FROM map GROUP BY round, area_id
        HAVING sum(main_weight) IS NULL OR sum(main_weight) <= 0
        """
    ).fetchall()
    if bad:
        detail = _examples([f"第{r}轮 {a}" for r, a, _ in bad])
        raise WeightGenerationError(
            "每轮各片区的主权重合计须为正，零或非正的片区: " + detail
        )


def _sample_counts(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    n1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    n2 = con.execute("SELECT count(*) FROM s2").fetchone()[0]
    return int(n1), int(n2)


# ---------------------------------------------------------------- drawing
def _replicate_rng(seed: int, district: str, replicate_index: int) -> random.Random:
    """Stable per-(district, replicate) PRNG.

    Keying independently means extending the replicate count never reshuffles
    earlier numbers and districts stay independent of each other.  The
    Mersenne Twister seeding used by ``random.Random(int)`` is stable across
    Python 3 versions.
    """
    key = f"{seed}|{district}|{replicate_index}".encode("utf-8")
    sub_seed = int.from_bytes(
        hashlib.blake2b(key, digest_size=16).digest(), "big"
    )
    return random.Random(sub_seed)


def _rep_id(replicate_index: int) -> str:
    return "rep" + str(replicate_index + 1).zfill(REP_ID_WIDTH)


def _create_hits_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE hits (
            district VARCHAR,
            area_id  VARCHAR,
            rep_id   VARCHAR,
            hits     INTEGER
        )
        """
    )


def _draw_hits_csv(
    district_areas: list[tuple[str, list[str]]],
    n_replicates: int,
    seed: int,
    hits_csv: Path,
) -> None:
    """Stream every district's paired hit counts into a scratch CSV.

    Only the current district's draw state lives in Python memory; the bulk
    load into DuckDB happens once via ``read_csv_auto`` afterwards.
    """
    with open(hits_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        for district, area_ids in district_areas:
            n_areas = len(area_ids)
            for k in range(n_replicates):
                rng = _replicate_rng(seed, district, k)
                counts = [0] * n_areas
                for _ in range(n_areas):
                    counts[rng.randrange(n_areas)] += 1
                rep_id = _rep_id(k)
                writer.writerows(
                    (district, area_id, rep_id, hits)
                    for area_id, hits in zip(area_ids, counts)
                )


def _load_hits_csv(con: duckdb.DuckDBPyConnection, hits_csv: Path) -> None:
    con.execute(
        f"""
        INSERT INTO hits
        SELECT CAST(column0 AS VARCHAR), CAST(column1 AS VARCHAR),
               CAST(column2 AS VARCHAR), CAST(column3 AS INTEGER)
        FROM read_csv_auto({_sql_string(str(hits_csv))}, header=false)
        """
    )


def _verify_hits(
    con: duckdb.DuckDBPyConnection,
    district_areas: list[tuple[str, list[str]]],
    n_replicates: int,
) -> None:
    """Integrity check on the loaded hit table before any Parquet is written."""
    expected_rows = sum(len(areas) for _, areas in district_areas) * n_replicates
    n_rows, n_reps, n_dupes = con.execute(
        """
        SELECT count(*), count(DISTINCT rep_id),
               (SELECT count(*) FROM (
                   SELECT district, area_id, rep_id FROM hits
                   GROUP BY 1, 2, 3 HAVING count(*) > 1
               ))
        FROM hits
        """
    ).fetchone()
    if int(n_rows) != expected_rows:
        raise WeightGenerationError(
            f"片区抽中次数表行数 {n_rows} 与预期 {expected_rows} 不一致"
        )
    if int(n_reps) != n_replicates or int(n_dupes) != 0:
        raise WeightGenerationError("片区抽中次数表重抽覆盖不完整或存在重复")
    bad_hits = con.execute(
        "SELECT count(*) FROM hits WHERE hits IS NULL OR hits < 0"
    ).fetchone()[0]
    if bad_hits:
        raise WeightGenerationError(f"片区抽中次数存在非法值 {bad_hits} 条")
    # Each (district, replicate) must draw exactly as many hits as it has areas.
    bad_sum = con.execute(
        """
        SELECT count(*) FROM (
            SELECT h.district, h.rep_id, sum(h.hits) AS drawn, d.n_areas
            FROM hits h
            JOIN (
                SELECT district, count(DISTINCT area_id) AS n_areas
                FROM hits GROUP BY 1
            ) d ON d.district = h.district
            GROUP BY h.district, h.rep_id, d.n_areas
            HAVING sum(h.hits) <> d.n_areas
        )
        """
    ).fetchone()[0]
    if bad_sum:
        raise WeightGenerationError(
            f"存在 {bad_sum} 个 (区, 重抽) 的抽中次数合计不等于片区数"
        )


def _write_round_parquet(
    con: duckdb.DuckDBPyConnection,
    round_no: int,
    n_samples: int,
    n_replicates: int,
    out_path: Path,
) -> None:
    """Emit one round's Parquet: full coverage, zeros kept, deterministic order."""
    con.execute(
        f"""
        COPY (
            SELECT m.sample_id,
                   h.rep_id,
                   m.main_weight * CAST(h.hits AS DOUBLE) AS weight
            FROM map m
            JOIN hits h
              ON h.district = m.district AND h.area_id = m.area_id
            WHERE m.round = {round_no}
            ORDER BY m.sample_id, h.rep_id
        ) TO {_sql_string(str(out_path))} (FORMAT PARQUET)
        """
    )
    _verify_parquet(con, out_path, round_no, n_samples, n_replicates)


def _verify_parquet(
    con: duckdb.DuckDBPyConnection,
    out_path: Path,
    round_no: int,
    n_samples: int,
    n_replicates: int,
) -> None:
    reader = f"read_parquet({_sql_string(str(out_path))})"
    cols = _column_names(con, reader)
    _check_columns(cols, PARQUET_COLUMNS, f"第{round_no}轮输出 Parquet")
    n_rows, n_distinct_reps, n_distinct_samples, n_dupes = con.execute(
        f"""
        SELECT count(*),
               count(DISTINCT rep_id),
               count(DISTINCT sample_id),
               (SELECT count(*) FROM (
                   SELECT sample_id, rep_id FROM {reader}
                   GROUP BY 1, 2 HAVING count(*) > 1
               ))
        FROM {reader}
        """
    ).fetchone()
    expected_rows = n_samples * n_replicates
    if int(n_rows) != expected_rows:
        raise WeightGenerationError(
            f"第{round_no}轮 Parquet 行数 {n_rows} 与预期 {expected_rows} 不一致"
        )
    if int(n_distinct_reps) != n_replicates or int(n_distinct_samples) != n_samples:
        raise WeightGenerationError(
            f"第{round_no}轮 Parquet 覆盖不完整（样本 {n_distinct_samples}/{n_samples}，"
            f"重抽 {n_distinct_reps}/{n_replicates}）"
        )
    if int(n_dupes) != 0:
        raise WeightGenerationError(f"第{round_no}轮 Parquet 出现重复的 (样本, 重抽) 组合")
    bad_w = con.execute(
        f"SELECT count(*) FROM {reader} "
        "WHERE weight IS NULL OR NOT isfinite(weight) OR weight < 0"
    ).fetchone()[0]
    if bad_w:
        raise WeightGenerationError(f"第{round_no}轮 Parquet 出现非法权重 {bad_w} 条")


# ---------------------------------------------------------------- output
def _build_manifest(
    n_replicates: int,
    seed: int,
    n1: int,
    n2: int,
    district_areas: list[tuple[str, list[str]]],
) -> dict:
    return {
        "format": "paired-replicate-weights/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_replicates": n_replicates,
        "rep_id_pattern": f"rep000001 .. rep{n_replicates:0{REP_ID_WIDTH}d}",
        "rounds": {
            "1": {
                "sample_csv": "round1/samples.csv",
                "replicate_parquet": "round1/replicates.parquet",
                "n_samples": n1,
                "n_rep_rows": n1 * n_replicates,
            },
            "2": {
                "sample_csv": "round2/samples.csv",
                "replicate_parquet": "round2/replicates.parquet",
                "n_samples": n2,
                "n_rep_rows": n2 * n_replicates,
            },
        },
        "districts": [
            {"district": district, "n_areas": len(areas), "areas": areas}
            for district, areas in district_areas
        ],
    }


def _assemble_zip(
    paths: JobPaths,
    csv1_path: str,
    csv2_path: str,
    manifest: dict,
) -> None:
    """Write the ZIP only after both Parquet files exist and were verified."""
    if not paths.parquet_round1.is_file() or not paths.parquet_round2.is_file():
        raise WeightGenerationError("两轮 Parquet 未全部生成，拒绝提供下载")

    members = (
        ("round1/samples.csv", Path(csv1_path)),
        ("round1/replicates.parquet", paths.parquet_round1),
        ("round2/samples.csv", Path(csv2_path)),
        ("round2/replicates.parquet", paths.parquet_round2),
    )
    # STORED avoids extra compression memory/CPU; Parquet is already compressed.
    with zipfile.ZipFile(paths.zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, src in members:
            zf.write(src, arcname)
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        )

    with zipfile.ZipFile(paths.zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise WeightGenerationError(f"ZIP 完整性校验失败: {bad}")
        names = set(zf.namelist())
        expected = {m[0] for m in members} | {"manifest.json"}
        if names != expected:
            raise WeightGenerationError(
                f"ZIP 内容不符（缺少 {sorted(expected - names)}，"
                f"多余 {sorted(names - expected)}）"
            )
