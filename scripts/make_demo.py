"""Generate a tiny two-round demo dataset for manual browser/API trials.

Output layout (under --out, default ./demo_data):
  round1/samples.csv, round1/replicates.parquet
  round2/samples.csv, round2/replicates.parquet
  mismatch/round2_replicates.parquet   (different rep ids -> blank interval)
  weight_job/round1_samples.csv, round2_samples.csv, crosswalk.csv
      (inputs for POST /api/weight-jobs; districts A/B, areas a1/a2/b1)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb


def write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "district", "bus", "main_weight"])
        w.writerows(rows)


def write_parquet(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("CREATE TEMP TABLE r (sample_id VARCHAR, rep_id VARCHAR, weight DOUBLE)")
        con.executemany("INSERT INTO r VALUES (?, ?, ?)", rows)
        con.execute("COPY r TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data")
    args = ap.parse_args()
    root = Path(args.out)

    # Round 1: district A s1(bus=1,w=4) s2(bus=0,w=6); district B s3(bus=1,w=2)
    write_csv(root / "round1" / "samples.csv", [
        ("s1", "A", 1, 4.0), ("s2", "A", 0, 6.0), ("s3", "B", 1, 2.0),
    ])
    write_parquet(root / "round1" / "replicates.parquet", [
        ("s1", "rep1", 8.0), ("s2", "rep1", 2.0), ("s3", "rep1", 2.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 10.0), ("s3", "rep2", 2.0),
    ])

    # Round 2: A t1(bus=1,w=9) t2(bus=1,w=1); B t3(bus=0,w=5)
    write_csv(root / "round2" / "samples.csv", [
        ("t1", "A", 1, 9.0), ("t2", "A", 1, 1.0), ("t3", "B", 0, 5.0),
    ])
    write_parquet(root / "round2" / "replicates.parquet", [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0),
        ("t1", "rep2", 6.0), ("t2", "rep2", 4.0), ("t3", "rep2", 5.0),
    ])

    # Mismatched replicate ids (rep1/rep3 instead of rep1/rep2)
    write_parquet(root / "mismatch" / "round2_replicates.parquet", [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0),
        ("t1", "rep3", 6.0), ("t2", "rep3", 4.0), ("t3", "rep3", 5.0),
    ])

    # Weight-generation job inputs: original-format sample CSVs plus a
    # crosswalk. District A holds areas a1/a2 (two samples each), district B
    # a single area b1; the area sets match across both rounds.
    wj = root / "weight_job"
    write_csv(wj / "round1_samples.csv", [
        ("s1", "A", 1, 4.0), ("s2", "A", 0, 6.0),
        ("s3", "A", 1, 3.0), ("s4", "A", 0, 5.0),
        ("s5", "B", 1, 2.0), ("s6", "B", 1, 8.0),
    ])
    write_csv(wj / "round2_samples.csv", [
        ("t1", "A", 1, 9.0), ("t2", "A", 1, 1.0),
        ("t3", "A", 0, 2.0), ("t4", "A", 1, 7.0),
        ("t5", "B", 0, 5.0), ("t6", "B", 1, 5.0),
    ])
    xw = wj / "crosswalk.csv"
    xw.parent.mkdir(parents=True, exist_ok=True)
    with open(xw, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "sample_id", "area_id"])
        w.writerows([
            (1, "s1", "a1"), (1, "s2", "a1"), (1, "s3", "a2"), (1, "s4", "a2"),
            (1, "s5", "b1"), (1, "s6", "b1"),
            (2, "t1", "a1"), (2, "t2", "a1"), (2, "t3", "a2"), (2, "t4", "a2"),
            (2, "t5", "b1"), (2, "t6", "b1"),
        ])
    print(f"demo data written under {root}")


if __name__ == "__main__":
    main()
