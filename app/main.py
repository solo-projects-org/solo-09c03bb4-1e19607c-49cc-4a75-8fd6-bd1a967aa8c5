"""FastAPI application: upload rounds, inspect status, query districts."""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import analysis, db, ingest, weights

UPLOAD_ROOT = Path(os.environ.get("UPLOAD_TMP_DIR", str(db.TMP_DIR)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(title="公交出行比例两轮重抽分析", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict:
    payload = {"rounds": {}}
    for round_no in (1, 2):
        gen = db.current_generation(round_no)
        if gen is None:
            payload["rounds"][str(round_no)] = {"loaded": False}
            continue
        info = db.generation_info(gen)
        # info: round, created_at, n_samples, n_replicates, n_rep_rows
        payload["rounds"][str(round_no)] = {
            "loaded": True,
            "generation": gen,
            "created_at": info[1].isoformat() if info else None,
            "n_samples": info[2],
            "n_replicates": info[3],
            "n_rep_rows": info[4],
        }
    payload["districts"] = analysis.list_districts()
    return payload


@app.post("/api/rounds/{round_no}")
def upload_round(
    round_no: int,
    sample_csv: UploadFile = File(...),
    replicate_parquet: UploadFile = File(...),
) -> JSONResponse:
    # Sync endpoint: Starlette runs it in the worker thread, and uploads are
    # streamed straight to disk instead of being buffered in memory.
    if round_no not in (1, 2):
        raise HTTPException(status_code=422, detail="轮次必须是 1 或 2")
    if not sample_csv.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="样本文件必须是 .csv")
    if not replicate_parquet.filename.lower().endswith(".parquet"):
        raise HTTPException(status_code=422, detail="重抽权重文件必须是 .parquet")

    upload_dir = tempfile.mkdtemp(prefix=f"round{round_no}_", dir=str(UPLOAD_ROOT))
    try:
        csv_path = ingest.save_upload(upload_dir, sample_csv.filename, sample_csv.file)
        pq_path = ingest.save_upload(
            upload_dir, replicate_parquet.filename, replicate_parquet.file
        )
        try:
            result = ingest.ingest_round(round_no, csv_path, pq_path)
        except ingest.IngestionError as exc:
            # Failed replacement: previously published data remains queryable.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        ingest.remove_upload_dir(upload_dir)

    return JSONResponse({"ok": True, **result})


def _require_csv(upload: UploadFile, label: str) -> None:
    if not (upload.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail=f"{label}必须是 .csv")


@app.post("/api/weight-jobs")
def create_weight_job(
    round1_sample_csv: UploadFile = File(...),
    round2_sample_csv: UploadFile = File(...),
    crosswalk_csv: UploadFile = File(...),
    n_replicates: str | None = Form(None),
    seed: str | None = Form(None),
) -> JSONResponse:
    """Validate the three CSVs, draw paired replicates, build one ZIP.

    Runs in Starlette's worker thread; uploads stream to disk first and the
    scratch DuckDB lives on the data volume, so the service memory cap is
    unaffected by input size.  A failure here never touches the published
    survey database, so current data stays queryable.
    """
    _require_csv(round1_sample_csv, "第一轮样本文件")
    _require_csv(round2_sample_csv, "第二轮样本文件")
    _require_csv(crosswalk_csv, "对照文件")

    try:
        n_reps = weights.parse_positive_int(
            n_replicates, "重抽次数", weights.MAX_REPLICATES
        )
        seed_value = weights.parse_seed(seed)
    except weights.WeightGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    upload_dir = tempfile.mkdtemp(prefix="wjob_in_", dir=str(UPLOAD_ROOT))
    job = weights.create_job_dir()
    try:
        csv1 = ingest.save_upload(upload_dir, round1_sample_csv.filename, round1_sample_csv.file)
        csv2 = ingest.save_upload(upload_dir, round2_sample_csv.filename, round2_sample_csv.file)
        xw = ingest.save_upload(upload_dir, crosswalk_csv.filename, crosswalk_csv.file)
        try:
            result = weights.generate(csv1, csv2, xw, n_reps, seed_value, job)
        except weights.WeightGenerationError as exc:
            shutil.rmtree(job.job_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            # Never leave a half-written job directory behind on hard failures.
            shutil.rmtree(job.job_dir, ignore_errors=True)
            raise
    finally:
        ingest.remove_upload_dir(upload_dir)

    weights.prune_old_jobs()
    return JSONResponse(
        {
            "ok": True,
            **result,
            "download_url": f"/api/weight-jobs/{result['job_id']}/download",
        }
    )


@app.get("/api/weight-jobs/{job_id}/download")
def download_weight_job(job_id: str) -> FileResponse:
    # Job ids are server-generated hex tokens; never interpolate user input
    # into a path beyond an exact-name lookup inside the jobs directory.
    if not job_id or not all(c in "0123456789abcdef" for c in job_id):
        raise HTTPException(status_code=422, detail="任务号不合法")
    zip_path = db._data_dir() / "jobs" / job_id / "paired_replicate_weights.zip"
    if not zip_path.is_file():
        raise HTTPException(status_code=404, detail="任务不存在或已过期，请重新生成")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="paired_replicate_weights.zip",
    )


@app.get("/api/districts")
def districts() -> dict:
    return {"districts": analysis.list_districts()}


# GET with a query parameter is the natural fit for the browser UI.
@app.get("/api/result")
def result(district: str) -> dict:
    district = (district or "").strip()
    if not district:
        raise HTTPException(status_code=422, detail="请选择一个区")
    try:
        return analysis.compute(district)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
