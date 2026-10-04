"""ブラウザから Decision Model を使うための Web アプリ．"""

import json
import os
from functools import cache
from importlib.resources import files
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hakari.batch import BatchError, Tag, read_csv
from hakari.decision import Choice, DecisionClient, DecisionError, DecisionRequest
from hakari.jobs import FINISHED, Job, JobStore

app = FastAPI(title="Hakari")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class ChoiceIn(BaseModel):
    label: str
    description: str = ""


class DecideIn(BaseModel):
    context: str
    field: str
    meaning: str = ""
    choices: list[ChoiceIn]


class ProbabilityOut(BaseModel):
    code: str
    label: str
    probability: float


class DecideOut(BaseModel):
    answer: str
    probabilities: list[ProbabilityOut]


@cache
def get_decision_client() -> DecisionClient:
    return DecisionClient(
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        model=os.environ.get("HAKARI_MODEL", "nimble"),
    )


@cache
def get_job_store() -> JobStore:
    return JobStore(client=get_decision_client())


def _page(name: str) -> str:
    return files("hakari").joinpath(f"static/{name}").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page("index.html")


@app.get("/batch", response_class=HTMLResponse)
def batch_page() -> str:
    return _page("batch.html")


@app.post("/api/decide", response_model=DecideOut)
def decide(body: DecideIn, client: DecisionClient = Depends(get_decision_client)) -> DecideOut:
    try:
        req = DecisionRequest(
            context=body.context,
            field=body.field,
            meaning=body.meaning,
            choices=[Choice(c.label, c.description) for c in body.choices],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        result = client.decide(req)
    except DecisionError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return DecideOut(
        answer=result.answer,
        probabilities=[
            ProbabilityOut(code=p.code, label=p.label, probability=p.probability) for p in result.probabilities
        ],
    )


@app.post("/api/csv/columns")
async def csv_columns(file: UploadFile) -> dict:
    try:
        table = read_csv(await file.read())
    except BatchError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"headers": table.headers, "rows": len(table.rows)}


def _parse_tags(raw: str) -> list[Tag]:
    try:
        items = json.loads(raw)
        tags = [Tag(name=str(t["name"]).strip(), meaning=str(t.get("meaning", "")).strip()) for t in items]
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        raise HTTPException(status_code=422, detail=f"タグの指定が不正: {e}") from e
    if not tags:
        raise HTTPException(status_code=422, detail="タグを1つ以上指定してください")
    names = [t.name for t in tags]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=422, detail="タグ名が重複している")
    return tags


@app.post("/api/batch", status_code=202)
async def start_batch(
    file: UploadFile,
    column: str = Form(...),
    tags: str = Form(...),
    store: JobStore = Depends(get_job_store),
) -> dict:
    parsed_tags = _parse_tags(tags)
    try:
        table = read_csv(await file.read())
        index = table.column_index(column)
    except BatchError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return store.start(Job(table=table, column=index, tags=parsed_tags)).snapshot()


def _get_job(job_id: str, store: JobStore) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つからない")
    return job


@app.get("/api/batch/{job_id}")
def batch_status(job_id: str, store: JobStore = Depends(get_job_store)) -> dict:
    return _get_job(job_id, store).snapshot()


@app.post("/api/batch/{job_id}/cancel")
def cancel_batch(job_id: str, store: JobStore = Depends(get_job_store)) -> dict:
    job = _get_job(job_id, store)
    job.cancel()
    return job.snapshot()


@app.get("/api/batch/{job_id}/result.csv")
def batch_result(job_id: str, store: JobStore = Depends(get_job_store)) -> Response:
    job = _get_job(job_id, store)
    if job.status not in FINISHED:
        raise HTTPException(status_code=409, detail="処理中のためまだダウンロードできない")
    return Response(
        content=job.result_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="hakari_result.csv"'},
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HAKARI_HOST", "127.0.0.1"), port=int(os.environ.get("HAKARI_PORT", "8000")))
