"""ブラウザから Decision Model を使うための Web アプリ．"""

import os
from functools import cache
from importlib.resources import files

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from hakari.decision import Choice, DecisionClient, DecisionError, DecisionRequest

app = FastAPI(title="Hakari")


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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return files("hakari").joinpath("static/index.html").read_text(encoding="utf-8")


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


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HAKARI_HOST", "127.0.0.1"), port=int(os.environ.get("HAKARI_PORT", "8000")))
