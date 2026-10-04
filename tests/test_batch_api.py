import json

import pytest
from fastapi.testclient import TestClient

from hakari.app import app, get_job_store
from hakari.decision import ChoiceProbability, DecisionError, DecisionResult
from hakari.jobs import JobStore

CSV = "id,感想\n1,価格が高い\n2,接客が良い\n3,\n".encode()
TAGS = [{"name": "価格", "meaning": "価格への言及"}, {"name": "接客", "meaning": ""}]


class FakeClient:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    def decide(self, req):
        if self.fail_on and self.fail_on in req.context:
            raise DecisionError("Ollama が落ちた")
        p = 0.9 if req.field in req.context else 0.2
        probs = sorted(
            [ChoiceProbability("A", "true", p), ChoiceProbability("B", "false", 1 - p)],
            key=lambda x: x.probability,
            reverse=True,
        )
        return DecisionResult(answer=probs[0].label, probabilities=probs)


class DeferredRunner:
    """start 直後の状態を検証できるよう，実行を明示的に呼ぶまで遅らせる．"""

    def __init__(self):
        self.pending = []

    def __call__(self, fn):
        self.pending.append(fn)

    def run_all(self):
        while self.pending:
            self.pending.pop(0)()


@pytest.fixture
def runner():
    return DeferredRunner()


@pytest.fixture
def store(runner):
    return JobStore(client=FakeClient(), runner=runner)


@pytest.fixture
def client(store):
    app.dependency_overrides[get_job_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def start(client, csv=CSV, column="感想", tags=TAGS):
    return client.post(
        "/api/batch",
        files={"file": ("survey.csv", csv, "text/csv")},
        data={"column": column, "tags": json.dumps(tags, ensure_ascii=False)},
    )


def test_batch_page_serves_html(client):
    res = client.get("/batch")
    assert res.status_code == 200
    assert "<form" in res.text


def test_columns_returns_headers_and_row_count(client):
    res = client.post("/api/csv/columns", files={"file": ("s.csv", "id,感想\n1,a\n2,b\n".encode("cp932"), "text/csv")})
    assert res.status_code == 200
    assert res.json() == {"headers": ["id", "感想"], "rows": 2}


def test_columns_rejects_broken_csv(client):
    res = client.post("/api/csv/columns", files={"file": ("s.csv", b"", "text/csv")})
    assert res.status_code == 422


def test_start_returns_202_with_job_snapshot(client):
    res = start(client)
    assert res.status_code == 202
    body = res.json()
    assert body["status"] == "pending"
    assert body["total"] == 3
    assert body["done"] == 0


def test_status_reflects_progress_after_run(client, runner):
    job_id = start(client).json()["id"]
    runner.run_all()

    res = client.get(f"/api/batch/{job_id}")
    assert res.status_code == 200
    assert res.json() == {
        "id": job_id,
        "status": "done",
        "done": 3,
        "total": 3,
        "error": None,
        "summary": {"価格": 1, "接客": 1},
    }


def test_status_unknown_job_is_404(client):
    assert client.get("/api/batch/nope").status_code == 404


def test_result_csv_download(client, runner):
    job_id = start(client).json()["id"]
    runner.run_all()

    res = client.get(f"/api/batch/{job_id}/result.csv")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers["content-disposition"]
    lines = res.content.decode("utf-8-sig").splitlines()
    assert lines[0] == "id,感想,価格,価格_確率,接客,接客_確率"
    assert lines[1] == "1,価格が高い,TRUE,0.900,FALSE,0.200"


def test_result_csv_while_running_is_409(client):
    job_id = start(client).json()["id"]
    assert client.get(f"/api/batch/{job_id}/result.csv").status_code == 409


def test_result_csv_available_after_failure(store, client, runner):
    store.client = FakeClient(fail_on="接客")
    job_id = start(client).json()["id"]
    runner.run_all()

    assert client.get(f"/api/batch/{job_id}").json()["status"] == "failed"
    res = client.get(f"/api/batch/{job_id}/result.csv")
    assert res.status_code == 200
    assert len(res.content.decode("utf-8-sig").splitlines()) == 2


def test_cancel(client, runner):
    job_id = start(client).json()["id"]
    res = client.post(f"/api/batch/{job_id}/cancel")
    assert res.status_code == 200
    runner.run_all()
    assert client.get(f"/api/batch/{job_id}").json()["status"] == "cancelled"


def test_cancel_unknown_job_is_404(client):
    assert client.post("/api/batch/nope/cancel").status_code == 404


@pytest.mark.parametrize(
    "kwargs",
    [
        {"column": "無い列"},
        {"tags": []},
        {"tags": [{"name": " ", "meaning": ""}]},
        {"tags": [{"name": "価格"}, {"name": "価格"}]},
        {"csv": b""},
    ],
)
def test_start_rejects_invalid_input(client, kwargs):
    res = start(client, **kwargs)
    assert res.status_code == 422
    assert isinstance(res.json()["detail"], str)


def test_start_rejects_malformed_tags_json(client):
    res = client.post(
        "/api/batch",
        files={"file": ("s.csv", CSV, "text/csv")},
        data={"column": "感想", "tags": "not json"},
    )
    assert res.status_code == 422
