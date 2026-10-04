import pytest
from fastapi.testclient import TestClient

from hakari.app import app, get_decision_client
from hakari.decision import ChoiceProbability, DecisionError, DecisionResult


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.requests = []

    def decide(self, req):
        self.requests.append(req)
        if self.error:
            raise self.error
        return self.result


RESULT = DecisionResult(
    answer="negative",
    probabilities=[
        ChoiceProbability("B", "negative", 0.9),
        ChoiceProbability("A", "positive", 0.1),
    ],
)

BODY = {
    "context": "料理が冷めていた．",
    "field": "sentiment",
    "meaning": "感情",
    "choices": [{"label": "positive"}, {"label": "negative", "description": "不満"}],
}


@pytest.fixture
def fake():
    return FakeClient(result=RESULT)


@pytest.fixture
def client(fake):
    app.dependency_overrides[get_decision_client] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_index_serves_html_form(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<form" in res.text


def test_decide_returns_answer_and_probabilities(client, fake):
    res = client.post("/api/decide", json=BODY)

    assert res.status_code == 200
    assert res.json() == {
        "answer": "negative",
        "probabilities": [
            {"code": "B", "label": "negative", "probability": 0.9},
            {"code": "A", "label": "positive", "probability": 0.1},
        ],
    }
    req = fake.requests[0]
    assert req.context == "料理が冷めていた．"
    assert req.field == "sentiment"
    assert req.meaning == "感情"
    assert [(c.label, c.description) for c in req.choices] == [("positive", ""), ("negative", "不満")]


def test_decide_meaning_is_optional(client, fake):
    body = {k: v for k, v in BODY.items() if k != "meaning"}
    assert client.post("/api/decide", json=body).status_code == 200
    assert fake.requests[0].meaning == ""


def test_decide_rejects_invalid_input_with_422(client, fake):
    res = client.post("/api/decide", json={**BODY, "choices": [{"label": "only"}]})
    assert res.status_code == 422
    assert "detail" in res.json()
    assert fake.requests == []


def test_decide_rejects_missing_fields(client):
    assert client.post("/api/decide", json={"context": "x"}).status_code == 422


def test_decide_returns_502_when_model_fails(fake, client):
    fake.error = DecisionError("Ollama への問い合わせに失敗した")
    res = client.post("/api/decide", json=BODY)
    assert res.status_code == 502
    assert res.json() == {"detail": "Ollama への問い合わせに失敗した"}


def test_stylesheet_is_served(client):
    res = client.get("/static/style.css")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/css")
