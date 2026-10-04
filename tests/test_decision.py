import json
import math

import httpx
import pytest

from hakari.decision import (
    Choice,
    DecisionClient,
    DecisionError,
    DecisionRequest,
    build_prompt,
    choice_codes,
    to_probabilities,
)


def make_request(**overrides):
    params = dict(
        context="料理が冷めていて店員にも無視された．",
        field="sentiment",
        meaning="レビュー全体の感情",
        choices=[Choice("positive"), Choice("negative", "不満を述べている"), Choice("neutral")],
    )
    params.update(overrides)
    return DecisionRequest(**params)


def lp(token, logprob):
    return {"token": token, "logprob": logprob}


# --- choice_codes ---


def test_choice_codes_are_sequential_letters():
    assert choice_codes(3) == ["A", "B", "C"]


def test_choice_codes_rejects_more_than_26():
    with pytest.raises(ValueError):
        choice_codes(27)


# --- DecisionRequest のバリデーション ---


def test_request_requires_at_least_two_choices():
    with pytest.raises(ValueError):
        make_request(choices=[Choice("only")])


def test_request_requires_context():
    with pytest.raises(ValueError):
        make_request(context="   ")


def test_request_rejects_blank_choice_label():
    with pytest.raises(ValueError):
        make_request(choices=[Choice("a"), Choice(" ")])


# --- build_prompt ---


def test_prompt_contains_schema_and_context():
    prompt = build_prompt(make_request())
    assert prompt == (
        "Schema:\n"
        "field: sentiment\n"
        "meaning: レビュー全体の感情\n"
        "choices:\n"
        "A: positive\n"
        "B: negative - 不満を述べている\n"
        "C: neutral\n"
        "\n"
        "Context:\n"
        "料理が冷めていて店員にも無視された．\n"
        "\n"
        "Field: sentiment"
    )


def test_prompt_omits_meaning_when_empty():
    prompt = build_prompt(make_request(meaning=""))
    assert "meaning:" not in prompt


# --- to_probabilities ---


def test_probabilities_are_normalized_over_choice_codes():
    top = [lp("B", math.log(0.6)), lp("C", math.log(0.2)), lp("A", math.log(0.1)), lp("D", math.log(0.1))]
    probs = to_probabilities(top, ["A", "B", "C"])
    assert probs.keys() == {"A", "B", "C"}
    assert probs["B"] == pytest.approx(0.6 / 0.9)
    assert probs["C"] == pytest.approx(0.2 / 0.9)
    assert probs["A"] == pytest.approx(0.1 / 0.9)


def test_probabilities_missing_code_is_zero():
    top = [lp("A", math.log(0.9)), lp("B", math.log(0.05))]
    probs = to_probabilities(top, ["A", "B", "C"])
    assert probs["C"] == 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


def test_probabilities_ignore_surrounding_whitespace_in_token():
    top = [lp(" A", math.log(0.5)), lp("B", math.log(0.5))]
    probs = to_probabilities(top, ["A", "B"])
    assert probs["A"] == pytest.approx(0.5)


def test_probabilities_merge_duplicate_tokens():
    top = [lp("A", math.log(0.4)), lp(" A", math.log(0.4)), lp("B", math.log(0.2))]
    probs = to_probabilities(top, ["A", "B"])
    assert probs["A"] == pytest.approx(0.8)


def test_probabilities_raise_when_no_code_found():
    with pytest.raises(DecisionError):
        to_probabilities([lp("The", -0.1)], ["A", "B"])


# --- DecisionClient ---


def ollama_response(top):
    return {
        "model": "nimble",
        "message": {"role": "assistant", "content": top[0]["token"]},
        "done": True,
        "logprobs": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "top_logprobs": top}],
    }


def make_client(handler):
    transport = httpx.MockTransport(handler)
    return DecisionClient(base_url="http://ollama.test", model="nimble", transport=transport)


def test_decide_sends_expected_payload():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=ollama_response([lp("B", -0.01), lp("C", -4.6), lp("A", -6.7)]))

    make_client(handler).decide(make_request())

    assert captured["url"] == "http://ollama.test/api/chat"
    body = captured["body"]
    assert body["model"] == "nimble"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["logprobs"] is True
    assert body["top_logprobs"] >= 3
    assert body["options"]["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": build_prompt(make_request())}]


def test_decide_returns_answer_and_sorted_probabilities():
    def handler(request):
        return httpx.Response(200, json=ollama_response([lp("B", -0.01), lp("C", -4.6), lp("A", -6.7)]))

    result = make_client(handler).decide(make_request())

    assert result.answer == "negative"
    assert [p.label for p in result.probabilities] == ["negative", "neutral", "positive"]
    assert [p.code for p in result.probabilities] == ["B", "C", "A"]
    assert sum(p.probability for p in result.probabilities) == pytest.approx(1.0)
    assert result.probabilities[0].probability > 0.98


def test_decide_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(DecisionError):
        make_client(handler).decide(make_request())


def test_decide_raises_when_ollama_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(DecisionError):
        make_client(handler).decide(make_request())


def test_decide_raises_when_logprobs_missing():
    def handler(request):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "B"}, "done": True})

    with pytest.raises(DecisionError):
        make_client(handler).decide(make_request())
