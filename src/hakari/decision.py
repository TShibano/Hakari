"""Ollama 上の Decision Model に分類を問い合わせ，選択肢ごとの確率を求める．"""

import math
import string
from dataclasses import dataclass

import httpx

MAX_CHOICES = len(string.ascii_uppercase)


class DecisionError(Exception):
    """モデルへの問い合わせ，または応答の解釈に失敗した．"""


@dataclass(frozen=True)
class Choice:
    label: str
    description: str = ""


@dataclass(frozen=True)
class DecisionRequest:
    context: str
    field: str
    meaning: str
    choices: list[Choice]

    def __post_init__(self):
        if not self.context.strip():
            raise ValueError("context は必須")
        if not self.field.strip():
            raise ValueError("field は必須")
        if len(self.choices) < 2:
            raise ValueError("選択肢は2つ以上必要")
        if any(not c.label.strip() for c in self.choices):
            raise ValueError("選択肢のラベルが空")
        choice_codes(len(self.choices))


@dataclass(frozen=True)
class ChoiceProbability:
    code: str
    label: str
    probability: float


@dataclass(frozen=True)
class DecisionResult:
    answer: str
    probabilities: list[ChoiceProbability]


def choice_codes(n: int) -> list[str]:
    if n > MAX_CHOICES:
        raise ValueError(f"選択肢は最大 {MAX_CHOICES} 個まで")
    return list(string.ascii_uppercase[:n])


def build_prompt(req: DecisionRequest) -> str:
    # 分類の指示自体は Modelfile の SYSTEM にあるため，ここではスキーマと文脈だけを渡す．
    lines = ["Schema:", f"field: {req.field}"]
    if req.meaning.strip():
        lines.append(f"meaning: {req.meaning}")
    lines.append("choices:")
    for code, choice in zip(choice_codes(len(req.choices)), req.choices):
        line = f"{code}: {choice.label}"
        if choice.description.strip():
            line += f" - {choice.description}"
        lines.append(line)
    lines += ["", "Context:", req.context, "", f"Field: {req.field}"]
    return "\n".join(lines)


def to_probabilities(top_logprobs: list[dict], codes: list[str]) -> dict[str, float]:
    """先頭トークンの候補から選択肢コードだけを取り出し，合計1に正規化する．"""
    mass = dict.fromkeys(codes, 0.0)
    for entry in top_logprobs:
        token = entry["token"].strip()
        if token in mass:
            mass[token] += math.exp(entry["logprob"])
    total = sum(mass.values())
    if total == 0:
        raise DecisionError("モデルの応答に選択肢コードが含まれていない")
    return {code: m / total for code, m in mass.items()}


class DecisionClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nimble",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def decide(self, req: DecisionRequest) -> DecisionResult:
        codes = choice_codes(len(req.choices))
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_prompt(req)}],
            "stream": False,
            "think": False,
            "logprobs": True,
            # 選択肢以外のトークン（小文字や空白付き）にも枠を取られるので余裕を持たせる．
            "top_logprobs": min(20, len(codes) + 5),
            "options": {"temperature": 0},
        }
        try:
            res = self._http.post("/api/chat", json=payload)
            res.raise_for_status()
        except httpx.HTTPError as e:
            raise DecisionError(f"Ollama への問い合わせに失敗した: {e}") from e

        try:
            top = res.json()["logprobs"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise DecisionError("Ollama の応答に logprobs が含まれていない") from e

        probs = to_probabilities(top, codes)
        labels = dict(zip(codes, (c.label for c in req.choices)))
        ranked = sorted(
            (ChoiceProbability(code, labels[code], p) for code, p in probs.items()),
            key=lambda cp: cp.probability,
            reverse=True,
        )
        return DecisionResult(answer=ranked[0].label, probabilities=ranked)
