# Hakari

macOS 上の Ollama で動かしている Decision Model（Nimble）を，Web ブラウザから使うための簡易アプリ．
文脈と選択肢を入力すると，モデルが選んだ回答と選択肢ごとの確率を返す．

## 前提

- Ollama が起動しており，`nimble` モデルが入っていること
- [uv](https://docs.astral.sh/uv/)

## 起動

```sh
uv sync
uv run hakari
```

ブラウザで <http://127.0.0.1:8000> を開く．

| 環境変数 | 既定値 | 内容 |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama の URL |
| `HAKARI_MODEL` | `nimble` | 使うモデル名 |
| `HAKARI_HOST` | `127.0.0.1` | 待ち受けアドレス |
| `HAKARI_PORT` | `8000` | 待ち受けポート |

## API

```sh
curl -s localhost:8000/api/decide -H 'content-type: application/json' -d '{
  "context": "料理が冷めていて店員にも無視された．",
  "field": "sentiment",
  "meaning": "レビュー全体の感情",
  "choices": [{"label": "positive"}, {"label": "negative", "description": "不満を述べている"}, {"label": "neutral"}]
}'
```

```json
{"answer": "negative", "probabilities": [{"code": "B", "label": "negative", "probability": 0.987}, ...]}
```

- `meaning`，`description` は省略可．選択肢は 2〜26 個．
- 入力不備は 422，Ollama への問い合わせ失敗は 502．

## 仕組み

Ollama の `/api/chat` に `think: false`，`logprobs: true` を付けて問い合わせ，
選択肢に振ったコード（A, B, ...）の先頭トークン確率を取り出して合計 1 に正規化している．

## テスト

```sh
uv run pytest
```
