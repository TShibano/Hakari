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

## CSV で一括タグ付け

<http://127.0.0.1:8000/batch> から，アンケートの自由記述に複数のタグを付けられる．

1. CSV をアップロードする（UTF-8 / Shift_JIS，1行目は見出し）
2. 自由記述の列を選ぶ
3. タグ名と意味（何を含むかを具体的に）を並べる
4. 開始すると，回答 × タグごとに true / false を判定する．進捗とタグごとの該当件数が表示される
5. 完了後，元の列に「タグ名」（TRUE/FALSE）と「タグ名_確率」の列を足した CSV をダウンロードできる

- 問い合わせ回数は「回答数 × タグ数」．1回あたり約0.4秒なので，1000件 × 10タグで1時間前後かかる．
- 途中で中断・失敗しても，判定済みの行はダウンロードできる．
- ジョブはメモリ上にだけ保持するので，サーバーを止めると消える．
- 確率が 0.5 前後の判定は迷いが大きいので，人が見直すとよい．
- 試すためのサンプルデータ（セミナー受講アンケート）と正解ラベルを [samples/](samples/README.md) に置いている．

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

一括タグ付けの API:

| メソッド | パス | 内容 |
| --- | --- | --- |
| POST | `/api/csv/columns` | CSV（`file`）の見出しと行数を返す |
| POST | `/api/batch` | `file`，`column`（列名），`tags`（`[{"name", "meaning"}]` の JSON）でジョブを開始し，202 を返す |
| GET | `/api/batch/{id}` | 進捗（`status`，`done`，`total`，`error`，`summary`） |
| POST | `/api/batch/{id}/cancel` | 中断する |
| GET | `/api/batch/{id}/result.csv` | 結果 CSV（処理中は 409） |

## 仕組み

Ollama の `/api/chat` に `think: false`，`logprobs: true` を付けて問い合わせ，
選択肢に振ったコード（A, B, ...）の先頭トークン確率を取り出して合計 1 に正規化している．

## テスト

```sh
uv run pytest
```
