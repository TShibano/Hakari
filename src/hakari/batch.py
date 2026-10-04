"""アンケート自由記述を，複数タグの true / false 判定で一括タグ付けする．"""

import csv
import io
from collections.abc import Iterator
from dataclasses import dataclass

from hakari.decision import Choice, DecisionClient, DecisionRequest

# Excel で保存された日本語 CSV は Shift_JIS(cp932) のことが多いため，UTF-8 で読めなければ試す．
ENCODINGS = ("utf-8-sig", "cp932")
TRUE_FALSE = [Choice("true", "該当する"), Choice("false", "該当しない")]


class BatchError(Exception):
    """CSV やタグ指定が不正で，一括処理を始められない．"""


@dataclass(frozen=True)
class Tag:
    name: str
    meaning: str = ""

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("タグ名は必須")


@dataclass(frozen=True)
class TagResult:
    value: bool
    probability: float  # true である確率


@dataclass(frozen=True)
class Table:
    headers: list[str]
    rows: list[list[str]]

    def column_index(self, name: str) -> int:
        try:
            return self.headers.index(name)
        except ValueError:
            raise BatchError(f"列「{name}」が見つからない") from None


# 1行分の結果．空欄の回答は判定しないので None になる．
RowResult = list[TagResult | None]


def read_csv(data: bytes) -> Table:
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise BatchError("CSV の文字コードを判別できない（UTF-8 か Shift_JIS で保存してください）")

    records = [r for r in csv.reader(io.StringIO(text, newline="")) if any(cell.strip() for cell in r)]
    if not records:
        raise BatchError("CSV が空")
    headers, *rows = records
    width = len(headers)
    return Table(headers=headers, rows=[r + [""] * (width - len(r)) for r in rows])


def tag_text(client: DecisionClient, text: str, tags: list[Tag]) -> RowResult:
    if not text.strip():
        return [None] * len(tags)
    results: RowResult = []
    for tag in tags:
        res = client.decide(DecisionRequest(context=text, field=tag.name, meaning=tag.meaning, choices=TRUE_FALSE))
        p_true = next(p.probability for p in res.probabilities if p.label == "true")
        results.append(TagResult(value=res.answer == "true", probability=p_true))
    return results


def iter_tag_rows(client: DecisionClient, table: Table, column: int, tags: list[Tag]) -> Iterator[RowResult]:
    """1行ずつ判定して返す．途中で失敗・中断しても，それまでの結果を呼び出し側が保持できるようにするため．"""
    for row in table.rows:
        yield tag_text(client, row[column], tags)


def write_csv(table: Table, tags: list[Tag], results: list[RowResult]) -> bytes:
    """元の列にタグごとの判定と確率の列を足した CSV を返す．Excel で開けるよう BOM 付き UTF-8 にする．"""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(table.headers + [col for t in tags for col in (t.name, f"{t.name}_確率")])
    for row, result in zip(table.rows, results):
        cells = []
        for r in result:
            cells += ["", ""] if r is None else ["TRUE" if r.value else "FALSE", f"{r.probability:.3f}"]
        writer.writerow(row + cells)
    return buf.getvalue().encode("utf-8-sig")


def summarize(tags: list[Tag], results: list[RowResult]) -> dict[str, int]:
    return {t.name: sum(1 for row in results if row[i] is not None and row[i].value) for i, t in enumerate(tags)}
