import csv
import io

import pytest

from hakari.batch import (
    BatchError,
    Tag,
    TagResult,
    read_csv,
    summarize,
    iter_tag_rows,
    tag_text,
    write_csv,
)
from hakari.decision import ChoiceProbability, DecisionError, DecisionResult


class FakeClient:
    """文脈にタグ名が含まれていれば true と判定する偽のクライアント．"""

    def __init__(self, fail_on=None):
        self.requests = []
        self.fail_on = fail_on

    def decide(self, req):
        self.requests.append(req)
        if self.fail_on and self.fail_on in req.context:
            raise DecisionError("失敗")
        p = 0.9 if req.field in req.context else 0.2
        probs = [ChoiceProbability("A", "true", p), ChoiceProbability("B", "false", 1 - p)]
        probs.sort(key=lambda x: x.probability, reverse=True)
        return DecisionResult(answer=probs[0].label, probabilities=probs)


TAGS = [Tag("価格", "価格への言及"), Tag("接客", "スタッフの対応への言及")]


# --- read_csv ---


def test_read_csv_utf8():
    table = read_csv("id,感想\n1,高い\n2,親切\n".encode())
    assert table.headers == ["id", "感想"]
    assert table.rows == [["1", "高い"], ["2", "親切"]]


def test_read_csv_utf8_with_bom():
    table = read_csv("id,感想\n1,高い\n".encode("utf-8-sig"))
    assert table.headers == ["id", "感想"]


def test_read_csv_cp932():
    table = read_csv("id,感想\n1,高い\n".encode("cp932"))
    assert table.headers == ["id", "感想"]
    assert table.rows == [["1", "高い"]]


def test_read_csv_handles_quoted_newlines():
    table = read_csv('id,感想\n1,"一行目\n二行目"\n'.encode())
    assert table.rows == [["1", "一行目\n二行目"]]


def test_read_csv_pads_short_rows():
    table = read_csv("a,b,c\n1,2\n".encode())
    assert table.rows == [["1", "2", ""]]


def test_read_csv_skips_completely_empty_lines():
    table = read_csv("a,b\n1,2\n\n3,4\n".encode())
    assert table.rows == [["1", "2"], ["3", "4"]]


def test_read_csv_rejects_empty_file():
    with pytest.raises(BatchError):
        read_csv(b"")


def test_read_csv_rejects_undecodable_bytes():
    with pytest.raises(BatchError):
        read_csv(b"\xff\xfe\xfa\x00\x81")


def test_column_index_by_name():
    table = read_csv("id,感想\n1,x\n".encode())
    assert table.column_index("感想") == 1
    with pytest.raises(BatchError):
        table.column_index("無い列")


# --- Tag ---


def test_tag_requires_name():
    with pytest.raises(ValueError):
        Tag(" ", "意味")


# --- tag_text ---


def test_tag_text_asks_true_false_per_tag():
    client = FakeClient()
    results = tag_text(client, "価格が高い", TAGS)

    assert results == [TagResult(True, pytest.approx(0.9)), TagResult(False, pytest.approx(0.2))]
    assert [r.field for r in client.requests] == ["価格", "接客"]
    req = client.requests[0]
    assert req.context == "価格が高い"
    assert req.meaning == "価格への言及"
    assert [c.label for c in req.choices] == ["true", "false"]


def test_tag_text_skips_blank_text():
    client = FakeClient()
    assert tag_text(client, "  \n", TAGS) == [None, None]
    assert client.requests == []


# --- iter_tag_rows ---


def test_iter_tag_rows_yields_result_per_row():
    table = read_csv("id,感想\n1,価格が高い\n2,\n3,接客が良い\n".encode())

    results = list(iter_tag_rows(FakeClient(), table, 1, TAGS))

    assert [[r.value if r else None for r in row] for row in results] == [
        [True, False],
        [None, None],
        [False, True],
    ]


def test_iter_tag_rows_is_lazy():
    table = read_csv("id,感想\n1,a\n2,b\n3,c\n".encode())
    client = FakeClient()

    it = iter_tag_rows(client, table, 1, TAGS)
    next(it)

    assert len(client.requests) == len(TAGS)


def test_iter_tag_rows_keeps_earlier_results_on_error():
    table = read_csv("id,感想\n1,価格\n2,壊れる\n".encode())
    collected = []

    with pytest.raises(DecisionError):
        for row in iter_tag_rows(FakeClient(fail_on="壊れる"), table, 1, TAGS):
            collected.append(row)

    assert len(collected) == 1


# --- write_csv ---


def test_write_csv_appends_tag_and_probability_columns():
    table = read_csv("id,感想\n1,価格が高い\n2,\n".encode())
    results = [[TagResult(True, 0.9), TagResult(False, 0.2)], [None, None]]

    data = write_csv(table, TAGS, results)

    assert data.startswith("﻿".encode())
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    assert rows == [
        ["id", "感想", "価格", "価格_確率", "接客", "接客_確率"],
        ["1", "価格が高い", "TRUE", "0.900", "FALSE", "0.200"],
        ["2", "", "", "", "", ""],
    ]


def test_write_csv_only_includes_processed_rows():
    table = read_csv("id,感想\n1,a\n2,b\n".encode())
    data = write_csv(table, TAGS, [[TagResult(True, 0.9), TagResult(True, 0.8)]])
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    assert len(rows) == 2


# --- summarize ---


def test_summarize_counts_true_per_tag():
    results = [
        [TagResult(True, 0.9), TagResult(False, 0.2)],
        [None, None],
        [TagResult(True, 0.8), TagResult(True, 0.6)],
    ]
    assert summarize(TAGS, results) == {"価格": 2, "接客": 1}
