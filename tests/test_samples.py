"""samples/ 配下のサンプルデータ同士の整合性を確認する．"""

import json
from pathlib import Path

import pytest

from hakari.batch import Tag, read_csv

SAMPLES = Path(__file__).parent.parent / "samples"
SURVEY_HEADERS = ["回答者ID", "年代", "職種", "受講形式", "満足度", "感想"]


@pytest.fixture(scope="module")
def survey():
    return read_csv((SAMPLES / "seminar_survey.csv").read_bytes())


@pytest.fixture(scope="module")
def tags():
    return [Tag(**t) for t in json.loads((SAMPLES / "seminar_tags.json").read_text(encoding="utf-8"))]


@pytest.fixture(scope="module")
def labels():
    return read_csv((SAMPLES / "seminar_labels.csv").read_bytes())


def test_survey_has_expected_columns_and_25_respondents(survey):
    assert survey.headers == SURVEY_HEADERS
    assert len(survey.rows) == 25
    ids = [r[0] for r in survey.rows]
    assert len(set(ids)) == 25


def test_survey_attribute_values_are_valid(survey):
    for row in survey.rows:
        assert row[3] in ("会場", "オンライン")
        assert row[4] in ("1", "2", "3", "4", "5")


def test_survey_includes_edge_cases(survey):
    comments = [r[5] for r in survey.rows]
    assert any(c == "" for c in comments), "空欄の回答を含める"
    assert any("特になし" in c for c in comments), "中身の無い回答を含める"
    assert any("\n" in c for c in comments), "セル内改行を含める"


def test_tags_have_unique_names_and_meanings(tags):
    assert len(tags) >= 4
    assert len({t.name for t in tags}) == len(tags)
    assert all(t.meaning.strip() for t in tags)


def test_labels_cover_every_respondent_and_tag(survey, tags, labels):
    assert labels.headers == ["回答者ID"] + [t.name for t in tags]
    assert [r[0] for r in labels.rows] == [r[0] for r in survey.rows]
    for row in labels.rows:
        assert set(row[1:]) <= {"TRUE", "FALSE"}


def test_blank_comments_have_no_tags(survey, labels):
    for s, l in zip(survey.rows, labels.rows):
        if not s[5].strip():
            assert set(l[1:]) == {"FALSE"}


def test_every_tag_has_positive_and_negative_examples(tags, labels):
    for i, tag in enumerate(tags, start=1):
        values = {row[i] for row in labels.rows}
        assert values == {"TRUE", "FALSE"}, f"タグ「{tag.name}」に正例と負例の両方が必要"
