"""
tests/test_cards.py - steps 10 and 11: the model-card schema and the 'which model for which job' report.
"""

import json
from datetime import date, timedelta

import pytest

import model_cards_schema
import zoo_report
from model_cards_schema import CARDS_FILE, check_cards, is_stale, load_cards
from zoo_report import JOBS, report_rows


def write_cards(tmp_path, cards) -> str:
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(cards), encoding="utf-8")
    return path


def shipped_cards() -> list[dict]:
    return json.loads(CARDS_FILE.read_text(encoding="utf-8"))


def test_the_shipped_cards_pass_the_schema_check():
    cards, problems = check_cards()
    assert problems == []
    assert len(cards) >= 4            # the four models of this episode (more after the challenge)


def test_every_card_has_a_source_and_a_checked_date():
    for card in load_cards():
        assert card.source.startswith("https://")
        assert card.checked <= date.today()


def test_a_foundation_model_does_not_have_to_generate():
    cards = load_cards()
    assert any(not card.generates for card in cards)              # MiniLM and CLIP write nothing
    assert any(card.generates for card in cards)


def test_a_wrong_weights_value_is_explained(tmp_path):
    cards = shipped_cards()
    cards[1]["weights"] = "free"
    _, problems = check_cards(write_cards(tmp_path, cards))
    assert problems == ["Qwen2.5-0.5B-Instruct: weights: Input should be 'open' or 'closed'"]


def test_closed_weights_cannot_have_a_download_size(tmp_path):
    cards = shipped_cards()
    cards[2]["download_mb"] = 500
    _, problems = check_cards(write_cards(tmp_path, cards))
    assert "closed weights cannot be downloaded" in problems[0]


def test_a_missing_source_is_reported(tmp_path):
    cards = shipped_cards()
    del cards[3]["source"]
    _, problems = check_cards(write_cards(tmp_path, cards))
    assert problems == ["CLIP ViT-B/32: source: Field required"]


def test_a_checked_date_in_the_future_is_rejected(tmp_path):
    cards = shipped_cards()
    cards[0]["checked"] = str(date.today() + timedelta(days=1))
    _, problems = check_cards(write_cards(tmp_path, cards))
    assert "is in the future" in problems[0]


def test_the_same_model_twice_is_reported(tmp_path):
    cards = shipped_cards()
    _, problems = check_cards(write_cards(tmp_path, cards + [cards[0]]))
    assert "appears twice" in problems[0]


def test_broken_json_gives_a_clear_message(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text('[{"name": "oops",}]', encoding="utf-8")
    _, problems = check_cards(path)
    assert "is not valid JSON" in problems[0]


def test_old_cards_are_marked_stale():
    card = load_cards()[0]
    assert not is_stale(card, today=card.checked)
    assert is_stale(card, today=card.checked + timedelta(days=200))


def test_the_report_has_one_row_per_job():
    rows = report_rows(load_cards())
    assert len(rows) == len(JOBS) >= 5
    clip_row = next(row for row in rows if row[1] == "CLIP ViT-B/32")
    assert clip_row[3] == "no"                                    # CLIP writes nothing
    assert clip_row[5] == "research only"                         # read the card before you ship it


def test_a_job_without_a_card_is_reported():
    with pytest.raises(KeyError, match="no card for someone/no-such-model"):
        report_rows(load_cards(), jobs=[("Read minds", "someone/no-such-model", "minds.py")])


def test_the_scripts_print_their_summaries(capsys):
    cards = len(load_cards())
    assert model_cards_schema.main([]) == 0
    assert zoo_report.main() == 0
    out = capsys.readouterr().out
    assert f"{cards} cards in model_cards.json, all passed the schema check." in out
    assert "Which model for which job" in out
    assert f"{len(JOBS)} jobs, {cards} model cards." in out
