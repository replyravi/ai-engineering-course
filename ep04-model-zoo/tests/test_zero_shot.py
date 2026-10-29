"""
tests/test_zero_shot.py - step 7: zero-shot labels with structured output, using a FAKE model.

structured_reply(...) scripts exactly what with_structured_output(ReviewLabels) returns,
so we can test our label parser (match_labels) without a key or the internet.
"""

import pytest
from pydantic import ValidationError

import zero_shot
from testing.fakes import fake_llm, structured_reply
from zero_shot import ReviewLabel, ReviewLabels, label_reviews, match_labels
from zoo_fixtures import sample_reviews

REVIEWS = ["Great tea.", "Cold coffee, rude staff.", "Nice bun, dirty plate."]
KEY_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL"]


def labels(*pairs) -> ReviewLabels:
    """labels((1, "positive"), (2, "negative")) -> the form the model fills in."""
    return ReviewLabels(labels=[ReviewLabel(number=n, label=label) for n, label in pairs])


def test_labels_come_back_in_review_order():
    llm = fake_llm(structured_reply(labels((3, "mixed"), (1, "positive"), (2, "negative"))))
    assert label_reviews(REVIEWS, llm) == ["positive", "negative", "mixed"]


def test_all_reviews_go_in_one_request():
    llm = fake_llm(structured_reply(labels((1, "positive"), (2, "negative"), (3, "mixed"))))
    label_reviews(REVIEWS, llm)
    assert len(llm.received) == 1                                  # one request, not three
    sent = llm.received[0][-1].content
    assert "1. Great tea." in sent and "3. Nice bun, dirty plate." in sent


def test_a_missing_label_is_reported():
    with pytest.raises(ValueError, match=r"no label for review\(s\) \[2\]"):
        match_labels(REVIEWS, labels((1, "positive"), (3, "mixed")))


def test_two_labels_for_one_review_are_reported():
    with pytest.raises(ValueError, match="review 1 got two labels"):
        match_labels(REVIEWS, labels((1, "positive"), (1, "negative"), (2, "negative"), (3, "mixed")))


def test_a_review_number_that_does_not_exist_is_reported():
    with pytest.raises(ValueError, match=r"do not exist: \[4\]"):
        match_labels(REVIEWS, labels((1, "positive"), (2, "negative"), (3, "mixed"), (4, "mixed")))


def test_only_three_labels_are_allowed():
    with pytest.raises(ValidationError):
        ReviewLabel(number=1, label="angry")


def test_main_prints_ten_labels_with_a_fake_model(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(zero_shot, "LABELS_FILE", tmp_path / "labels.json")
    monkeypatch.setattr(zero_shot, "OUTPUTS", tmp_path)
    monkeypatch.setattr(zero_shot, "load_reviews", sample_reviews)   # the ten sample reviews, not yours
    scripted = ["positive", "negative", "negative", "mixed", "positive",
                "positive", "negative", "positive", "mixed", "negative"]
    llm = fake_llm(structured_reply(labels(*enumerate(scripted, start=1))))
    assert zero_shot.main(llm=llm) == 0
    out = capsys.readouterr().out
    assert "Labelled examples given to the model: 0" in out
    assert "Counts: positive 4, negative 4, mixed 2" in out
    assert (tmp_path / "labels.json").exists()


def test_main_without_a_key_prints_friendly_help(capsys, monkeypatch):
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert zero_shot.main() == 1
    out = capsys.readouterr().out
    assert "No API key found for LLM_PROVIDER=gemini" in out
    assert "https://aistudio.google.com/apikey" in out
