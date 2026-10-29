"""
test_tone.py - Step 3: test the tone tool with KNOWN sentences, like a scale with a known weight.

The word-list fallback and the report logic are tested with exact, hand-checked numbers.
The real local model is tested only if it is already on this computer (run python -m tools.tone once).
"""

import json

import pytest

from notebook import CaseFile
from office_helpers import FixedSentiment
from tools.tone import (KNOWN_SENTENCES, CheckToneTool, LocalSentimentModel, WordListSentiment,
                        model_is_downloaded, sentences, tone_report)


def test_word_list_counts_warm_and_cold_words():
    words = WordListSentiment()
    # "congratulations" is warm, nothing is cold: warm 1, cold 0, plus 1 neutral weight -> 1/2 each
    assert words.scores("Congratulations on the new outlet") == {"negative": 0.0, "neutral": 0.5, "positive": 0.5}
    # "losing" and "costing" are cold: 2 / 3
    assert words.scores(KNOWN_SENTENCES[2])["negative"] == 0.6667
    # no warm or cold word at all: fully neutral, the honest answer
    assert words.scores("The order is ready for pickup.") == {"negative": 0.0, "neutral": 1.0, "positive": 0.0}


def test_a_plain_email_passes():
    report = tone_report("Dear Nandini,\nThe free trial runs for 30 days.", FixedSentiment(negative=0.1))
    assert report["verdict"] == "ok"
    assert report["reasons"] == []
    assert report["label"] == "neutral"


def test_pushy_phrases_and_exclamation_marks_are_reasons():
    report = tone_report("Act now! Last chance! Hurry!", FixedSentiment())
    assert report["verdict"] == "check"
    assert report["pushy_phrases"] == ["act now", "last chance", "hurry"]
    assert report["exclamation_marks"] == 3
    assert "3 exclamation marks, keep it to 1 or none" in report["reasons"]


def test_one_cold_sentence_is_named_so_the_reviewer_can_fix_it():
    cold = "Your staff keep losing pre-orders."
    report = tone_report(f"Dear Nandini,\n{cold} We can help.", FixedSentiment(per_sentence={cold: 0.91}))
    assert report["verdict"] == "check"
    assert report["cold_sentences"] == [{"sentence": 2, "negative": 0.91, "text": cold}]


def test_long_emails_are_flagged():
    report = tone_report("word " * 151, FixedSentiment())
    assert report["words"] == 151
    assert report["reasons"] == ["151 words, keep it to 150 or fewer"]


def test_sentences_splits_lines_and_full_stops():
    assert sentences("Dear Arun,\nHello there. How are you?\n\nBest wishes,\nNeha") == [
        "Dear Arun,", "Hello there.", "How are you?", "Best wishes,", "Neha"]


def test_the_tool_returns_json_and_notes_the_call():
    case = CaseFile()
    tool = CheckToneTool(sentiment=WordListSentiment(), case=case)
    result = json.loads(tool.run(text="Thank you, glad to help."))
    assert result["verdict"] == "ok" and result["engine"] == "word list (fallback)"
    assert case.run["tool_calls"] == ["check_tone"]


@pytest.mark.skipif(not model_is_downloaded(), reason="run python -m tools.tone once with internet")
def test_the_real_model_on_the_known_sentences():
    model = LocalSentimentModel(download=False)
    labels = [tone_report(s, model)["label"] for s in KNOWN_SENTENCES]
    verdicts = [tone_report(s, model)["verdict"] for s in KNOWN_SENTENCES]
    assert labels == ["positive", "neutral", "negative", "neutral"]
    assert verdicts == ["ok", "check", "check", "ok"]
    assert model.scores(KNOWN_SENTENCES[0])["positive"] == pytest.approx(0.9845, abs=0.0005)
    assert model.scores(KNOWN_SENTENCES[2])["negative"] == pytest.approx(0.9129, abs=0.0005)
