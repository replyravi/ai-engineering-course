"""
tests/test_rules.py - approach A, the hand-written rulebook.

These tests also pin down the rulebook's KNOWN weaknesses (sarcasm and "not"),
so the episode's story stays true if someone edits the rules.
"""

import pytest

from a_rules import RULEBOOK, accuracy, rules_label


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("The paneer roll was tasty.", "praise"),
        ("The sambar was cold.", "complaint"),
        ("Could you open a second counter?", "suggestion"),
        ("THANK YOU for the clean tables!", "praise"),         # capitals and punctuation do not matter
    ],
)
def test_simple_sentences(sentence, expected):
    assert rules_label(sentence) == expected


def test_rules_are_read_top_to_bottom():
    # "love" is a praise word, but the suggestion rules come first in the rulebook.
    assert list(RULEBOOK) == ["suggestion", "praise", "complaint"]
    assert rules_label("Love the food, please add more seats.") == "suggestion"


def test_no_matching_word_means_unknown():
    assert rules_label("The paratha arrived after lunch.") == "unknown"


def test_whole_words_only():
    # "cold" must not match inside "scolded"
    assert rules_label("The manager scolded nobody.") == "unknown"


def test_known_miss_sarcasm():
    # The cold-open sentence: "great" fools the rules. A person reads it as a complaint.
    assert rules_label("Great job, the dosa was cold again.") == "praise"


def test_known_miss_negation():
    # The rules see "fresh" and miss the "not" in front of it.
    assert rules_label("The chutney was not fresh today.") == "praise"


def test_accuracy():
    assert accuracy(["praise", "complaint"], ["praise", "suggestion"]) == 0.5
    assert accuracy(["praise"], ["praise"]) == 1.0
