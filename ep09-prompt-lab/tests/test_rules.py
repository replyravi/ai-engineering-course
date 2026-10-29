"""
tests/test_rules.py - check_rules() on fixed strings (Step 10).

Every rule is tested on messages we wrote by hand, so we know the right answer.
"""

import pytest

import rules
from common import load_notes
from models import Settlement
from rules import SAMPLE_MESSAGES, check_rules, emojis, fallback_message, sentences
from settle import settle

AMOUNTS = [740, 230]       # note 1: Kabir owes Meera 740, Arjun owes Meera 230


def passed(message, amounts=AMOUNTS):
    return {r.rule: r.passed for r in check_rules(message, amounts)}


def test_the_message_that_looks_fine_breaks_four_rules():
    result = passed(SAMPLE_MESSAGES["looks fine at a glance"])
    assert result == {"at most 3 sentences": False, "exactly one emoji": False, "ends with a question": False,
                      "mentions every amount": True, "no promises": False}


def test_the_good_sample_passes_every_rule():
    assert all(passed(SAMPLE_MESSAGES["follows every rule"]).values())


@pytest.mark.parametrize("text, count", [
    ("Pay by Sunday? 🙂", 1),
    ("Thumbs up 👍🏽", 1),             # skin tone glued to the thumb
    ("The family 👨‍👩‍👧 came too", 1),  # three people joined into one emoji
    ("Go India 🇮🇳", 1),               # a flag is two letters, one emoji
    ("Kabir owes Meera ₹740.", 0),     # the rupee sign is not an emoji
    ("Two 😊🌄", 2),
])
def test_emoji_counting(text, count):
    assert len(emojis(text)) == count


@pytest.mark.parametrize("text, count", [
    ("One. Two! Three?", 3),
    ("Kabir owes Rs. 740 to Meera. Done?", 2),     # "Rs." is not the end of a sentence
    ("Arjun owes 450.50 rupees. Ok?", 2),          # a decimal point is not either
    ("No full stop at the end", 1),
])
def test_sentence_counting(text, count):
    assert len(sentences(text)) == count


def test_a_question_mark_before_the_emoji_still_ends_with_a_question():
    assert passed("Kabir owes Meera 740 and Arjun owes Meera 230. Pay by Sunday? 🙂")["ends with a question"]


def test_amounts_with_commas_are_found():
    assert passed("Kabir owes Ishita 1,520 rupees. Ok? 🙂", [1520])["mentions every amount"]


def test_a_missing_amount_is_named():
    result = check_rules("Kabir owes Meera 740. Ok? 🙂", AMOUNTS)
    amount_rule = next(r for r in result if r.rule == "mentions every amount")
    assert not amount_rule.passed and amount_rule.detail == "missing: 230"


@pytest.mark.parametrize("text, ok", [
    ("Meera will pay you back. Ok? 🙂", False),
    ("You get a refund. Ok? 🙂", False),
    ("We guarantee it. Ok? 🙂", False),
    ("Kabir, please pay Meera 740. Ok? 🙂", True),
])
def test_promises_are_caught(text, ok):
    assert passed(text)["no promises"] is ok


@pytest.mark.parametrize("note", load_notes(), ids=lambda n: n.slug)
def test_the_fallback_message_always_passes(note):
    settlement = settle(note.expenses)
    message = fallback_message(note.trip, settlement)
    assert all(r.passed for r in check_rules(message, [t.amount for t in settlement.transfers]))


def test_the_fallback_message_when_nobody_owes():
    message = fallback_message("Day out", Settlement(transfers=[]))
    assert all(r.passed for r in check_rules(message, []))


def test_the_command_line_checks_both_samples(capsys):
    assert rules.main([]) == 0
    output = capsys.readouterr().out
    assert "1 of 5 rules pass" in output and "5 of 5 rules pass" in output
