"""
tests/test_ladder.py - the ladder report (Step 12) with a SCRIPTED model.

Every model call gets the next scripted reply, in the order ladder.py asks:
question 1 rung 1, question 2 rung 1, question 2 rung 2, and so on.
We check OUR logic: the grading, when to climb, when to stop, and the report format.
"""

import ladder
from common import REFUSAL
from helpdesk_fixtures import KeywordEmbeddings, fixed_today, no_keys  # noqa: F401
from testing.fakes import fake_llm, tool_call

SCRIPT = [
    # 1 general: rung 1 answers
    "Namaste! A meal-box subscription brings home-style meals on a schedule. You pay once for many meals.",
    # 2 fact: rung 1 does not know, rung 2 reads the handbook
    "Namaste! I don't know.",
    "Namaste! Skip today's lunch before 9:30 am [handbook.md#4].",
    # 3 days: rung 1 and rung 2 do not know, rung 3 calls the tool
    "Namaste! I don't know.",
    f"Namaste! {REFUSAL}",
    tool_call("days_until_holiday", {}),
    "Namaste! The next kitchen holiday, Kitchen deep-clean day, is in 25 days.",
    # 4 tone: rung 1 answers in the house style
    "Namaste! Thank you, we will tell the kitchen team.",
    # 5 not covered: every rung says it does not know
    "Namaste! I don't know.",
    f"Namaste! {REFUSAL}",
    "Namaste! I don't know.",
]


def test_grading_rules():
    general, fact, days, tone, not_covered = ladder.QUESTIONS
    assert ladder.grade(general, "A plan is ...", 25) == "solved"
    assert ladder.grade(fact, "Before 9:30 am.", 25) == "solved"
    assert ladder.grade(fact, "Before 10 am.", 25) == "wrong"            # fluent, but not our rule
    assert ladder.grade(days, "It is 25 days away.", 25) == "solved"
    assert ladder.grade(days, "It is 125 days away.", 25) == "wrong"     # whole numbers only
    assert ladder.grade(tone, "Namaste! Thanks!", 25) == "solved"
    assert ladder.grade(not_covered, "I don't know.", 25) == "I don't know"
    assert ladder.grade(not_covered, "Yes, 10 percent off.", 25) == "answered?"


def test_answer_key_for_the_holiday_question(fixed_today):
    from datetime import date
    assert ladder.expected_days(date(2026, 10, 23)) == 25


def test_the_ladder_stops_at_the_cheapest_rung_that_works(fixed_today, capsys):
    llm = fake_llm(*SCRIPT)
    assert ladder.main([], llm=llm, embeddings=KeywordEmbeddings()) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()

    def row(number):
        return next(line for line in lines if line.startswith(f"{number}  "))

    assert row(1).endswith("Prompt")
    assert "I don't know" in row(2) and row(2).endswith("Context")
    assert "solved (tool)" in row(3) and row(3).endswith("Tools")
    assert row(4).endswith("Prompt (no fine-tune needed)")
    assert row(5).endswith("none: says I don't know (correct)")
    assert "Model calls: 11" in output
    assert "  2. must mention 9:30" in output and "  3. must say 25" in output   # the answer key
    assert len(llm.received) == len(SCRIPT)                                     # every reply was used


def test_an_answer_to_a_question_the_handbook_cannot_answer_is_flagged(fixed_today):
    item = ladder.QUESTIONS[4]
    llm = fake_llm("Namaste! Students get 10 percent off.")             # invented!
    from rung3_context import load_paragraphs
    row = ladder.climb(item, llm, KeywordEmbeddings(), load_paragraphs(), 25)
    assert row["cells"] == ["answered?", "-", "-"]
    assert row["winner"] == "check it: Prompt answered without a source"


def test_ladder_without_a_key_prints_the_answer_key_then_help(no_keys, fixed_today, capsys):
    assert ladder.main([]) == 1
    output = capsys.readouterr().out
    assert output.startswith("Answer key (checked by our code, not by a model):")
    assert "  3. must say 25" in output
    assert "https://aistudio.google.com/apikey" in output


def test_the_footer_tells_the_truth_about_the_tone_question(capsys):
    from datetime import date
    rows = [{"question": item["question"], "cells": ["-", "-", "-"], "winner": "Prompt", "answer": "x", "calls": 1}
            for item in ladder.QUESTIONS]
    rows[3]["winner"] = "Context"                                      # the prompt missed the house tone
    ladder.print_report(rows, "fake-model", date(2026, 10, 23))
    assert "Question 4 missed the house tone" in capsys.readouterr().out
