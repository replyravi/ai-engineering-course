"""
tests/test_scoreboard.py - the scoreboard maths with scripted fakes (Steps 13 and 14).

The fake answers are written by us, so we know the right score before the code counts it.
"""

import csv

from langchain_core.runnables import RunnableLambda

import scoreboard
from common import load_notes
from lab_fixtures import key_reply, lines_reply, no_keys  # noqa: F401
from scoreboard import Row, choose, score_technique
from testing.fakes import fake_llm

NOTES = load_notes()


def test_score_and_variation_maths():
    row = Row("x", "x_v1", cells={1: [True, True], 2: [True, False], 3: [None, False], 4: [False, False]})
    assert row.score() == (3, 8)
    assert row.varied() == [2]              # only note 2 was right in one run and wrong in another


def test_a_technique_is_scored_on_every_note_and_saved(tmp_path):
    first_three = NOTES[:3]
    replies = [lines_reply(first_three[0]), "Arjun owes Meera 999", lines_reply(first_three[2])]
    row = score_technique("rtc", "balance_rtc_v1", first_three, fake_llm(*replies), results_dir=tmp_path)

    assert row.cells == {1: [True], 2: [False], 3: [True]}
    assert row.score() == (2, 3)
    assert row.prompt == "balance_rtc_v1"
    with (tmp_path / "balance_rtc_v1.csv").open(encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert [r["result"] for r in saved] == ["right", "wrong", "right"]
    assert saved[0]["prompt"] == "balance_rtc_v1"


def test_repeats_run_the_whole_set_again(tmp_path):
    note = NOTES[:1]
    row = score_technique("rtc", "balance_rtc_v1", note, fake_llm(lines_reply(note[0]), "nothing useful"),
                          repeats=2, results_dir=tmp_path)
    assert row.cells == {1: [True, None]} and row.calls == 2


def test_structured_row_uses_python_for_the_maths(tmp_path):
    llm = fake_llm(*[key_reply(n) for n in NOTES])
    row = score_technique("structured", "expenses_v1", NOTES, llm, results_dir=tmp_path)
    assert row.score() == (10, 10)


def test_the_full_table_with_scripted_models(tmp_path, capsys):
    notes = NOTES[:2]
    llm = fake_llm(lines_reply(notes[0]), "I am not sure.")          # zero-shot: right, then no answer
    assert scoreboard.main(["--only", "zero-shot", "--notes", "1,2"], llm=llm, results_dir=tmp_path) == 0
    output = capsys.readouterr().out
    assert "zero-shot            balance_zero_v1       +  ?   1/2" in output
    assert "Python check" in output and "2/2" in output
    assert (tmp_path / "balance_zero_v1.csv").exists()


def test_without_a_key_it_shows_the_plan_and_the_python_check(no_keys, capsys):
    assert scoreboard.main([]) == 1
    output = capsys.readouterr().out
    assert output.count("needs a live run") == 5
    assert "settle(), no model" in output and "10/10" in output
    assert "A live run makes 50 model calls." in output
    assert "https://aistudio.google.com/apikey" in output


def test_a_new_prompt_version_can_replace_a_row():
    chosen = choose("rtc", ["rtc=balance_cot_v1"])
    assert chosen == {"rtc": ("role-task-context", "balance_cot_v1")}


def test_bad_options_give_a_clear_message(capsys):
    assert scoreboard.main(["--prompt", "magic=balance_rtc_v1"]) == 1
    assert "use one of zero-shot, rtc, few-shot, cot, structured" in capsys.readouterr().out


def test_a_rate_limit_stops_the_run_politely(tmp_path, capsys):
    def out_of_quota(_):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    assert scoreboard.main(["--only", "zero-shot"], llm=RunnableLambda(out_of_quota), results_dir=tmp_path) == 1
    output = capsys.readouterr().out
    assert "free-tier limit was reached" in output and "--pause 4" in output
