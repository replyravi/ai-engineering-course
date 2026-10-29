"""
tests/test_tool.py - the days_until_holiday tool on fixed dates (Step 10).

A tool is plain Python, so we test it like plain Python: no model at all.
We pin "today" with HELPDESK_TODAY so the day counts never change.
"""

from datetime import date

import rung5_tool
from helpdesk_fixtures import fixed_today  # noqa: F401
from rung5_tool import days_until_holiday, holiday_answer, load_holidays, today


def test_the_list_has_five_holidays_in_date_order():
    holidays = load_holidays()
    assert len(holidays) == 5
    assert holidays == sorted(holidays)


def test_today_can_be_pinned(fixed_today):
    assert today() == date(2026, 10, 23)


def test_next_holiday_on_a_fixed_date(fixed_today):
    assert days_until_holiday.invoke({}) == (
        "Kitchen deep-clean day is on Tuesday 17 November 2026: 25 days from today (Friday 23 October 2026)."
    )


def test_a_named_holiday(fixed_today):
    assert "Staff family day is on Monday 28 December 2026: 66 days" in days_until_holiday.invoke({"holiday": "family"})


def test_an_unknown_holiday_lists_the_known_ones(fixed_today):
    answer = days_until_holiday.invoke({"holiday": "Diwali"})
    assert answer.startswith("No kitchen holiday called 'Diwali' in the list.")
    assert "Kitchen deep-clean day" in answer


def test_on_the_holiday_itself():
    assert "(0 days)" in holiday_answer(on=date(2026, 11, 17))


def test_after_the_last_holiday():
    assert holiday_answer(on=date(2027, 9, 1)).startswith("No upcoming kitchen holiday in the list")
    assert "days ago" in holiday_answer("family", on=date(2027, 1, 4))


def test_a_broken_file_returns_an_error_message_instead_of_crashing(tmp_path):
    broken = tmp_path / "holidays.csv"
    broken.write_text("date,name\nnot-a-date,Oops\n", encoding="utf-8")
    assert holiday_answer(path=broken).startswith("Error: could not read the holiday list")
    assert holiday_answer(path=tmp_path / "missing.csv").startswith("Error:")


def test_what_the_model_sees():
    assert days_until_holiday.name == "days_until_holiday"
    assert "Read-only" in days_until_holiday.description
    assert days_until_holiday.args == {"holiday": {"default": "", "title": "Holiday", "type": "string"}}


def test_tool_only_needs_no_model(fixed_today, capsys):
    # (fixed_today also makes pytest undo the date that --today sets)
    assert rung5_tool.main(["--tool-only", "--today", "2026-10-23"]) == 0
    output = capsys.readouterr().out
    assert "What the model sees:" in output
    assert "25 days from today" in output
