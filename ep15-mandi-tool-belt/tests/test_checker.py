"""
tests/test_checker.py - the grader (Steps 8 and 9).

1. The truth queries are checked against plain Python over the seed rows (we check the checker).
2. The grading rules: ANSWER line, 1 percent tolerance, city names.
3. A whole scoreboard with a SCRIPTED model: four right answers and one wrong ON PURPOSE.
"""

import pytest
from mandi_fixtures import no_keys, temp_db  # noqa: F401  (pytest fixtures)

import check_answers
from check_answers import answer_line, check_all, first_number, grade, load_questions, print_scoreboard, run_value
from seed_db import make_rows
from testing.fakes import fake_llm, tool_call

QUESTIONS, TOLERANCE = load_questions()
BY_ID = {q["id"]: q for q in QUESTIONS}


def rows_of(vegetable, city=None, day=None):
    return [r for r in make_rows() if r["vegetable"] == vegetable
            and (city is None or r["city"] == city) and (day is None or r["day"] == day)]


def test_truth_queries_match_plain_python(temp_db):
    onions_day3 = rows_of("onion", day=3)
    brinjal = rows_of("brinjal", "Indore")
    potato = rows_of("potato", "Pune")
    tomato = {r["day"]: r["price_rs_per_kg"] for r in rows_of("tomato", "Pune")}
    python_answers = {
        1: min(onions_day3, key=lambda r: r["price_rs_per_kg"])["city"],
        2: round(sum(r["price_rs_per_kg"] for r in brinjal) / len(brinjal), 2),
        3: max(potato, key=lambda r: r["arrivals_quintals"])["day"],
        4: sum(r["arrivals_quintals"] for r in onions_day3 if r["city"] in ("Nashik", "Pune")),
        5: round((tomato[10] - tomato[1]) * 100 / tomato[1], 2),
    }
    sql_answers = {q["id"]: run_value(q["truth_sql"], temp_db) for q in QUESTIONS}
    assert sql_answers == python_answers == {1: "Indore", 2: 33.86, 3: 10, 4: 1075, 5: 23.13}


def test_the_trap_query_leaks_rows(temp_db):
    trap = BY_ID[4]
    assert run_value(trap["trap_sql"], temp_db) == 15862                 # not 1075
    assert check_answers.count_rows(trap["trap_sql"], temp_db) == 51     # 1 Nashik row + all 50 Pune rows
    assert check_answers.count_rows(trap["truth_sql"], temp_db) == 2


@pytest.mark.parametrize("reply, expected", [
    ("Indore had the cheapest onions.\nANSWER: Indore", "Indore"),
    ("answer: 23.1%", "23.1%"),
    ("ANSWER: 1,000\nANSWER: 1,075 quintals", "1,075 quintals"),       # the LAST answer line counts
    ("It was Indore.", None),
])
def test_answer_line(reply, expected):
    assert answer_line(reply) == expected


def test_first_number():
    assert first_number("Rs 1,075 quintals") == 1075
    assert first_number("23.13%") == 23.13
    assert first_number("day ten") is None


@pytest.mark.parametrize("question_id, reply, passed", [
    (1, "ANSWER: Indore.", True),
    (1, "ANSWER: Nashik", False),
    (2, "ANSWER: Rs 33.9", True),           # within 1 percent of 33.86
    (2, "ANSWER: 35", False),
    (3, "ANSWER: Day 10", True),
    (4, "ANSWER: 15862", False),
    (5, "ANSWER: 23.1", True),
    (5, "The price rose by about 23 percent.", False),   # no ANSWER line
])
def test_grade(question_id, reply, passed):
    truth = {1: "Indore", 2: 33.86, 3: 10, 4: 1075, 5: 23.13}[question_id]
    assert grade(BY_ID[question_id], truth, reply, TOLERANCE)["passed"] is passed


SCRIPT = [
    # 1: list, describe, select, answer
    tool_call("list_tables", {}),
    tool_call("describe_table", {"table_name": "daily_prices"}),
    tool_call("run_select", {"query": "SELECT city FROM daily_prices WHERE vegetable = 'onion' AND day = 3 "
                                      "ORDER BY price_rs_per_kg LIMIT 1"}),
    "[SCRIPTED] Indore.\nANSWER: Indore",
    # 2
    tool_call("run_select", {"query": "SELECT AVG(price_rs_per_kg) FROM daily_prices "
                                      "WHERE vegetable = 'brinjal' AND city = 'Indore'"}),
    "[SCRIPTED] About Rs 33.86 per kg.\nANSWER: 33.86",
    # 3
    tool_call("run_select", {"query": "SELECT day FROM daily_prices WHERE vegetable = 'potato' AND city = 'Pune' "
                                      "ORDER BY arrivals_quintals DESC LIMIT 1"}),
    "[SCRIPTED] Day 10.\nANSWER: 10",
    # 4: WRONG ON PURPOSE - the SQL without brackets
    tool_call("run_select", {"query": BY_ID[4]["trap_sql"]}),
    "[SCRIPTED] 15862 quintals.\nANSWER: 15862",
    # 5: SQL, then the calculator
    tool_call("run_select", {"query": "SELECT day, price_rs_per_kg FROM daily_prices WHERE vegetable = 'tomato' "
                                      "AND city = 'Pune' AND day IN (1, 10)"}),
    tool_call("calculator", {"expression": "(36.2 - 29.4) / 29.4 * 100"}),
    "[SCRIPTED] It rose about 23.13 percent.\nANSWER: 23.13",
]


def test_scoreboard_with_one_wrong_answer_on_purpose(temp_db, capsys):
    tools = check_answers.tools_for(None, db_path=temp_db)
    rows = check_all(QUESTIONS, tools, fake_llm(*SCRIPT), TOLERANCE, db_path=temp_db)
    assert [row["passed"] for row in rows] == [True, True, True, False, True]

    print_scoreboard(rows, "test")
    out = capsys.readouterr().out
    assert "Score: 4/5" in out
    assert " 1  PASS    Indore   Indore   list, describe, select" in out
    assert " 4  FAIL    1075     15862    select" in out
    assert " 5  PASS    23.13    23.13    select, calc" in out
    assert "FAIL 4: How many quintals of onions reached Nashik and Pune on day 3, taken together?" in out
    assert "SQL used: " + BY_ID[4]["trap_sql"] + " LIMIT 20" in out      # the SQL to look at
    assert "Tokens: not reported by this model (13 model calls)" in out  # the fake reports no usage


def test_the_question_carries_the_answer_rule(temp_db):
    llm = fake_llm("ANSWER: Indore")
    check_all(QUESTIONS[:1], check_answers.tools_for(None, db_path=temp_db), llm, TOLERANCE, db_path=temp_db)
    assert llm.received[0][-1].content.endswith("ANSWER: <only the number or the city name, no units>")


def test_replay_trap_fails_then_passes(temp_db, capsys):
    assert check_answers.replay_trap(db_path=temp_db) == [False, True]
    out = capsys.readouterr().out
    assert "rows the WHERE clause lets through: 51" in out
    assert "Grade: FAIL (expected 1075 (within 1 percent), the agent said 15862)" in out
    assert "Grade: PASS" in out


def test_truths_need_no_key(no_keys, temp_db, capsys):
    assert check_answers.main(["--truths"], db_path=temp_db) == 0
    assert "right answer: 1075" in capsys.readouterr().out


def test_no_key_prints_the_help(no_keys, temp_db, capsys):
    assert check_answers.main([], db_path=temp_db) == 1
    assert "Get a FREE Gemini key" in capsys.readouterr().out


def test_failed_runs_show_error_and_no_fake_zero(temp_db, monkeypatch, capsys):
    def offline(*args, **kwargs):          # like a run with no internet: the model is never reached
        raise ConnectionError("no internet")

    monkeypatch.setattr(check_answers, "run_agent", offline)
    rows = check_all(QUESTIONS[:1], check_answers.tools_for(None, db_path=temp_db), fake_llm(), TOLERANCE,
                     db_path=temp_db)
    assert rows[0]["agent"] == "ERROR"
    print_scoreboard(rows, "test")
    out = capsys.readouterr().out
    assert out.splitlines()[2].split() == ["1", "FAIL", "Indore", "ERROR", "none", "-"]
    assert "Tokens: none counted (no model call finished)" in out
    assert "0 in + 0 out" not in out


def test_a_run_stopped_by_a_cap_says_so(temp_db):
    endless = fake_llm(*[tool_call("list_tables", {}) for _ in range(20)])
    rows = check_all(QUESTIONS[:1], check_answers.tools_for(None, db_path=temp_db), endless, TOLERANCE,
                     db_path=temp_db)
    assert rows[0]["passed"] is False
    assert rows[0]["reason"] == "a cap stopped the run: Model call limits exceeded: run limit (8/8)"
