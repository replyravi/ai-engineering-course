"""
check_answers.py - Steps 8 and 9: grade the agent against answers we worked out ourselves.

Never trust an agent's number until you have checked it. Like checking a bill against the
receipt: data/questions.json holds 5 questions, each with a truth query written by hand.
We run the truth query ourselves, ask the agent the question, and compare.

    python check_answers.py --truths            the questions, their truth SQL and the right answers (no key)
    python check_answers.py                     Step 8: the scoreboard, run_select with its BASIC description
    python check_answers.py --fix description   Step 9: the same, with the CAREFUL description (brackets, IN)
    python check_answers.py --fix price-tool    Step 9, option 2: get_prices instead of run_select
    python check_answers.py --replay-trap       a SCRIPTED agent falls into the AND/OR trap, then avoids it (no key)

Grading: the agent ends its reply with "ANSWER: <value>". Numbers pass within 1 percent of the
truth (so 23.1 passes for 23.13); city names must match.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
import math
import re
from pathlib import Path

from agent import explain_error, final_answer, print_trace, run_agent, sources, stopped_by_cap, tool_calls
from llm import MissingAPIKeyError, get_llm
from seed_db import DB_PATH
from token_meter import format_meter, meter
from tools import tool_belt
from tools.ep12_sql_guard import run_guarded
from tools.sql import ALLOWED

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "data" / "questions.json"

ANSWER_RULE = ("\n\nEnd your reply with one last line in this form: "
               "ANSWER: <only the number or the city name, no units>")

SHORT_NAMES = {"calculator": "calc", "list_tables": "list", "describe_table": "describe",
               "run_select": "select", "get_prices": "prices", "search_docs": "docs"}


# ---------------------------------------------------------------------------
# 1. The truth: our own SQL, run read-only
# ---------------------------------------------------------------------------

def load_questions(path=QUESTIONS) -> tuple[list[dict], float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"], data["tolerance_percent"]


def run_value(sql: str, db_path=None):
    """The first value of the first row of a query, through the same read-only guard as the agent."""
    result = run_guarded(sql, db_path or DB_PATH, allowed=ALLOWED)
    if not result.ok:
        raise RuntimeError(f"{result.message} ({sql})")
    return result.rows[0][0]


def shown(value) -> str:
    """1075 -> '1075', 33.86 -> '33.86', 10.0 -> '10'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# 2. Grading one reply
# ---------------------------------------------------------------------------

def answer_line(reply: str) -> str | None:
    """The value after the last 'ANSWER:' in the reply, or None."""
    found = re.findall(r"ANSWER:\s*(.+)", reply, flags=re.IGNORECASE)
    return found[-1].strip() if found else None


def first_number(text: str) -> float | None:
    """The first number in a piece of text: 'Rs 1,075 quintals' -> 1075.0, '23.13%' -> 23.13."""
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def grade(question: dict, truth, reply: str, tolerance_percent: float = 1) -> dict:
    """PASS or FAIL for one reply, with the value we read from it."""
    value = answer_line(reply)
    if value is None:
        return {"passed": False, "agent": "-", "reason": "no 'ANSWER:' line in the reply"}
    if question["kind"] == "text":
        clean = re.sub(r"[^a-z]", "", value.lower())
        passed = clean == re.sub(r"[^a-z]", "", str(truth).lower())
        return {"passed": passed, "agent": value, "reason": "" if passed else f"expected {truth}"}
    number = first_number(value)
    if number is None:
        return {"passed": False, "agent": value, "reason": "no number after 'ANSWER:'"}
    passed = math.isclose(number, float(truth), rel_tol=tolerance_percent / 100, abs_tol=0.05)
    return {"passed": passed, "agent": shown(number),
            "reason": "" if passed else f"expected {shown(truth)} (within {tolerance_percent:g} percent)"}


# ---------------------------------------------------------------------------
# 3. The scoreboard
# ---------------------------------------------------------------------------

def tools_for(fix: str | None, db_path=None) -> list:
    """The tools of the check: calculator + SQL. fix: None, 'description' or 'price-tool'."""
    return tool_belt(("calc", "sql"), careful=(fix == "description"), price_tool=(fix == "price-tool"),
                     db_path=db_path)


def check_all(questions, tools, llm, tolerance_percent=1, db_path=None) -> list[dict]:
    """Ask every question (a fresh agent each time) and grade the replies."""
    rows = []
    for question in questions:
        truth = run_value(question["truth_sql"], db_path)
        try:
            messages = run_agent(question["question"] + ANSWER_RULE, tools, llm)
        except Exception as error:      # for example the free-tier limit: record it, go on
            rows.append({"question": question, "truth": truth, "messages": [], "passed": False,
                         "agent": "ERROR", "reason": explain_error(error)})
            continue
        result = grade(question, truth, final_answer(messages), tolerance_percent)
        if stopped_by_cap(messages):
            result["reason"] = f"a cap stopped the run: {stopped_by_cap(messages)}"
        rows.append({"question": question, "truth": truth, "messages": messages, **result})
    return rows


def print_scoreboard(rows, label: str, out=print) -> None:
    out(f"Scoreboard: {len(rows)} question{'s' if len(rows) != 1 else ''}, {label}")
    out(f" #  result  {'truth':<8} {'agent':<8} {'tools called':<40} tokens")
    for row in rows:
        calls = ", ".join(SHORT_NAMES.get(name, name) for name, _ in tool_calls(row["messages"])) or "none"
        usage = meter(row["messages"])
        if usage["model_calls"] == 0:          # an ERROR row: the run stopped with an error, no messages kept
            tokens = "-"
        else:
            tokens = f"{usage['total']:,}" if usage["model_calls"] > usage["not_reported"] else "n/r"
        out(f" {row['question']['id']}  {'PASS' if row['passed'] else 'FAIL':<6}  {shown(row['truth']):<8} "
            f"{str(row['agent'])[:8]:<8} {calls[:40]:<40} {tokens}")
    passed = sum(row["passed"] for row in rows)
    out(f"Score: {passed}/{len(rows)}")
    out(format_meter(meter([m for row in rows for m in row["messages"]])))
    for row in rows:
        if row["passed"]:
            continue
        out(f"\nFAIL {row['question']['id']}: {row['question']['question']}")
        out(f"  {row['reason']}")
        for source in sources(row["messages"]):
            out(f"  {source}")


# ---------------------------------------------------------------------------
# 4. The replay: what the trap looks like, with a SCRIPTED model (no AI, no key)
# ---------------------------------------------------------------------------

def count_rows(sql: str, db_path=None) -> int:
    """How many rows the WHERE clause of a SUM query lets through."""
    return run_value(sql.replace("SELECT SUM(arrivals_quintals)", "SELECT COUNT(*)"), db_path)


def replay_trap(db_path=None, out=print) -> list[bool]:
    from testing.fakes import fake_llm, tool_call     # the course's scripted fake model

    questions, tolerance = load_questions()
    trap = next(q for q in questions if "trap_sql" in q)
    truth = run_value(trap["truth_sql"], db_path)
    out("Replay: a SCRIPTED model (no AI, no key) writes the SQL. The tools, the database and the grader are real.")
    out(f"\nQuestion {trap['id']}: {trap['question']}")
    out(f"Truth (our SQL): {shown(truth)}")
    out(f"Why it is a trap: {trap['trap']}")
    results = []
    for title, sql in [("Without brackets", trap["trap_sql"]), ("With brackets", trap["truth_sql"])]:
        value = shown(run_value(sql, db_path))
        out(f"\n--- {title} ---")
        out(f"rows the WHERE clause lets through: {count_rows(sql, db_path)}")
        script = fake_llm(tool_call("run_select", {"query": sql}),
                          f"[SCRIPTED] Nashik and Pune got {value} quintals of onions on day 3.\nANSWER: {value}")
        messages = run_agent(trap["question"] + ANSWER_RULE, tools_for(None, db_path), script)
        print_trace(messages, out=out)
        result = grade(trap, truth, final_answer(messages), tolerance)
        verdict = "PASS" if result["passed"] else f"FAIL ({result['reason']}, the agent said {result['agent']})"
        out(f"Grade: {verdict}")
        results.append(result["passed"])
    return results


# ---------------------------------------------------------------------------
# 5. The command line
# ---------------------------------------------------------------------------

def main(argv=None, llm=None, out=print, db_path=None) -> int:
    parser = argparse.ArgumentParser(description="Grade the agent against hand-written SQL truths.")
    parser.add_argument("--fix", choices=["description", "price-tool"], help="Step 9: try a fix")
    parser.add_argument("--truths", action="store_true", help="show the questions and the right answers (no key)")
    parser.add_argument("--replay-trap", action="store_true", help="the trap with a scripted model (no key)")
    args = parser.parse_args(argv)

    db_path = Path(db_path or DB_PATH)
    if not db_path.exists():
        out(f"{db_path.name} not found. Create it first with:  python seed_db.py")
        return 1
    questions, tolerance = load_questions()

    if args.truths:
        for q in questions:
            out(f"{q['id']}. {q['question']}")
            out(f"   truth SQL: {q['truth_sql']}")
            out(f"   right answer: {shown(run_value(q['truth_sql'], db_path))}")
        return 0
    if args.replay_trap:
        replay_trap(db_path, out=out)
        return 0

    label = {None: "run_select with its BASIC description",
             "description": "run_select with the CAREFUL description (the fix)",
             "price-tool": "get_prices instead of run_select (the parameterised fix)"}[args.fix]
    try:
        llm = llm or get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        out(str(error).strip())
        return 1
    rows = check_all(questions, tools_for(args.fix, db_path), llm, tolerance, db_path)
    print_scoreboard(rows, label, out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
