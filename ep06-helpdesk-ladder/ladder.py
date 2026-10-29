"""
ladder.py - the Help Desk Ladder report (Step 12): which rung solves which question?

Climb only as high as you need. For every sample question we try the rungs in order,
cheapest first, and stop at the first one whose answer passes OUR check:

    Prompt      house rules + the question                      (what rung1 and rung2 scripts do)
    Context     + the top 3 handbook paragraphs                  (rung3_context.py)
    Tools       an agent with the days_until_holiday tool        (rung5_tool.py)
    Fine-tune   never run here: it changes style and behaviour, and it is a weak way to add facts

The checks are plain Python (a time that must appear, the right number of days, the house
greeting, an honest "I don't know"), so the report does not depend on a model grading itself.

Run:  python ladder.py
      python ladder.py --today 2026-10-23      (pretend today is another date)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import os
import re
from datetime import date

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import HOUSE_RULES, friendly_error, is_refusal
from llm import MissingAPIKeyError, get_embeddings, get_llm, get_model_name, get_provider
from rung3_context import answer_with_context, load_paragraphs
from rung5_tool import ask_agent, load_holidays, today, tools_used

RUNGS = ["Prompt", "Context", "Tools"]

# The sample questions and what a passing answer must contain.
QUESTIONS = [
    {"question": "Explain in two sentences what a meal-box subscription is.",
     "kind": "general", "check": "any answer that is not 'I don't know'"},
    {"question": "What is the last time I can skip today's lunch?",
     "kind": "fact", "must_contain": "9:30", "check": "must mention 9:30 (handbook.md#4)"},
    {"question": "How many days until the next kitchen holiday?",
     "kind": "days", "check": "must give the right number of days"},
    {"question": "A customer wrote: 'Thanks, the dal was lovely today!' Write our reply.",
     "kind": "tone", "must_contain": "namaste", "check": "must use the house greeting Namaste"},
    {"question": "Is there a student discount?",
     "kind": "not_covered", "check": "must say I don't know (the handbook has no discount)"},
]

# Rung 1: the house rules, plus permission to say "I don't know".
PROMPT_RUNG = ChatPromptTemplate.from_messages([
    ("system", HOUSE_RULES + " Answer from general knowledge only. If the question needs Dabbaloop's "
               "own rules, times, prices or dates, or today's date, say: I don't know."),
    ("human", "{question}"),
])


def next_holiday(on: date) -> tuple[int, str] | None:
    """(days, name) of the next kitchen holiday, worked out by our code (the answer key for question 3)."""
    upcoming = [(d, n) for d, n in load_holidays() if d >= on]
    return ((upcoming[0][0] - on).days, upcoming[0][1]) if upcoming else None


def expected_days(on: date) -> int | None:
    found = next_holiday(on)
    return found[0] if found else None


def grade(item: dict, answer: str, days: int | None) -> str:
    """Check one answer: 'solved', "I don't know", 'wrong' or 'answered?'."""
    if is_refusal(answer):
        return "I don't know"
    kind = item["kind"]
    if kind == "not_covered":
        return "answered?"           # the handbook has no answer, so any answer may be invented
    if kind == "general":
        return "solved" if answer.strip() else "wrong"
    if kind == "days":
        return "solved" if days is not None and re.search(rf"\b{days}\b", answer) else "wrong"
    return "solved" if item["must_contain"] in answer.lower() else "wrong"


def try_rung(rung: str, question: str, llm, embeddings, paragraphs) -> tuple[str, int, list[str]]:
    """Ask one rung. Returns (answer text, model calls used, tools used)."""
    if rung == "Prompt":
        return (PROMPT_RUNG | llm | StrOutputParser()).invoke({"question": question}), 1, []
    if rung == "Context":
        answer, _ = answer_with_context(question, llm, embeddings, paragraphs=paragraphs)
        return answer, 1, []
    messages = ask_agent(question, llm)
    return messages[-1].text, sum(1 for m in messages if m.type == "ai"), tools_used(messages)


def climb(item: dict, llm, embeddings, paragraphs, days) -> dict:
    """Try the rungs cheapest first and stop at the first one that passes the check."""
    row = {"question": item["question"], "cells": [], "winner": None, "answer": "", "calls": 0}
    for rung in RUNGS:
        answer, calls, used = try_rung(rung, item["question"], llm, embeddings, paragraphs)
        row["calls"] += calls
        status = grade(item, answer, days)
        if status == "solved" and rung == "Tools":
            status = "solved (tool)" if used else "solved (no tool)"
        row["cells"].append(status)
        row["answer"] = answer
        if status.startswith("solved"):
            row["winner"] = rung + (" (no fine-tune needed)" if item["kind"] == "tone" else "")
            break
        if status == "answered?":
            row["winner"] = f"check it: {rung} answered without a source"
            break
    if row["winner"] is None:
        row["winner"] = ("none: says I don't know (correct)" if item["kind"] == "not_covered"
                         else "none of the rungs")
    row["cells"] += ["-"] * (len(RUNGS) - len(row["cells"]))   # rungs we did not need to try
    return row


def print_report(rows: list[dict], model: str, on: date) -> None:
    print(f"Help Desk Ladder report   model: {model}   today: {on.isoformat()}\n")
    header = f"{'#':<3}{'Question':<46}" + "".join(f"{r:<15}" for r in RUNGS) + "Cheapest rung that worked"
    print(header)
    print("-" * len(header))
    for number, row in enumerate(rows, start=1):
        question = row["question"] if len(row["question"]) <= 44 else row["question"][:41] + "..."
        print(f"{number:<3}{question:<46}" + "".join(f"{c:<15}" for c in row["cells"]) + row["winner"])
    tone = next(n for n, item in enumerate(QUESTIONS, start=1) if item["kind"] == "tone")
    if rows[tone - 1]["winner"].startswith("Prompt"):
        print(f"\nFine-tune (the 4th rung): not tried. The house tone came from the prompt (question {tone}),")
    else:
        print(f"\nFine-tune (the 4th rung): not tried. Question {tone} missed the house tone: write a clearer")
        print("house-style rule in the prompt first. That is far cheaper than fine-tuning,")
    print("and fine-tuning is a weak way to teach facts that change, like dates and rules.")
    print(f"Model calls: {sum(row['calls'] for row in rows)}\n")
    print("Answer from the last rung tried:")
    for number, row in enumerate(rows, start=1):
        print(f"  {number}. {' '.join(row['answer'].split())}")


def main(argv=None, llm=None, embeddings=None) -> int:
    parser = argparse.ArgumentParser(description="Which rung of the ladder solves each question?")
    parser.add_argument("--today", type=date.fromisoformat, help="pretend today is this date, like 2026-10-23")
    args = parser.parse_args(argv)
    if args.today:
        os.environ["HELPDESK_TODAY"] = args.today.isoformat()

    on = today()
    days = expected_days(on)
    print("Answer key (checked by our code, not by a model):")
    for number, item in enumerate(QUESTIONS, start=1):
        check = item["check"]
        if item["kind"] == "days":
            found = next_holiday(on)
            check = f"must say {found[0]} (days to {found[1]})" if found else "no upcoming holiday in the list"
        print(f"  {number}. {check}")
    print()

    try:
        if llm is None:
            llm = get_llm()
        if embeddings is None:
            embeddings = get_embeddings()
        paragraphs = load_paragraphs()
        rows = [climb(item, llm, embeddings, paragraphs, days) for item in QUESTIONS]
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1

    provider = get_provider()
    print_report(rows, f"{get_model_name(provider)} ({provider})", on)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
