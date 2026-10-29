"""
rung5_tool.py - Rung 5: a TOOL and your first AGENT (Steps 10 and 11).

A model cannot know today's date or read our holiday list. A tool can.
An agent is a loop:
    1. the model reads the question and decides: answer now, or call a tool?
    2. if it calls a tool, OUR Python function runs and the result goes back to the model
    3. the model reads the result and decides again, until it answers

The model only sees the tool's name, its description (the docstring) and its arguments.
Our tool is read-only: it reads data/holidays.csv and changes nothing.

Run:  python rung5_tool.py                        (Step 10: the agent answers)
      python rung5_tool.py --trace                (Step 11: print every message of the loop)
      python rung5_tool.py --tool-only            (run the tool yourself, no model call, no key)
      python rung5_tool.py --tool-only --today 2026-10-23     (pretend today is another date)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import csv
import json
import os
from datetime import date

from langchain.agents import create_agent
from langchain.tools import tool

from common import HOLIDAYS_FILE, HOUSE_RULES, friendly_error
from llm import MissingAPIKeyError, get_llm

QUESTION = "How many days until the next kitchen holiday?"


# ---------------------------------------------------------------------------
# The plain Python behind the tool (easy to test on a fixed date)
# ---------------------------------------------------------------------------

def today() -> date:
    """Today's date. Set HELPDESK_TODAY=2026-10-23 (or use --today) to pretend it is another day."""
    fixed = os.getenv("HELPDESK_TODAY", "").strip()
    return date.fromisoformat(fixed) if fixed else date.today()


def load_holidays(path=HOLIDAYS_FILE) -> list[tuple[date, str]]:
    """The kitchen holidays as (date, name), earliest first."""
    with open(path, newline="", encoding="utf-8") as file:
        rows = [(date.fromisoformat(row["date"].strip()), row["name"].strip()) for row in csv.DictReader(file)]
    return sorted(rows)


def nice(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B %Y}"      # Tuesday 17 November 2026 (and Friday 5 March 2027)


def holiday_answer(holiday: str = "", on: date | None = None, path=HOLIDAYS_FILE) -> str:
    """The sentence the tool returns: which holiday, on which date, how many days from today."""
    on = on or today()
    try:
        holidays = load_holidays(path)
    except (OSError, KeyError, ValueError) as error:
        # Return the problem as text: an exception inside a tool would stop the whole agent.
        return f"Error: could not read the holiday list ({error})."

    if holiday.strip():
        matches = [(d, n) for d, n in holidays if holiday.strip().lower() in n.lower()]
        if not matches:
            known = ", ".join(n for _, n in holidays)
            return f"No kitchen holiday called '{holiday}' in the list. Known holidays: {known}."
        upcoming = [(d, n) for d, n in matches if d >= on]
        if not upcoming:
            d, n = matches[-1]
            return f"{n} was on {nice(d)}, {(on - d).days} days ago. The next one is not in the list yet."
    else:
        upcoming = [(d, n) for d, n in holidays if d >= on]
        if not upcoming:
            d, n = holidays[-1]
            return (f"No upcoming kitchen holiday in the list (the last one was {n} on {nice(d)}). "
                    "The help desk has not added the next dates yet.")

    d, n = upcoming[0]
    days = (d - on).days
    if days == 0:
        return f"Today, {nice(on)}, is {n}: the kitchen is closed (0 days)."
    return f"{n} is on {nice(d)}: {days} days from today ({nice(on)})."


# ---------------------------------------------------------------------------
# The tool: the docstring and the typed argument are what the model sees
# ---------------------------------------------------------------------------

@tool
def days_until_holiday(holiday: str = "") -> str:
    """Count the days from today until a Dabbaloop kitchen holiday (no deliveries that day).

    Leave holiday empty for the next one, or give part of its name, like 'family day'.
    Read-only: it only reads the holiday list.
    """
    return holiday_answer(holiday)


AGENT_RULES = (HOUSE_RULES + " Use the days_until_holiday tool for any question about kitchen holidays. "
               "You do not know today's date; the tool does. Never guess a date or a number of days. "
               "If no tool can answer the question, say: I don't know.")

# Safety limit for the loop (LangGraph calls it recursion_limit). A normal answer (model, tool,
# model) fits easily; a model that keeps calling tools is stopped after 5 model calls.
# Without a limit, create_agent allows 9,999 steps, which could use up your free quota.
MAX_AGENT_STEPS = 10


def build_agent(llm=None):
    """The agent: a model + our one tool + the house rules. Pass llm=fake_llm(...) in tests."""
    if llm is None:
        llm = get_llm()
    return create_agent(model=llm, tools=[days_until_holiday], system_prompt=AGENT_RULES)


def ask_agent(question: str, llm=None) -> list:
    """Run the loop once. Returns every message: human, ai (tool call), tool, ai (answer)."""
    result = build_agent(llm).invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"recursion_limit": MAX_AGENT_STEPS},      # stop a loop that never answers
    )
    return result["messages"]


def tools_used(messages) -> list[str]:
    """The names of the tools the model asked for, in order."""
    return [call["name"] for m in messages if m.type == "ai" for call in m.tool_calls]


def print_trace(messages) -> None:
    """Step 11: every message of the loop, labelled by its type."""
    print(f"Message trace ({len(messages)} messages):")
    for number, m in enumerate(messages, start=1):
        if m.type == "ai" and m.tool_calls:
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)
            print(f"  {number}. ai     asks for a tool: {calls}")
        elif m.type == "tool":
            print(f"  {number}. tool   {m.name} -> {m.text}")
        else:
            print(f"  {number}. {m.type:<6} {m.text}")
    model_calls = sum(1 for m in messages if m.type == "ai")
    print(f"Tool calls: {len(tools_used(messages))}   Model calls: {model_calls}")


def show_tool() -> None:
    """What the model is shown about the tool, then the tool run by hand."""
    print("What the model sees:")
    print(f"  name:        {days_until_holiday.name}")
    first, *rest = [line.strip() for line in days_until_holiday.description.splitlines() if line.strip()]
    print(f"  description: {first}")
    for line in rest:
        print(f"               {line}")
    print(f"  arguments:   {json.dumps(days_until_holiday.args)}")
    print(f"\nToday is {nice(today())}.")
    print(f"days_until_holiday()              -> {days_until_holiday.invoke({})}")
    print(f"days_until_holiday('family day')  -> {days_until_holiday.invoke({'holiday': 'family day'})}")
    print(f"days_until_holiday('Diwali')      -> {days_until_holiday.invoke({'holiday': 'Diwali'})}")


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="Rung 5: a holiday tool and an agent.")
    parser.add_argument("question", nargs="?", default=QUESTION)
    parser.add_argument("--trace", action="store_true", help="Step 11: print every message")
    parser.add_argument("--tool-only", action="store_true", help="run the tool yourself (no model call)")
    parser.add_argument("--today", type=date.fromisoformat, help="pretend today is this date, like 2026-10-23")
    args = parser.parse_args(argv)
    if args.today:
        os.environ["HELPDESK_TODAY"] = args.today.isoformat()

    if args.tool_only:
        show_tool()
        return 0

    print(f"Question: {args.question}\n")
    try:
        messages = ask_agent(args.question, llm)
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1

    if args.trace:
        print_trace(messages)
    else:
        print(messages[-1].text)            # the final AI message is the answer
        used = tools_used(messages)
        print(f"\n(tools the agent used: {', '.join(used) if used else 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
