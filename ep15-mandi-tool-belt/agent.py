"""
agent.py - ONE agent that holds any tools from the belt, with caps and a readable trace.

create_agent runs the loop for us:
    1. the model reads the question and the tool cards (name, description, argument schema)
    2. it answers, OR it returns a structured tool call: {"name": "run_select", "args": {...}}
    3. our Python runs that tool and sends the result back as a tool message
    4. repeat until the model answers, or a cap stops the run

Caps (every run has all three):
    MAX_MODEL_CALLS  at most 8 model calls per question     (ModelCallLimitMiddleware)
    TOKEN_BUDGET     at most 30,000 tokens per question      (token_meter.token_budget)
    RECURSION_LIMIT  LangGraph's own step limit, a last safety net. LangGraph counts every node it
                     runs, about 5 per model call here (2 checks before, the model, 1 after, the tools),
                     so 50 is never reached while the call cap works. Without it the limit is 9,999.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import json
import re

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from llm import get_llm
from token_meter import TOKEN_BUDGET, is_cap_notice, token_budget

MAX_MODEL_CALLS = 8
RECURSION_LIMIT = 50

BASE_RULES = ("You are the Mandi Tool Belt assistant for vegetable prices in Nashik, Pune and Indore. "
              "The price data is invented sample data for a course, not real market rates. "
              "If a question needs none of your tools, just answer it. Keep answers short.")

# One rule per tool the agent holds, so the prompt never mentions a tool it does not have.
TOOL_RULES = {
    "calculator": "For every calculation, call the calculator, even for small sums. Never do maths in your head.",
    "run_select": ("For prices and arrivals, use the SQL tools: list_tables, then describe_table, then run_select. "
                   "Never guess a price."),
    "get_prices": "For prices and arrivals, use get_prices. Never guess a price.",
    "search_docs": ("For the rule book (buyer charges, bag and crate sizes, units), use search_docs "
                    "and name the source URL in your answer."),
}


def system_prompt_for(tools) -> str:
    names = {t.name for t in tools}
    return " ".join([BASE_RULES] + [rule for name, rule in TOOL_RULES.items() if name in names])


def build_agent(tools, llm=None, max_model_calls=MAX_MODEL_CALLS, max_tokens=TOKEN_BUDGET,
                extra_middleware=(), checkpointer=None):
    """The agent: a model + the tools + the caps. Pass llm=fake_llm(...) in tests."""
    if llm is None:
        llm = get_llm()
    caps = [ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"), token_budget(max_tokens)]
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt_for(tools),
                        middleware=[*caps, *extra_middleware], checkpointer=checkpointer)


def run_agent(question, tools, llm=None, **caps) -> list:
    """Ask one question. Returns every message of the run: human, ai, tool, ..., ai."""
    agent = build_agent(tools, llm, **caps)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]},
                          {"recursion_limit": RECURSION_LIMIT})
    return result["messages"]


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------

def tool_calls(messages) -> list[tuple[str, dict]]:
    """Every tool the model asked for, in order, as (name, args)."""
    return [(c["name"], c["args"]) for m in messages if m.type == "ai" for c in m.tool_calls]


def final_answer(messages) -> str:
    """The text of the last AI message (or the note of the cap that stopped the run)."""
    for message in reversed(messages):
        if message.type == "ai" and not message.tool_calls:
            return message.text.strip()
    return "(no answer)"


def stopped_by_cap(messages) -> str | None:
    """The cap's note when a cap ended the run, else None."""
    last = messages[-1]
    return last.text if is_cap_notice(last) else None


def sources(messages) -> list[str]:
    """Where the facts came from: every SQL the tools ran and every page search_docs returned."""
    found = []
    for message in messages:
        if message.type != "tool":
            continue
        for line in message.text.splitlines():
            if line.startswith("SQL used"):
                found.append(line)
        found += [f"page: {url}" for url in re.findall(r"source: (\S+)", message.text)]
    return list(dict.fromkeys(found))          # keep the order, drop repeats


def short(text: str, width: int, max_lines: int) -> list[str]:
    """At most max_lines lines of a message, each cut to width, plus how many lines were hidden."""
    lines = [line if len(line) <= width else line[: width - 3] + "..." for line in text.strip().splitlines()]
    lines = [line for line in lines if line.strip()] or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"(+{len(lines) - max_lines} more lines)"]
    return lines


def print_trace(messages, width=100, max_lines=5, out=print) -> None:
    """Every message of the run, numbered. Tool results show their first lines."""
    out(f"Trace ({len(messages)} messages):")

    def show(number, label, text, limit=max_lines):
        first, *rest = short(text, width, limit)
        out(f"  {number:>2}. {label} {first}")
        for line in rest:
            out(f"      {' ' * len(label)} {line}")

    for number, m in enumerate(messages, start=1):
        if m.type == "ai" and m.tool_calls:
            if m.text.strip():
                show(number, "ai     says:", m.text, 2)
            for c in m.tool_calls:
                out(f"  {number:>2}. ai     calls {c['name']}({json.dumps(c['args'], ensure_ascii=False)})")
        elif m.type == "tool":
            show(number, f"tool   {m.name} ->", m.text)
        else:
            show(number, f"{m.type:<6}", m.text)
    calls = tool_calls(messages)
    if calls:
        out(f"Tool calls: {len(calls)} ({', '.join(name for name, _ in calls)})")
    else:
        out("Tool calls: 0 (the model answered by itself)")


def explain_error(error: Exception) -> str:
    """A short, friendly explanation of a failed run, with the fix."""
    text = f"{type(error).__name__}: {error}"
    if type(error).__name__ == "GraphRecursionError":
        return (f"The agent hit LangGraph's step limit (RECURSION_LIMIT = {RECURSION_LIMIT} in agent.py) "
                "before any other cap stopped it.")
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (f"The free-tier limit was reached ({text[:120]}...).\n"
                "Wait a minute and try again. Your limits: https://aistudio.google.com/rate-limit")
    if "Connect" in type(error).__name__:
        return f"Could not reach the model's server ({text}). Check your internet connection."
    return f"The model call failed: {text}"
