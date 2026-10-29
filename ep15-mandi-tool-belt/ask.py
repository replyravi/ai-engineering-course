"""
ask.py - ask the Mandi Tool Belt agent one question: trace, answer, sources and tokens.

    python ask.py fare            Step 3: calculator only, an auto-rickshaw fare
    python ask.py general         Step 4: the same agent, a question that needs no tool
    python ask.py onions          Step 7: SQL tools, "Which city had the cheapest onions on day 3?"
    python ask.py percent         Step 10: SQL then calculator
    python ask.py --list-tools    Step 13: the whole belt the agent holds (no key needed)
    python ask.py full            Step 15: docs + SQL + maths in one question

    python ask.py "your own question" --tools calc,sql,docs
    python ask.py onions --approve-sql      a human approves every SQL query before it runs

Every run is saved in runs/last_run.json, so python token_meter.py can show it call by call.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse

from agent import (MAX_MODEL_CALLS, RECURSION_LIMIT, build_agent, explain_error, final_answer, print_trace,
                   run_agent, sources, stopped_by_cap)
from llm import MissingAPIKeyError, get_llm
from token_meter import TOKEN_BUDGET, format_meter, meter, save_run
from tools import KINDS, tool_belt

# name -> (tool kinds, question). The numbers in the questions are invented sample data.
PRESETS = {
    "fare": ("calc", "An auto-rickshaw charges Rs 30 for the first 1.5 km and Rs 17 for every km after that. "
                     "Waiting adds Rs 20. What is the fare for an 8.5 km ride with waiting?"),
    "general": ("calc", "In one sentence, what is a mandi?"),
    "onions": ("sql", "Which city had the cheapest onions on day 3?"),
    "percent": ("calc,sql", "By what percent did the tomato price in Pune rise from day 1 to day 10?"),
    "full": ("calc,sql,docs", "A hotel in Pune buys 3 bags of onions at the day 7 Pune price. Using the rule book, "
                              "what is the total bill with commission, market fee and loading?"),
}


def parse_kinds(text: str) -> list[str]:
    kinds = [k.strip() for k in text.split(",") if k.strip()]
    return list(KINDS) if kinds == ["all"] else kinds


def model_name(llm) -> str:
    """The model id the chat model object holds (a fake model in tests has none)."""
    return getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__


def tool_cards(tools) -> list[str]:
    """One line per tool: what the model is shown (name, arguments, first line of the description)."""
    lines = [f"The agent holds {len(tools)} tool{'s' if len(tools) != 1 else ''}:"]
    for number, t in enumerate(tools, start=1):
        signature = f"{t.name}({', '.join(t.args)})"
        first = " ".join(t.description.split()).split(". ")[0].rstrip(".") + "."   # the first sentence
        lines.append(f"  {number}. {signature:<28} {first if len(first) <= 84 else first[:81] + '...'}")
    return lines


def run_with_approval(question, tools, llm, read=input, write=print) -> list:
    """Run the agent, but pause before every run_select so a human can approve or reject the SQL."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    gate = HumanInTheLoopMiddleware(interrupt_on={"run_select": {"allowed_decisions": ["approve", "reject"]}})
    agent = build_agent(tools, llm, extra_middleware=[gate], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "ask-py"}, "recursion_limit": RECURSION_LIMIT}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config, version="v2")
    while result.interrupts:
        decisions = []
        for request in result.interrupts[0].value["action_requests"]:
            write(f"\nThe agent wants to run:  {request['args'].get('query')}")
            yes = read("Run this SQL? [y/N] ").strip().lower() in ("y", "yes")
            decisions.append({"type": "approve"} if yes else
                             {"type": "reject", "message": "A human said no to this query."})
        result = agent.invoke(Command(resume={"decisions": decisions}), config, version="v2")
    return result.value["messages"]


def main(argv=None, llm=None, read=input, write=print) -> int:
    parser = argparse.ArgumentParser(description="Ask the Mandi Tool Belt agent one question.")
    parser.add_argument("question", nargs="?", default="onions",
                        help=f"a preset ({', '.join(PRESETS)}) or your own question in quotes")
    parser.add_argument("--tools", default=None, help="tool kinds: calc,sql,docs or all (presets choose for you)")
    parser.add_argument("--list-tools", action="store_true", help="show the tools and caps, then stop (no key)")
    parser.add_argument("--approve-sql", action="store_true", help="ask a human before every run_select")
    args = parser.parse_args(argv)

    preset_kinds, question = PRESETS.get(args.question, ("all", args.question))
    kinds = parse_kinds(args.tools or ("all" if args.list_tools and not args.tools else preset_kinds))
    try:
        tools = tool_belt(kinds)
    except ValueError as error:
        write(str(error))
        return 1

    if args.list_tools:
        for line in tool_cards(tools):
            write(line)
        write(f"Caps: at most {MAX_MODEL_CALLS} model calls and {TOKEN_BUDGET:,} tokens per question "
              f"(recursion_limit {RECURSION_LIMIT} as a last safety net)")
        return 0

    write(f"Question: {question}")
    write(f"Tools: {', '.join(t.name for t in tools)}")
    if "sql" in kinds:
        from tools import sql
        if not sql.DB_PATH.exists():          # stop before the model spends tokens on a missing database
            write(f"{sql.DB_PATH.name} not found. Create it first with:  python seed_db.py")
            return 1
    try:
        if llm is None:
            llm = get_llm()
        write(f"Model: {model_name(llm)}   Caps: {MAX_MODEL_CALLS} model calls, {TOKEN_BUDGET:,} tokens\n")
        if args.approve_sql:
            messages = run_with_approval(question, tools, llm, read, write)
        else:
            messages = run_agent(question, tools, llm)
    except (MissingAPIKeyError, ValueError) as error:
        write(str(error).strip())
        return 1
    except Exception as error:
        write(explain_error(error))
        return 1

    print_trace(messages, out=write)
    write(f"\nAnswer: {final_answer(messages)}")
    if stopped_by_cap(messages):
        write("(A cap stopped this run before the model finished. See the last line of the trace.)")
    found = sources(messages)
    if found:
        write("Sources (what the tools returned):")
        for source in found:
            write(f"  {source}")
    write(format_meter(meter(messages)))
    save_run(question, messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
