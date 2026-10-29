"""
token_meter.py - Step 14: count the tokens of every model call, and cap them.

Every AI message a LangChain chat model returns carries `usage_metadata`:
    {"input_tokens": 812, "output_tokens": 45, "total_tokens": 857,
     "output_token_details": {"reasoning": 30}}          # Gemini: thinking tokens, part of output_tokens
So the meter just adds up the AI messages of a run. It works for every provider, because it
reads the messages, not a provider-specific callback. A call that reports NO usage is counted
as "not reported", never as a silent 0.

    python token_meter.py        the per-call table of your last ask.py run (saved in runs/last_run.json)

The cap: token_budget(max_tokens) is a small middleware. Before each model call it adds up the
tokens used so far and ends the run when the budget is spent.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import json
from pathlib import Path

from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessage, messages_from_dict, messages_to_dict

HERE = Path(__file__).resolve().parent
LAST_RUN = HERE / "runs" / "last_run.json"

TOKEN_BUDGET = 30_000      # tokens per question (input + output, thinking included)
CALL_LIMIT_NOTICE = "Model call limits exceeded"     # what ModelCallLimitMiddleware writes when it stops a run


def is_cap_notice(message) -> bool:
    """True for the note a cap adds when it stops a run. It is an AI message, but not a model call."""
    return message.type == "ai" and (message.name == "token_budget" or message.text.startswith(CALL_LIMIT_NOTICE))


def usage_of(message) -> dict | None:
    """The usage of one AI message, or None when the provider reported none."""
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    details = usage.get("output_token_details") or {}
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "thinking": details.get("reasoning", 0),
        "total": usage.get("total_tokens", 0),
    }


def meter(messages) -> dict:
    """Add up every AI message of a run."""
    totals = {"model_calls": 0, "not_reported": 0, "input": 0, "output": 0, "thinking": 0, "total": 0}
    for message in messages:
        if message.type != "ai" or is_cap_notice(message):
            continue
        totals["model_calls"] += 1
        usage = usage_of(message)
        if usage is None:
            totals["not_reported"] += 1
            continue
        for key in ("input", "output", "thinking", "total"):
            totals[key] += usage[key]
    return totals


def format_meter(totals: dict) -> str:
    """One line for the end of every run."""
    calls = f"{totals['model_calls']} model call{'s' if totals['model_calls'] != 1 else ''}"
    if totals["model_calls"] == 0:                  # no AI message at all, e.g. every run stopped with an error
        return "Tokens: none counted (no model call finished)"
    if totals["not_reported"] == totals["model_calls"]:
        return f"Tokens: not reported by this model ({calls})"
    line = (f"Tokens: {totals['input']:,} in + {totals['output']:,} out = {totals['total']:,} "
            f"(thinking {totals['thinking']:,} of the out), {calls}")
    if totals["not_reported"]:
        line += f", {totals['not_reported']} call(s) reported no usage"
    return line


def token_budget(max_tokens: int = TOKEN_BUDGET):
    """A middleware that stops the agent before a model call once max_tokens are used up."""

    @before_model(can_jump_to=["end"])
    def check_token_budget(state, runtime):
        used = meter(state["messages"])["total"]
        if used < max_tokens:
            return None
        notice = AIMessage(content=f"Stopped: the token budget of {max_tokens:,} was used up ({used:,} tokens).",
                           name="token_budget")
        return {"messages": [notice], "jump_to": "end"}

    return check_token_budget


# ---------------------------------------------------------------------------
# Save a run, and show it call by call
# ---------------------------------------------------------------------------

def save_run(question: str, messages, path=None) -> Path:
    path = Path(path or LAST_RUN)
    path.parent.mkdir(exist_ok=True)
    data = {"question": question, "messages": messages_to_dict(messages)}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_run(path=None):
    data = json.loads(Path(path or LAST_RUN).read_text(encoding="utf-8"))
    return data["question"], messages_from_dict(data["messages"])


def per_call_table(messages) -> list[str]:
    """One line per model call: what it asked for, and its tokens."""
    lines = [f"{'call':>4}  {'asked for':<34} {'in':>7} {'out':>6} {'thinking':>9} {'total':>7}"]
    number = 0
    for message in messages:
        if message.type != "ai" or is_cap_notice(message):
            continue
        number += 1
        asked = ", ".join(c["name"] for c in message.tool_calls) or "final answer"
        usage = usage_of(message)
        if usage is None:
            lines.append(f"{number:>4}  {asked:<34} {'not reported':>32}")
        else:
            lines.append(f"{number:>4}  {asked:<34} {usage['input']:>7,} {usage['output']:>6,} "
                         f"{usage['thinking']:>9,} {usage['total']:>7,}")
    return lines


def main(path=None) -> int:
    path = Path(path or LAST_RUN)
    if not path.exists():
        print("No saved run yet. Ask the agent something first, for example:  python ask.py onions")
        return 1
    question, messages = load_run(path)
    print(f"Last run: {question}")
    for line in per_call_table(messages):
        print(line)
    print(format_meter(meter(messages)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
