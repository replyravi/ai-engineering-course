# ---------------------------------------------------------------------------
# Copied from Episode 11 (ep11-party-planner-chains/calculator.py) so this folder runs on its own.
# Do not edit this copy: tests/test_calc.py checks that it still matches the original.
# Episode 15 uses calculate() and the calculator tool from it (see tools/calc.py).
# Everything below the next line is the original file, unchanged.
# ---------------------------------------------------------------------------
"""
calculator.py - a safe calculator TOOL, and an agent that uses it (Step 12).

A model writes numbers as tokens (pieces of text), so a long sum can come out wrong.
A tool fixes that: the model writes the EXPRESSION, Python computes the NUMBER.

Reuse it in your own projects (Episode 15 does):

    from calculator import calculator                   # the LangChain tool
    calculator.invoke({"expression": "938.40 / 12 * 17"})     # -> "1329.4"

    from calculator import calculate                    # the plain Python function
    calculate("12 * 154")                                     # -> 1848.0

    calculator(expression: str) -> str
        expression  numbers with + - * / // ** and brackets, like "(48750 * 18) / 100".
                    Commas inside numbers are fine ("48,750").
        returns     the result as text, rounded to 6 decimal places ("8775", "4352.117647"),
                    or a message that starts with "Error:". It never raises, so a wrong
                    expression never crashes the agent: the model reads the error and tries again.

    calculate(expression: str) -> float
        the same check and maths, but returns the number and raises CalculatorError.

Safety: we NEVER call eval() on text a model wrote.
  1. Our own check allows ONLY digits, + - * / ( ) . and spaces. No letters, so no names,
     no functions and no Python commands can get in.
  2. Every whole number becomes a decimal (12 -> 12.0). Python then never builds a giant
     whole number like 9**9**9 (about 370 million digits, which keeps the computer busy for
     minutes); as decimals it fails at once with "too big".
  3. numexpr (the library the old LLMMathChain used) does the arithmetic, with no variables
     and no Python built-ins available.

Run:  python calculator.py --tool-only        (what the model sees + the tool run by hand, no key)
      python calculator.py                    (Step 12: the agent answers, with its message trace)
      python calculator.py "your own question"

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
import math
import re

import numexpr
from langchain.agents import create_agent
from langchain.tools import tool

from llm import MissingAPIKeyError, get_llm

MAX_LENGTH = 200              # longer expressions are refused
LARGEST_EXACT = 10**15        # above this, decimal numbers can no longer hold every whole number exactly
ALLOWED = re.compile(r"[0-9+\-*/().\s]+")


class CalculatorError(ValueError):
    """The expression is not allowed or cannot be computed. The message says why."""


# ---------------------------------------------------------------------------
# 1. The plain Python calculator
# ---------------------------------------------------------------------------

def calculate(expression: str) -> float:
    """Compute an arithmetic expression safely. Raises CalculatorError with a clear reason."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("the expression is empty.")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression.strip())   # 48,750 -> 48750
    if len(text) > MAX_LENGTH:
        raise CalculatorError(f"the expression is longer than {MAX_LENGTH} characters.")
    if "%" in text:
        raise CalculatorError("write a percentage as / 100: 18% of 500 is 500 * 18 / 100.")
    if "^" in text:
        raise CalculatorError("write powers with **: 2 to the power 10 is 2**10.")
    if not ALLOWED.fullmatch(text):
        bad = " ".join(sorted(set(re.sub(r"[0-9+\-*/().\s]", "", text))))
        raise CalculatorError(f"only numbers, + - * / ** and brackets are allowed (found: {bad}).")

    as_decimals = re.sub(r"(?<![\d.])(\d+)(?![\d.])", r"\1.0", text)   # 12 -> 12.0, 0.15 stays
    try:
        value = numexpr.evaluate(as_decimals, local_dict={}, global_dict={}).item()
    except ZeroDivisionError:
        raise CalculatorError("division by zero.") from None
    except OverflowError:
        raise CalculatorError("the result is too big.") from None
    except (SyntaxError, ValueError, TypeError, KeyError):
        raise CalculatorError(f"'{expression}' is not a complete calculation.") from None

    if not isinstance(value, float):                 # for example (-8)**0.5 is not a real number
        raise CalculatorError("the result is not a real number.")
    if not math.isfinite(value):
        raise CalculatorError("the result is too big.")
    if abs(value) >= LARGEST_EXACT:
        raise CalculatorError("the result has more than 15 digits, too big to be exact.")
    return value


def format_number(value: float) -> str:
    """1848.0 -> '1848', 4352.1176470588 -> '4352.117647' (6 decimal places at most)."""
    rounded = round(value, 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# 2. The tool: the name, the docstring and the typed argument are what the model sees
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate an arithmetic expression exactly, like '12 * 154' or '(938.40 / 12) * 17'.

    Use only numbers, + - * / ** and brackets. Use brackets to make the order clear.
    Write a percentage as / 100: 18% of 500 is '500 * 18 / 100'.
    Use this for EVERY sum, product, split or percentage. Never do maths in your head.
    """
    try:
        return format_number(calculate(expression))
    except CalculatorError as error:
        return f"Error: {error}"           # returned, not raised: the agent keeps running


# ---------------------------------------------------------------------------
# 3. An agent that uses the tool (the 2026 replacement for LLMMathChain)
# ---------------------------------------------------------------------------

# Rs 938.40 is the sample shopping list for 12 guests from costing.py (Step 6).
QUESTION = ("Our South Indian lunch for 12 guests costs Rs 938.40 in groceries. Five more guests "
            "join and the cost per guest stays the same. What is the new total?")

AGENT_RULES = ("You help a host plan a party. For ANY arithmetic, call the calculator tool, even for "
               "small sums. Never calculate in your head. Answer in one short sentence, in rupees.")

# Our own limit for the loop (LangGraph calls it recursion_limit). A normal answer needs
# model, tool, model; a model that keeps calling the tool is stopped after 5 model calls.
MAX_AGENT_STEPS = 10


def build_math_agent(llm=None):
    """The agent: a model + the calculator tool. Pass llm=fake_llm(...) in tests."""
    if llm is None:
        llm = get_llm()
    return create_agent(model=llm, tools=[calculator], system_prompt=AGENT_RULES)


def ask_agent(question: str, llm=None) -> list:
    """Run the loop once and return every message: human, ai (tool call), tool, ai (answer)."""
    result = build_math_agent(llm).invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"recursion_limit": MAX_AGENT_STEPS},
    )
    return result["messages"]


def print_trace(messages) -> None:
    """Every message of the agent loop, labelled by its type."""
    print(f"Message trace ({len(messages)} messages):")
    for number, m in enumerate(messages, start=1):
        if m.type == "ai" and m.tool_calls:
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)
            print(f"  {number}. ai     asks for a tool: {calls}")
        elif m.type == "tool":
            print(f"  {number}. tool   {m.name} -> {m.text}")
        else:
            print(f"  {number}. {m.type:<6} {m.text}")
    tool_calls = sum(len(m.tool_calls) for m in messages if m.type == "ai")
    model_calls = sum(1 for m in messages if m.type == "ai")
    print(f"Tool calls: {tool_calls}   Model calls: {model_calls}")


TOOL_EXAMPLES = [
    "12 * 154",
    "938.40 / 12 * 17",
    "48,750 * 18 / 100",
    "10 + 5 * 2",
    "(10 + 5) * 2",
    "73986 / 17",
    "18% of 500",
    "1 / 0",
    "9**9**9",
    "__import__('os').system('ls')",
]


def show_tool() -> None:
    """What the model is shown about the tool, then the tool run by hand (no model)."""
    print("What the model sees:")
    print(f"  name:        {calculator.name}")
    first, *rest = [line.strip() for line in calculator.description.splitlines() if line.strip()]
    print(f"  description: {first}")
    for line in rest:
        print(f"               {line}")
    print(f"  arguments:   {json.dumps(calculator.args)}")
    print("\nThe tool, run by hand (our Python, no model):")
    for expression in TOOL_EXAMPLES:
        shown = f'calculator("{expression}")'
        print(f"  {shown:<44} -> {calculator.invoke({'expression': expression})}")


def explain_error(error: Exception) -> str:
    """A short, friendly explanation of a failed run, with the fix."""
    text = f"{type(error).__name__}: {error}"
    if type(error).__name__ == "GraphRecursionError":
        return ("The agent stopped: it used all its steps without a final answer (MAX_AGENT_STEPS in\n"
                "calculator.py). Ask a clearer question, or try LLM_MODEL=gemini-3.8-flash in .env.")
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (f"The free-tier limit was reached ({text[:120]}...).\n"
                "Wait a minute and try again. Your limits: https://aistudio.google.com/rate-limit")
    if "Connect" in type(error).__name__:
        return f"Could not reach the model's server ({text}). Check your internet connection."
    return f"The model call failed: {text}"


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="A safe calculator tool and an agent that uses it.")
    parser.add_argument("question", nargs="?", default=QUESTION)
    parser.add_argument("--tool-only", action="store_true", help="run the tool by hand (no model, no key)")
    args = parser.parse_args(argv)

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
        print(explain_error(error))
        return 1
    print_trace(messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
