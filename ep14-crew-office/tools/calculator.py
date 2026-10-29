"""
tools/calculator.py - the safe calculator from Episode 11, as a CrewAI tool for the research desk.

Episode 11's rule: the model writes the EXPRESSION, Python computes the NUMBER. If a desk agent
wants a number no other tool gave (say, the gap between two returns), it must use this tool,
and the answer goes into the Ledger like every other tool number. A number the model works
out in its head is caught by the claims checker in Step 14.

calculate() and format_number() below are copied unchanged from
ep11-party-planner-chains/calculator.py. We copy instead of import because that file also
imports LangChain, which lives in the other environment (.venv), not in .venv-crewai.
Safety, as in Episode 11: no eval(), only digits, + - * / ( ) . and spaces; numexpr does the maths.

    python -m tools.calculator "12.34 - 3.5"

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import math
import re
import sys

import numexpr
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from notebook import Ledger

MAX_LENGTH = 200              # longer expressions are refused
LARGEST_EXACT = 10**15        # above this, decimal numbers can no longer hold every whole number exactly
ALLOWED = re.compile(r"[0-9+\-*/().\s]+")


class CalculatorError(ValueError):
    """The expression is not allowed or cannot be computed. The message says why."""


# --- copied unchanged from Episode 11 (ep11-party-planner-chains/calculator.py) ---------------

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

# --- end of the copy ------------------------------------------------------------------------


class CalculatorInput(BaseModel):
    expression: str = Field(description="Numbers with + - * / ** and brackets, like '12.34 - 3.5'")


class CalculatorTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "calculator"
    description: str = (
        "Calculate an arithmetic expression exactly, like '12.34 - 3.5' or '(512.3 / 420) * 100'. "
        "Use it for EVERY number that no other tool gave you. Never do maths in your head."
    )
    args_schema: type[BaseModel] = CalculatorInput
    ledger: Ledger = Field(default_factory=Ledger)

    def _run(self, expression: str) -> str:
        try:
            value = calculate(expression)
        except CalculatorError as error:
            return f"Error: {error}"        # returned, not raised: the agent reads it and tries again
        self.ledger.record(self.name, value)
        return format_number(value)


if __name__ == "__main__":
    expression = " ".join(sys.argv[1:]) or "12.34 - 3.5"
    print(f'calculator("{expression}") -> {CalculatorTool().run(expression=expression)}')
