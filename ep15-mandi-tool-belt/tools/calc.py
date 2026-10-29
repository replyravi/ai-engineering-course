"""
tools/calc.py - Step 2: the calculator tool, and why brackets matter.

We reuse Episode 11's calculator (copied into tools/ep11_calculator.py): the model writes the
EXPRESSION, Python computes the NUMBER. Its own check lets in only numbers, + - * / ** and
brackets, then numexpr does the maths. We never call eval() on text a model wrote.

To an agent, a tool is four things: a name, a description, an argument schema and a return value.
This file prints exactly what the model is sent, then runs the tool by hand.

    python -m tools.calc                        what the model sees + brackets change the answer (no key)
    python -m tools.calc "3 * 50 * 25.7"        run one expression through the tool

(Run it with -m from the episode folder: tools/ is a package, like a folder of tools on a belt.)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import json
import sys

from langchain_core.utils.function_calling import convert_to_openai_tool

from tools.ep11_calculator import CalculatorError, calculate, calculator, format_number

__all__ = ["calculator", "calculate", "format_number", "CalculatorError"]

# Same numbers, two answers: BODMAS says brackets first, then / and *, then + and -.
PRECEDENCE = [
    ("A 20 kg crate of tomatoes at Rs 28 per kg plus Rs 50 transport, shared by 2 shops",
     "(20 * 28 + 50) / 2", "20 * 28 + 50 / 2"),
    ("An 8.5 km auto ride: Rs 30 for the first 1.5 km, Rs 17 per km after that, Rs 20 waiting",
     "30 + (8.5 - 1.5) * 17 + 20", "30 + 8.5 - 1.5 * 17 + 20"),
]

# Things a model might write that the tool refuses (it returns the reason as text, it never crashes).
REFUSED = ["__import__('os').system('ls')", "round(2.5)", "2 ^ 10", "9**9**9"]


def what_the_model_sees() -> dict:
    """The tool as the model receives it: name, description and a JSON schema of the arguments."""
    return convert_to_openai_tool(calculator)["function"]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        expression = " ".join(argv)
        print(f'calculator("{expression}") -> {calculator.invoke({"expression": expression})}')
        return 0

    print("What the model sees (the tool's name, description and argument schema):")
    print(json.dumps(what_the_model_sees(), indent=2))

    print("\nBrackets change the answer (the tool, run by hand):")
    for story, with_brackets, without in PRECEDENCE:
        print(f"  {story}")
        print(f"    {with_brackets:<28} -> {calculator.invoke({'expression': with_brackets})}")
        print(f"    {without:<28} -> {calculator.invoke({'expression': without})}")

    print("\nRefused (returned as text, so an agent reads the reason and tries again):")
    for expression in REFUSED:
        print(f"  {expression:<32} -> {calculator.invoke({'expression': expression})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
