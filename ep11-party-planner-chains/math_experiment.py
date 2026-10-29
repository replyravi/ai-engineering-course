"""
math_experiment.py - the model versus Python on five party sums (Step 11).

We ask the model five sums ALONE (no tools, no calculator) and let Python work out
the same five with calculate() from calculator.py. Then we compare them, to the paisa.

Why can a model get a sum wrong? It writes a number as tokens, small pieces of text,
one after another, each one the most likely next piece. Nothing in that adds the digits.
Some runs get every answer right, some do not, and you cannot tell which from the text.

The 18% GST below is an illustrative rate for practice, not tax advice.

Run:  python math_experiment.py --answers-only    (the questions and Python's answers, no key)
      python math_experiment.py                   (ask the model, then compare)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from calculator import calculate, format_number
from chains import friendly_error
from costing import SAMPLE_ITEMS, load_prices, scale_and_cost
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

TOLERANCE = 0.01          # one paisa


def shopping_list_question() -> dict:
    """Question 1 uses the real line costs of the Step 6 shopping list (12 guests)."""
    bill = scale_and_cost(SAMPLE_ITEMS, 12, load_prices())
    amounts = ", ".join(f"{line.item} {line.cost}" for line in bill.lines)
    return {"label": f"Shopping list total ({len(bill.lines)} lines)",
            "question": f"Add up this shopping list, in rupees: {amounts}. What is the total?",
            "expression": " + ".join(str(line.cost) for line in bill.lines)}


def build_questions() -> list[dict]:
    """Five sums a party host might ask. The expression is what Python computes."""
    return [
        shopping_list_question(),
        {"label": "Caterer quote plus 18% GST",
         "question": "A caterer quotes Rs 48,750 before tax. Add 18% GST. What is the total in rupees?",
         "expression": "48750 * (100 + 18) / 100"},
        {"label": "GST inside a bill",
         "question": "A catering bill of Rs 73,986 already includes 18% GST. "
                     "How many rupees of it are GST?",
         "expression": "73986 * 18 / (100 + 18)"},
        {"label": "Split among 17 guests",
         "question": "Split Rs 73,986 equally among 17 guests. How much does each guest pay, to the paisa?",
         "expression": "73986 / 17"},
        {"label": "Wedding plates",
         "question": "A wedding caterer serves 2,347 plates at Rs 1,869 per plate. "
                     "What is the total in rupees?",
         "expression": "2347 * 1869"},
    ]


PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a quick mental-maths helper. Reply with ONLY the final number: "
               "no working, no words, no currency sign."),
    ("human", "{question}"),
])


def build_chain(llm=None):
    """question -> the model's reply as text. No tools: the model is on its own."""
    if llm is None:
        llm = get_llm()
    return PROMPT | llm | StrOutputParser()


def extract_number(reply: str) -> float | None:
    """The last number in the reply: 'Rs 57,525.00' -> 57525.0. None if there is no number."""
    text = re.sub(r"(?<=\d),(?=\d)", "", reply)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def judge(said: float | None, exact: float) -> str:
    if said is None:
        return "NO NUMBER"
    off = abs(said - exact)
    return "match" if off <= TOLERANCE else f"OFF by {format_number(off)}"


def run_experiment(llm=None) -> list[dict]:
    """Ask the model every question (one batch), compute every answer, compare."""
    questions = build_questions()
    replies = build_chain(llm).batch([{"question": q["question"]} for q in questions])
    rows = []
    for q, reply in zip(questions, replies):
        exact = calculate(q["expression"])
        said = extract_number(reply)
        rows.append({**q, "exact": exact, "reply": reply.strip(), "said": said, "result": judge(said, exact)})
    return rows


def print_answers_only() -> None:
    for number, q in enumerate(build_questions(), start=1):
        print(f"{number}. {q['label']}")
        print(f"   Question:   {q['question']}")
        print(f"   Expression: {q['expression']}")
        print(f"   Python:     {format_number(calculate(q['expression']))}\n")
    print("The 18% GST is an illustrative rate for practice, not tax advice.")


def print_rows(rows: list[dict]) -> None:
    print(f"{'#':<3}{'Sum':<34}{'Python (numexpr)':>18}{'Model said':>16}   Result")
    for number, row in enumerate(rows, start=1):
        said = "-" if row["said"] is None else format_number(row["said"])
        print(f"{number:<3}{row['label']:<34}{format_number(row['exact']):>18}{said:>16}   {row['result']}")
    matches = sum(row["result"] == "match" for row in rows)
    print(f"\nMatches: {matches} of {len(rows)} (to the paisa). This is ONE run of ONE model, not an accuracy rate.")


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="The model versus Python on five party sums.")
    parser.add_argument("--answers-only", action="store_true", help="Python's answers only (no key)")
    args = parser.parse_args(argv)

    if args.answers_only:
        print_answers_only()
        return 0

    try:
        model = llm if llm is not None else get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    if llm is None:
        print(f"Model: {get_model_name()} ({get_provider()}), answering alone with no tools.\n")
    try:
        rows = run_experiment(model)
    except Exception as error:
        print(friendly_error(error))
        return 1
    print_rows(rows)
    print("\nWhat the model actually wrote:")
    for number, row in enumerate(rows, start=1):
        print(f"  {number}. {row['reply'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
