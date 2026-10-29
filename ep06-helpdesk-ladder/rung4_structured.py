"""
rung4_structured.py - Rung 4: TYPED answers with with_structured_output (Step 9).

Free text is fine for a person, but code needs fields it can trust. We describe the answer
we want as a Pydantic class (a printed form with boxes), and the model must fill it in:

    HelpAnswer(answer="...", source="handbook.md#4", confidence="high")

The provider enforces the form (Gemini uses a JSON schema). If an answer does not fit,
Pydantic raises a clear ValidationError instead of our code guessing.

Run:  python rung4_structured.py
      python rung4_structured.py "Is there a student discount?"
      python rung4_structured.py --show-schema      (the form the model sees, no model call, no key)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from common import HOUSE_RULES, REFUSAL, friendly_error
from llm import MissingAPIKeyError, get_embeddings, get_llm
from rung3_context import QUESTION, format_context, load_paragraphs, rank


class HelpAnswer(BaseModel):
    """A help-desk answer that our code can read field by field."""

    answer: str = Field(description="The reply to the customer, at most three short sentences.")
    source: str = Field(description="The tag of the handbook paragraph the answer comes from, like "
                                    "handbook.md#N (N is the paragraph number). Write none if the handbook "
                                    "does not answer it.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high if one paragraph states the answer directly, medium if you had to combine "
                    "paragraphs, low if the handbook does not answer the question.")


PROMPT = ChatPromptTemplate.from_messages([
    ("system", HOUSE_RULES + "\n\n"
               "Answer using ONLY the handbook paragraphs below. Put the paragraph tag in the source "
               "field, not in the answer. If the paragraphs do not contain the answer, the answer is "
               f"\"{REFUSAL}\", the source is none and the confidence is low.\n\n"
               "Handbook paragraphs:\n{context}"),
    ("human", "{question}"),
])


def build_chain(llm=None):
    """prompt -> model that must return a HelpAnswer (not free text)."""
    if llm is None:
        llm = get_llm()
    return PROMPT | llm.with_structured_output(HelpAnswer)


def ask_typed(question: str, llm=None, embeddings=None, hits=None):
    """Retrieve the top 3 paragraphs (as in rung 3), then get a HelpAnswer. Returns (HelpAnswer, hits)."""
    if hits is None:
        if embeddings is None:
            embeddings = get_embeddings()
        hits = rank(question, load_paragraphs(), embeddings)
    result = build_chain(llm).invoke({"context": format_context(hits), "question": question})
    return result, hits


def check_source(result: HelpAnswer, hits) -> str:
    """Our code can now CHECK the answer: is the source one of the paragraphs we sent?"""
    sent = [p["tag"] for _, p in hits]
    if result.source.strip().lower() == "none":
        return "Source check: none (the handbook does not answer this)"
    if result.source in sent:
        return f"Source check: {result.source} is one of the {len(sent)} paragraphs we sent. OK"
    return f"Source check: WARNING - {result.source} is not one of the paragraphs we sent ({', '.join(sent)})"


def show_schema() -> None:
    """Print the form the model has to fill in, and what happens to a bad answer."""
    print("The form (JSON schema) the model must fill in:")
    print(json.dumps(HelpAnswer.model_json_schema(), indent=2))
    print("\nA badly filled form is rejected, not guessed:")
    try:
        HelpAnswer.model_validate({"answer": "Before 9:30 am.", "confidence": "very sure"})
    except ValidationError as error:
        print(error)


def main(argv=None, llm=None, embeddings=None) -> int:
    parser = argparse.ArgumentParser(description="Rung 4: a typed HelpAnswer instead of free text.")
    parser.add_argument("question", nargs="?", default=QUESTION)
    parser.add_argument("--show-schema", action="store_true", help="print the schema only (no model call)")
    args = parser.parse_args(argv)

    if args.show_schema:
        show_schema()
        return 0

    print(f"Question: {args.question}\n")
    try:
        if embeddings is None:
            embeddings = get_embeddings()
        hits = rank(args.question, load_paragraphs(), embeddings)
        print(f"Context sent: {', '.join(p['tag'] for _, p in hits)}\n")
        result, _ = ask_typed(args.question, llm, hits=hits)
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1

    print(type(result).__name__)                     # HelpAnswer, a Python object, not a string
    for name, value in result.model_dump().items():
        print(f"  {name:<11}: {value}")
    print(f"\n{check_source(result, hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
