"""
zero_shot.py - adapt a foundation model with a prompt only, no training (step 7).

We never show Gemini a single labelled example. We describe the job in the prompt,
and with_structured_output makes it fill in a form (the ReviewLabels class), so we get
back exactly one label per review instead of a paragraph.
All ten reviews go in ONE request, which saves your free-tier quota.

Run:  python zero_shot.py      (needs your free Gemini key in the root .env file)
"""

import json
from collections import Counter
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common import OUTPUTS, console, load_reviews, make_table
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

LABELS_FILE = OUTPUTS / "zero_shot_labels.json"


class ReviewLabel(BaseModel):
    """The sentiment of one numbered review."""

    number: int = Field(description="the review number, as given in the list")
    label: Literal["positive", "negative", "mixed"] = Field(
        description="positive, negative, or mixed (praises one thing and complains about another)")


class ReviewLabels(BaseModel):
    """One label for every review in the list."""

    labels: list[ReviewLabel]


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You label customer reviews of a small tea stall.\n"
     "positive: the customer is happy.\n"
     "negative: the customer is unhappy.\n"
     "mixed: the review praises one thing and complains about another.\n"
     "Give exactly one label for every numbered review."),
    ("human", "Reviews:\n{reviews}"),
])


def numbered(reviews: list[str]) -> str:
    """'1. first review' on the first line, '2. second review' on the next, and so on."""
    return "\n".join(f"{number}. {text}" for number, text in enumerate(reviews, start=1))


def build_labeller(llm=None):
    """prompt -> model that must answer with a ReviewLabels form."""
    if llm is None:
        llm = get_llm()
    return prompt | llm.with_structured_output(ReviewLabels)


def match_labels(reviews: list[str], answer: ReviewLabels) -> list[str]:
    """Put the model's labels in review order, and check that every review got exactly one."""
    by_number = {}
    for item in answer.labels:
        if item.number in by_number:
            raise ValueError(f"review {item.number} got two labels")
        by_number[item.number] = item.label
    expected = set(range(1, len(reviews) + 1))
    missing = sorted(expected - set(by_number))
    unknown = sorted(set(by_number) - expected)
    if missing:
        raise ValueError(f"no label for review(s) {missing}")
    if unknown:
        raise ValueError(f"labels for review numbers that do not exist: {unknown}")
    return [by_number[number] for number in sorted(expected)]


def label_reviews(reviews: list[str], llm=None) -> list[str]:
    """One request for all reviews. Returns one label per review, in order."""
    answer = build_labeller(llm).invoke({"reviews": numbered(reviews)})
    return match_labels(reviews, answer)


def main(llm=None) -> int:
    try:
        provider = get_provider()
        model = get_model_name(provider)
        print(f"Provider: {provider}   Model: {model}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")        # friendly help (no key, or a typo in LLM_PROVIDER)
        return 1

    reviews = load_reviews()
    print(f"Labelling {len(reviews)} reviews in one request. Labelled examples given to the model: 0\n")
    try:
        labels = label_reviews(reviews, llm)
    except ValueError as error:                 # the form came back incomplete or with a wrong label
        print(f"The model's answer did not fit the form ({error}). Run it again.")
        return 1

    rows = [[str(number), label, text] for number, (label, text) in enumerate(zip(labels, reviews), start=1)]
    console.print(make_table(["#", "label", "review"], rows, title="Zero-shot labels"))
    counts = Counter(labels)
    print("Counts:", ", ".join(f"{name} {counts[name]}" for name in ["positive", "negative", "mixed"]))

    OUTPUTS.mkdir(exist_ok=True)
    LABELS_FILE.write_text(json.dumps({"model": model, "labels": labels}, indent=2), encoding="utf-8")
    print(f"Saved outputs/{LABELS_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
