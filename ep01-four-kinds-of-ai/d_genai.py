"""
d_genai.py - Approach D: generative AI, zero-shot.

Gemini was pre-trained by Google on a huge amount of text (self-supervised: guess the
next word), then tuned to follow instructions. We give it ZERO training rows. We just ask.

with_structured_output makes the model fill in a small form (the FeedbackLabel class)
instead of writing free text, so we always get back exactly one of our three labels.

Run:  python d_genai.py      (needs your free Gemini key in the .env file at the repo root)
"""

import csv
import time
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common import HERE, load_and_split
from llm import MissingAPIKeyError, get_llm

RESULTS_FILE = HERE / "outputs" / "gemini_test_labels.csv"


class FeedbackLabel(BaseModel):
    """The form Gemini must fill in: one label, plus a short reason.

    Why this exists: free text is hard to score. with_structured_output
    forces the model to return exactly praise, complaint, or suggestion.

    What it is: a Pydantic model, not a function. label is one of those
    three strings. reason is at most a short explanation.

    Called by: build_classifier (passed to with_structured_output) and
    tests/test_genai.py, which builds FeedbackLabel(...) for the fake model.
    """

    label: Literal["praise", "complaint", "suggestion"] = Field(description="the kind of feedback")
    reason: str = Field(description="why, in at most ten words")


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You sort feedback for Byte Bites, a college canteen.\n"
     "praise: the student is happy with something.\n"
     "complaint: the student is unhappy with something.\n"
     "suggestion: the student asks for, or proposes, a change."),
    ("human", "Feedback: {text}"),
])


def build_classifier(llm=None):
    """Connect the prompt to a chat model that must return a FeedbackLabel.

    Why this exists: classify() should not rebuild the chain by hand, and
    tests need to pass a fake llm instead of calling Gemini.

    What it does: if llm is None, call get_llm() from llm.py. Then
    `prompt | llm.with_structured_output(FeedbackLabel)`. The `|` is a
    LangChain chain: text in, a FeedbackLabel object out.

    Called by: classify (below). main() gets the llm first and passes it in,
    so get_llm runs once, not once per sentence.
    """
    if llm is None:
        llm = get_llm()
    return prompt | llm.with_structured_output(FeedbackLabel)


def classify(texts, llm=None) -> list:
    """Ask the model to label each text. Return (FeedbackLabel, milliseconds) pairs.

    Why this exists: approach D has zero training rows. We just ask.
    compare.four_answers and the challenge in CHALLENGE.md reuse this
    instead of writing a second Gemini prompt.

    What it does: build_classifier(llm), then chain.invoke({"text": text})
    once per sentence. The milliseconds are the round trip, not training time.

    Called by: main (below, the 10 test rows), compare.four_answers
    (one sentence), and tests/test_genai.py (with fake_llm).
    compare.show_scoreboard does not call this. It reads the saved CSV.
    """
    classifier = build_classifier(llm)
    results = []
    for text in texts:
        start = time.perf_counter()
        answer = classifier.invoke({"text": text})
        results.append((answer, (time.perf_counter() - start) * 1000))
    return results


def save_results(test, results, path=RESULTS_FILE) -> None:
    """Write Gemini's labels to outputs/gemini_test_labels.csv.

    Why this exists: the free Gemini quota should be spent once.
    compare.saved_gemini_score reads this file and does not call the model again.

    What it does: one CSV row per test sentence, with id, text, human label,
    Gemini's label, reason, and milliseconds.

    Called by: main (below).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label", "gemini", "reason", "ms"])
        for (_, row), (answer, ms) in zip(test.iterrows(), results):
            writer.writerow([row["id"], row["text"], row["label"], answer.label, answer.reason, f"{ms:.0f}"])


def main(llm=None, results_file=RESULTS_FILE) -> int:
    """Label the 10 hidden test rows with Gemini and save the answers.

    Why this exists: this is the script entry for `python d_genai.py`.
    Training rows used: 0. The human labels are only the answer key.

    What it does: get_llm(), load_and_split(), classify(test texts),
    print right/wrong, save_results(). Returns 1 if the key is missing
    or LLM_PROVIDER is not a known name. Returns 0 after a successful run.
    Tests pass llm=fake_llm(...) so no key and no internet are needed.

    Called by: the bottom of this file, and tests/test_genai.py.
    """
    if llm is None:
        try:
            llm = get_llm()
        except (MissingAPIKeyError, ValueError) as error:
            print(f"\n{str(error).strip()}")   # friendly help (no key, or a typo in LLM_PROVIDER)
            return 1

    train, test = load_and_split()
    model_name = getattr(llm, "model", None) or type(llm).__name__
    print(f"{model_name} labels the {len(test)} test rows. Training rows used: 0\n")

    results = classify(test["text"], llm)

    print(f"  {'id':>3}  {'answer key':<11} {'model':<11} reason")
    right = 0
    for (_, row), (answer, ms) in zip(test.iterrows(), results):
        mark = "" if answer.label == row["label"] else "   <- miss"
        right += answer.label == row["label"]
        print(f"  {row['id']:>3}  {row['label']:<11} {answer.label:<11} {answer.reason}{mark}")

    ms_per_row = sum(ms for _, ms in results) / len(results)
    print(f"\nZero-shot accuracy on the {len(test)} test rows: {right / len(test):.0%} ({right} of {len(test)})")
    print(f"Time: {ms_per_row:.0f} ms per row (a round trip over the internet)")

    save_results(test, results, results_file)
    print(f"Saved {results_file.name} in outputs/ (compare.py reuses it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
