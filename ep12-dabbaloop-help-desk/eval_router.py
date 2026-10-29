"""
eval_router.py - Step 16: a mock drill for the signal box. Route 20 labelled questions and count.

    python eval_router.py                    the semantic router (free, local, no key)
    python eval_router.py --router llm       the LLM router (needs your key: 20 model calls)
    python eval_router.py --router both

Each question in data/router_questions.json has the route a person chose. We print:
  - every question with OK or MISS (and the router's reason for every miss)
  - a confusion table: rows = the right route, columns = what the router picked.
    The diagonal is correct; any other cell shows WHICH routes get mixed up.
  - accuracy = correct / total, the time per question and the model calls it cost

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
import time
from pathlib import Path

from llm import MissingAPIKeyError, get_model_name
from router import ROUTES, CallCounter, build_llm_router

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "data" / "router_questions.json"


def load_questions(path=QUESTIONS_FILE):
    """The labelled questions: [{"question": ..., "route": ...}, ...]"""
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def confusion_matrix(truth, predicted, labels=ROUTES):
    """counts[right_route][picked_route] = how many questions."""
    counts = {row: {column: 0 for column in labels} for row in labels}
    for right, picked in zip(truth, predicted):
        counts[right][picked] += 1
    return counts


def accuracy(truth, predicted):
    """Share of questions routed to the right place (1.0 = all of them)."""
    return sum(t == p for t, p in zip(truth, predicted)) / len(truth)


def format_confusion(counts, labels=ROUTES):
    """The confusion table as text."""
    width = max(len(label) for label in labels) + 2
    lines = [" " * width + "".join(f"{label:>{width}}" for label in labels)]
    for row in labels:
        lines.append(f"{row:<{width}}" + "".join(f"{counts[row][column]:>{width}}" for column in labels))
    return "\n".join(lines)


def run_router(router, questions):
    """Route every question. Returns (results, model_calls). Each result: question, truth, route, ms."""
    counter = CallCounter()
    results = []
    for item in questions:
        start = time.perf_counter()
        route = router.invoke({"question": item["question"]}, config={"callbacks": [counter]})
        results.append({"question": item["question"], "truth": item["route"], "route": route,
                        "ms": (time.perf_counter() - start) * 1000})
    return results, counter.calls


def report(name, results, calls):
    """Print the whole scoreboard for one router. Returns the accuracy."""
    truth = [r["truth"] for r in results]
    picked = [r["route"].destination for r in results]
    print(f"Router: {name}   Questions: {len(results)} ({QUESTIONS_FILE.parent.name}/{QUESTIONS_FILE.name})\n")
    for r in results:
        mark = "OK  " if r["truth"] == r["route"].destination else "MISS"
        print(f"{mark}  {r['truth']:<7} -> {r['route'].destination:<7} {r['question']}")
        if mark == "MISS":
            print(f"      reason: {r['route'].reason}")
    print("\nConfusion table (rows = the right route, columns = the router's pick):")
    print(format_confusion(confusion_matrix(truth, picked)))
    score = accuracy(truth, picked)
    right = sum(t == p for t, p in zip(truth, picked))
    print(f"\nAccuracy: {right}/{len(results)} = {score:.0%}")
    fallbacks = [r["question"] for r in results if r["route"].reason.startswith("fallback")]
    if fallbacks:
        print(f"Sent to chat by the fallback: {len(fallbacks)} ({'; '.join(fallbacks)})")
    ms = sum(r["ms"] for r in results) / len(results)
    print(f"Time: about {ms:.0f} ms per question. Model calls: {calls}")
    return score


def main():
    parser = argparse.ArgumentParser(description="Measure a router on 20 labelled questions.")
    parser.add_argument("--router", choices=["semantic", "llm", "both"], default="semantic")
    args = parser.parse_args()
    questions = load_questions()

    if args.router in ("semantic", "both"):
        from embed_router import SemanticRouter, build_semantic_router

        semantic = SemanticRouter()
        report(f"semantic (local embeddings, {len(semantic.texts)} examples)",
               *run_router(build_semantic_router(router=semantic), questions))
    if args.router in ("llm", "both"):
        if args.router == "both":
            print("\n" + "=" * 70 + "\n")
        try:
            router = build_llm_router()
        except MissingAPIKeyError as error:
            print(str(error).strip())
            raise SystemExit(1)
        report(f"llm ({get_model_name()}, structured output)",
               *run_router(router, questions))


if __name__ == "__main__":
    main()
