"""
make_practice_results.py - how the PRACTICE result files were made (you do not need to run it).

The practice data lets you try scorecard.py without a key and without spending quota:
    python scorecard.py --practice

Everything in practice/ is INVENTED by hand. "practice-model-a" and "practice-model-b" are not
real models, and no AI wrote their answers. The numbers come from a seeded random generator,
so running this file again gives exactly the same files.

    practice-model-a   fast and cheap, but fails some sums and refuses two safe questions
    practice-model-b   slower and dearer (it "thinks" first), strong overall, weak on JSON format

Run (from the episode folder):  python practice/make_practice_results.py

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import json
import random
import sys
from pathlib import Path

PRACTICE = Path(__file__).resolve().parent
sys.path.insert(0, str(PRACTICE.parent))      # so we can import eval_set from the episode folder

from eval_set import load_eval_set  # noqa: E402

MODELS = {
    "practice-model-a": {"fails": {"arith-02", "arith-03", "hi-en-02", "hi-en-04", "refuse-02", "refuse-04"},
                         "ms": (520, 90), "thinking": (0, 0), "seed": 5},
    "practice-model-b": {"fails": {"json-02", "json-04"},
                         "ms": (2900, 450), "thinking": (250, 500), "seed": 8},
}
ANSWER_BY_CHECK_TOKENS = {"exact": (2, 6), "json": (14, 24), "rubric": (12, 48)}


def practice_rows(items, plan: dict) -> list[dict]:
    rng = random.Random(plan["seed"])         # seeded: the same "random" numbers every time
    rows = []
    for item in items:
        latency = max(150, round(rng.gauss(*plan["ms"])))
        thinking = rng.randint(*plan["thinking"]) if plan["thinking"][1] else 0
        visible = rng.randint(*ANSWER_BY_CHECK_TOKENS[item.check])
        passed = item.id not in plan["fails"]
        rows.append({
            "id": item.id, "tag": item.tag, "check": item.check, "question": item.question,
            "answer": "(practice data: no model wrote this)",
            "latency_ms": latency,
            "ttft_ms": round(latency * rng.uniform(0.55, 0.9)),
            "input_tokens": 40 + len(item.question) // 4,     # about 4 characters per token, plus the instructions
            "output_tokens": visible + thinking,
            "thinking_tokens": thinking,
            "passed": passed,
            "reason": "practice data: marked by hand",
            "error": None,
            "judged_by": "practice:hand-marked" if item.check == "rubric" else None,
        })
    return rows


def main() -> None:
    items = load_eval_set(PRACTICE.parent / "eval_set.jsonl")
    folder = PRACTICE / "results"
    folder.mkdir(exist_ok=True)
    for model, plan in MODELS.items():
        result = {
            "meta": {"provider": "practice", "model": model, "eval_set": "eval_set.jsonl", "items": len(items),
                     "run_at": "2026-10-23T00:00:00", "judged_by": "practice:hand-marked",
                     "note": "Invented by hand to show the maths. Not a measurement of any model."},
            "rows": practice_rows(items, plan),
        }
        path = folder / f"{model}.json"
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote practice/results/{path.name}")


if __name__ == "__main__":
    main()
