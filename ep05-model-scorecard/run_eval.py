"""
run_eval.py - give a model your exam and mark it (steps 3, 4, 5 and 7).

For every question in eval_set.jsonl we:
  1. send it to the model and STREAM the answer, so we can time two things:
     the first piece of text (time to first token) and the whole answer (latency)
  2. read the token counts the provider sends back (usage metadata)
  3. mark it: exact match, a JSON check, or "waiting for the judge" for open answers
Then we print a table, a score per tag, and save everything in results/<model>.json.

Run:  python run_eval.py --first 5                (steps 3 and 4: the first five questions)
      python run_eval.py                          (step 5: all 20)
      python run_eval.py --model gemini-3.8-flash (step 7: the same exam for a second model)

Needs your free Gemini key in the .env file at the repo root.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from eval_set import EVAL_FILE, EvalSetError, load_eval_set, short
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"          # git ignores this folder

# The same instruction goes to every model, so the comparison is fair.
SYSTEM = (
    "You are taking a short test. Follow the format each question asks for exactly. "
    "When a question asks for only a number, a name, a time or JSON, reply with only that."
)

STOP_AFTER_ERRORS_IN_A_ROW = 3


# ---------------------------------------------------------------------------
# 1. Ask one question: stream it, time it, count its tokens
# ---------------------------------------------------------------------------

def ask(llm, question: str, clock=time.perf_counter) -> dict:
    """Send one question. Returns the answer, two timings in milliseconds, and the token counts."""
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    start = clock()
    first_text_at = None
    full = None
    for chunk in llm.stream(messages):
        if first_text_at is None and chunk.text:
            first_text_at = clock()                        # time to first token (TTFT)
        full = chunk if full is None else full + chunk     # adding chunks also adds up their token counts
    end = clock()

    usage = (full.usage_metadata if full is not None else None) or {}
    details = usage.get("output_token_details") or {}
    return {
        "answer": full.text if full is not None else "",   # .text, not .content (Gemini 3 sends content blocks)
        "latency_ms": round((end - start) * 1000),
        "ttft_ms": round((first_text_at - start) * 1000) if first_text_at is not None else None,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),       # Gemini counts thinking tokens as output tokens
        "thinking_tokens": details.get("reasoning", 0) if usage else None,
    }


# ---------------------------------------------------------------------------
# 2. Mark one answer
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Make small differences not count: ' 3,330. ' and '3330' both become '3330'."""
    text = text.strip().lower()
    text = text.replace(",", "")                 # 3,330 -> 3330
    text = re.sub(r"[₹$*`\"]", "", text)         # rupee and dollar signs, markdown bold, quotes
    text = text.strip(" .!'")                    # a full stop at the end
    return re.sub(r"\s+", " ", text)


def score_exact(item, reply: str) -> tuple[bool, str]:
    """PASS when the reply equals one of the accepted answers (after normalise)."""
    if normalise(reply) in {normalise(a) for a in item.accepted_answers()}:
        return True, "matches"
    return False, "expected " + " or ".join(item.accepted_answers())


def same_value(got, wanted) -> bool:
    """Compare one JSON value with what we expect. A list of wanted values means 'any of these'."""
    if isinstance(wanted, list):
        return any(same_value(got, w) for w in wanted)
    if isinstance(wanted, bool) or isinstance(got, bool):
        return got is wanted                     # true must be true, not "true" and not 1
    if isinstance(wanted, str):
        return isinstance(got, str) and normalise(got) == normalise(wanted)
    return got == wanted                         # numbers: 3 == 3.0


def score_json(item, reply: str) -> tuple[bool, str]:
    """PASS when the reply is a JSON object (and nothing else) with the expected keys and values."""
    text = reply.strip()
    if text.startswith("```"):
        return False, "wrapped in ``` code fences, so json.loads() fails"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, "not valid JSON"
    if not isinstance(data, dict):
        return False, "valid JSON, but not an object { ... }"
    for key, wanted in item.expect.items():
        if key not in data:
            return False, f'missing key "{key}"'
        if not same_value(data[key], wanted):
            return False, f'"{key}" is {json.dumps(data[key], ensure_ascii=False)}'
    return True, "valid JSON with the expected values"


def score(item, reply: str) -> tuple[bool | None, str]:
    """(True, reason), (False, reason), or (None, ...) for open answers the judge marks in step 6."""
    if item.check == "exact":
        return score_exact(item, reply)
    if item.check == "json":
        return score_json(item, reply)
    return None, "waiting for the judge (step 6)"


# ---------------------------------------------------------------------------
# 3. Run the whole exam
# ---------------------------------------------------------------------------

def is_rate_limit(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def result_mark(row: dict) -> str:
    if row.get("error"):
        return "ERROR"
    return {True: "PASS", False: "FAIL", None: "JUDGE"}[row["passed"]]


def run_eval(items, llm, provider: str, model: str, out=print, pause: float = 0.0, clock=time.perf_counter) -> dict:
    """Ask every question, mark it, and return {"meta": ..., "rows": [...]}."""
    rows, errors_in_a_row = [], 0
    for number, item in enumerate(items, start=1):
        row = {"id": item.id, "tag": item.tag, "check": item.check, "question": item.question}
        try:
            row.update(ask(llm, item.question, clock=clock))
            row["passed"], row["reason"] = score(item, row["answer"])
            row["error"] = None
            errors_in_a_row = 0
        except Exception as error:  # no internet, a wrong model name, the free-tier limit...
            row.update({"answer": "", "passed": None, "reason": "the call failed", "error": str(error)[:300]})
            errors_in_a_row += 1
        rows.append(row)

        note = f"   ({row['reason']})" if row["passed"] is False else ""
        shown = row["error"] if row["error"] else row["answer"]
        out(f"{number:>3}/{len(items)} {item.id:<10} {result_mark(row):<6} {short(shown, 46)}{note}")

        if row["error"] and is_rate_limit(Exception(row["error"])):
            out("\nStopped: the free-tier limit was reached (429 RESOURCE_EXHAUSTED). "
                "Wait a minute and run again, or add --pause 5.")
            break
        if errors_in_a_row >= STOP_AFTER_ERRORS_IN_A_ROW:
            out(f"\nStopped after {errors_in_a_row} failed calls in a row. The last error is shown above.")
            break
        if pause and number < len(items):
            time.sleep(pause)

    meta = {
        "provider": provider,
        "model": model,
        "eval_set": EVAL_FILE.name,
        "items": len(items),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "judged_by": None,
    }
    return {"meta": meta, "rows": rows}


# ---------------------------------------------------------------------------
# 4. Report and save
# ---------------------------------------------------------------------------

def by_tag(rows) -> dict:
    """{tag: {"passed": n, "scored": n, "waiting": n}} in the order the tags first appear."""
    tags = {}
    for row in rows:
        counts = tags.setdefault(row["tag"], {"passed": 0, "scored": 0, "waiting": 0})
        if row.get("error"):
            continue
        if row["passed"] is None:
            counts["waiting"] += 1
        else:
            counts["scored"] += 1
            counts["passed"] += row["passed"]
    return tags


def number_or_dash(value) -> str:
    return "-" if value is None else f"{value:,}"


def print_table(rows, out=print) -> None:
    out("\nPer question (ms = milliseconds, first token = when the first piece of text arrived)")
    out(f"  {'id':<10} {'check':<7} {'result':<6} {'total ms':>9} {'first token ms':>15} "
        f"{'tokens in':>10} {'tokens out':>11} {'of which thinking':>18}")
    for row in rows:
        out(f"  {row['id']:<10} {row['check']:<7} {result_mark(row):<6} {number_or_dash(row.get('latency_ms')):>9} "
            f"{number_or_dash(row.get('ttft_ms')):>15} {number_or_dash(row.get('input_tokens')):>10} "
            f"{number_or_dash(row.get('output_tokens')):>11} {number_or_dash(row.get('thinking_tokens')):>18}")


def print_scores(rows, out=print) -> None:
    tags = by_tag(rows)
    out("\nScore by tag")
    out(f"  {'tag':<18} {'passed':>6} {'scored':>7} {'score':>6}   waiting for the judge")
    total = {"passed": 0, "scored": 0, "waiting": 0}
    for tag, c in tags.items():
        for key in total:
            total[key] += c[key]
        percent = f"{c['passed'] / c['scored']:.0%}" if c["scored"] else "-"
        out(f"  {tag:<18} {c['passed']:>6} {c['scored']:>7} {percent:>6}   {c['waiting'] or ''}")
    percent = f"{total['passed'] / total['scored']:.0%}" if total["scored"] else "-"
    out(f"  {'all scored items':<18} {total['passed']:>6} {total['scored']:>7} {percent:>6}   {total['waiting'] or ''}")
    errors = sum(1 for row in rows if row.get("error"))
    if errors:
        out(f"  ({errors} question(s) failed to run and are not counted)")


def print_medians(rows, out=print) -> None:
    ok = [row for row in rows if not row.get("error")]
    if not ok:
        return
    latency = statistics.median(row["latency_ms"] for row in ok)
    ttft = [row["ttft_ms"] for row in ok if row.get("ttft_ms") is not None]
    line = f"\nMedian per question: {latency:,.0f} ms in total"
    if ttft:
        line += f", {statistics.median(ttft):,.0f} ms to the first token"
    tokens_in = [row["input_tokens"] for row in ok if row.get("input_tokens") is not None]
    tokens_out = [row["output_tokens"] for row in ok if row.get("output_tokens") is not None]
    if tokens_in and tokens_out:
        line += (f"\nAverage tokens per question: {sum(tokens_in) / len(tokens_in):,.0f} in, "
                 f"{sum(tokens_out) / len(tokens_out):,.0f} out")
    else:
        line += "\nTokens: not reported by this model"
    out(line)


def file_name_for(model: str) -> str:
    """'gemini-3.8-flash' -> 'gemini-3.8-flash.json', 'qwen3.5:4b' -> 'qwen3.5-4b.json'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model) + ".json"


def save_results(result: dict, folder=RESULTS_DIR) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name_for(result["meta"]["model"])
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_results(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None, llm=None, results_dir=RESULTS_DIR) -> int:
    parser = argparse.ArgumentParser(description="Run the eval set against one model.")
    parser.add_argument("--first", type=int, help="only the first N questions")
    parser.add_argument("--model", help="another model of the same provider (same as LLM_MODEL in .env)")
    parser.add_argument("--set", default=str(EVAL_FILE), help="the eval set file (default: eval_set.jsonl)")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions")
    args = parser.parse_args(argv)

    if args.model:
        os.environ["LLM_MODEL"] = args.model     # exactly what the LLM_MODEL line in .env does
    try:
        items = load_eval_set(args.set, args.first)
        provider = get_provider()
        model = get_model_name(provider)
        if llm is None:
            llm = get_llm()
    except EvalSetError as error:
        print(f"{error}\nRun:  python eval_set.py   to see every problem.")
        return 1
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")    # friendly help (no key, or a typo in LLM_PROVIDER), not a traceback
        return 1

    print(f"Model: {model} ({provider})   Eval set: {Path(args.set).name}, {len(items)} questions\n")
    result = run_eval(items, llm, provider, model, pause=args.pause)
    rows = result["rows"]
    if all(row.get("error") for row in rows):
        # Never overwrite a good result file with a run in which nothing worked.
        print("\nNothing saved: every call failed. Fix the error shown above and run again.")
        return 1
    print_table(rows)
    print_scores(rows)
    print_medians(rows)
    path = save_results(result, results_dir)
    print(f"\nSaved {path.parent.name}/{path.name}")
    checks = Counter(row["check"] for row in rows)
    if checks.get("rubric"):
        print(f"Next: python judge.py {path.parent.name}/{path.name}   (marks the {checks['rubric']} open answers)")
    return 1 if len(rows) < len(items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
