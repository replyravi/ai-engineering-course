"""
scorecard.py - rank models with a weighted decision matrix (steps 8, 9 and 13).

How a model is chosen, in three rounds:
  1. Must-haves (weights.toml [must_have]): licence, privacy, tools, and the worst quality,
     speed and cost you can accept. A model that fails one is dropped. No score can save it.
  2. Criterion scores from YOUR measurements (results/*.json), each from 0 to 1:
       quality  0 = your minimum pass rate,      1 = every answer passes
       speed    0 = your slowest acceptable,     1 = instant
       cost     0 = your highest acceptable,     1 = free
  3. Weighted total = the scores mixed with your weights (a profile in weights.toml).

It writes reports/scorecard.md (the ranked table with licence and data-use columns)
and reports/answers.html (every answer, escaped so nothing in it can run in a browser).

Run:  python scorecard.py                                    (the profile named in weights.toml)
      python scorecard.py --profile cost_first               (step 9)
      python scorecard.py --compare quality_first cost_first (step 9: why did the ranking change?)
      python scorecard.py --practice                         (no key needed: invented practice data)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import argparse
import json
import statistics
import tomllib
from datetime import date, datetime, timedelta
from pathlib import Path

from run_eval import RESULTS_DIR, by_tag, load_results
from safety_gate import escape_for_html

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"        # git ignores this folder
PRACTICE_DIR = HERE / "practice"
WEIGHTS_FILE = HERE / "weights.toml"

CRITERIA = ("quality", "speed", "cost")
CARD_FIELDS = ("id", "weights", "licence", "business_use", "used_to_improve_products", "data_use", "source", "checked")
CARD_STALE_DAYS = 180      # re-check a model card after about six months (as in Episode 04)
PRICE_STALE_DAYS = 90      # prices change faster: re-check after about three months
RETEST_AFTER_DAYS = 90     # a scorecard is a snapshot: run it again later


# ---------------------------------------------------------------------------
# 1. Read the files
# ---------------------------------------------------------------------------

def load_all_results(folder) -> list[dict]:
    return [load_results(path) for path in sorted(Path(folder).glob("*.json"))]


def load_cards(path) -> tuple[dict, list[str]]:
    """{model id: card}, plus a list of problems (missing fields)."""
    cards, problems = {}, []
    for card in json.loads(Path(path).read_text(encoding="utf-8")):
        missing = [field for field in CARD_FIELDS if field not in card]
        if missing:
            problems.append(f"{card.get('id', '?')}: missing {', '.join(missing)}")
            continue
        cards[card["id"]] = card
    return cards, problems


def load_toml(path) -> dict:
    with open(path, "rb") as f:          # tomllib is part of Python 3.11+, nothing to install
        return tomllib.load(f)


def profile_weights(weights_file: dict, name: str) -> dict:
    """The weights of one profile as fractions that add up to 1: {'quality': 0.6, ...}."""
    profiles = weights_file.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"no profile named {name!r} in weights.toml (there are: {', '.join(profiles)})")
    raw = {c: float(profiles[name].get(c, 0)) for c in CRITERIA}
    if any(w < 0 for w in raw.values()) or sum(raw.values()) == 0:
        raise ValueError(f"profile {name!r}: weights must be zero or more, and not all zero")
    total = sum(raw.values())
    return {c: w / total for c, w in raw.items()}


def describe_weights(name: str, weights: dict) -> str:
    return f"{name} (" + ", ".join(f"{c} {weights[c]:.0%}" for c in CRITERIA) + ")"


# ---------------------------------------------------------------------------
# 2. Measure each model from its result file
# ---------------------------------------------------------------------------

def measure(result: dict, price: dict | None) -> dict:
    """Quality, speed, tokens and estimated cost of one model, from its result rows."""
    rows = result["rows"]
    ok = [row for row in rows if not row.get("error")]
    scored = [row for row in ok if row["passed"] is not None]
    passed = sum(1 for row in scored if row["passed"])
    tokens_in = [row["input_tokens"] for row in ok if row.get("input_tokens") is not None]
    tokens_out = [row["output_tokens"] for row in ok if row.get("output_tokens") is not None]
    ttft = [row["ttft_ms"] for row in ok if row.get("ttft_ms") is not None]

    m = {
        "model": result["meta"]["model"],
        "provider": result["meta"]["provider"],
        "judged_by": result["meta"].get("judged_by"),
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "waiting": sum(1 for row in ok if row["passed"] is None),
        "passed": passed,
        "scored": len(scored),
        "quality": passed / len(scored) if scored else None,
        "median_ms": statistics.median(row["latency_ms"] for row in ok) if ok else None,
        "median_ttft_ms": statistics.median(ttft) if ttft else None,
        "avg_in": sum(tokens_in) / len(tokens_in) if tokens_in else None,
        "avg_out": sum(tokens_out) / len(tokens_out) if tokens_out else None,
        "by_tag": by_tag(rows),
    }
    m["cost_per_1000"] = cost_per_1000(m["avg_in"], m["avg_out"], price)
    return m


def cost_per_1000(avg_in, avg_out, price: dict | None) -> float | None:
    """Estimated US dollars for 1,000 questions like ours, at paid-tier prices."""
    if price is None or avg_in is None or avg_out is None:
        return None
    per_question = (avg_in * price["input_per_million"] + avg_out * price["output_per_million"]) / 1_000_000
    return per_question * 1000


# ---------------------------------------------------------------------------
# 3. Round 1: must-haves.  Round 2: criterion scores.  Round 3: weighted total.
# ---------------------------------------------------------------------------

def must_have_problems(m: dict, card: dict | None, must: dict) -> list[str]:
    """Why this model is dropped. An empty list means it stays in the race."""
    if card is None:
        return ["no card in model_cards.json: check its licence and data use first"]
    problems = []
    tier = must.get("tier", "free")
    if must.get("private_data") and card["used_to_improve_products"].get(tier, True):
        problems.append(f"on the {tier} tier the provider may use your prompts, and your data is private")
    if must.get("business_use") and card["business_use"] == "research only":
        problems.append("the licence allows research use only")
    if must.get("needs_tools") and not card.get("tools"):
        problems.append("no tool calling")
    if m["quality"] is None:
        problems.append("no scored answers yet")
    elif m["quality"] < must.get("min_quality", 0):
        problems.append(f"quality {m['quality']:.0%} is below your minimum of {must['min_quality']:.0%}")
    if m["median_ms"] is None:
        problems.append("no successful answers")
    elif "max_median_ms" in must and m["median_ms"] > must["max_median_ms"]:
        problems.append(f"median {m['median_ms']:,.0f} ms is slower than your limit of {must['max_median_ms']:,} ms")
    if m["cost_per_1000"] is None:
        problems.append("no price in prices.toml or no token counts (use 0 for a local model)")
    elif "max_cost_per_1000" in must and m["cost_per_1000"] > must["max_cost_per_1000"]:
        problems.append(f"${m['cost_per_1000']:.2f} per 1,000 questions is above your limit of ${must['max_cost_per_1000']:.2f}")
    return problems


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def criterion_scores(m: dict, must: dict) -> dict:
    """Each criterion from 0 (just acceptable) to 1 (perfect), on YOUR scale from weights.toml."""
    min_quality = must.get("min_quality", 0)
    return {
        "quality": clamp((m["quality"] - min_quality) / (1 - min_quality)) if min_quality < 1 else float(m["quality"] == 1),
        "speed": clamp(1 - m["median_ms"] / must.get("max_median_ms", 10000)),
        "cost": clamp(1 - m["cost_per_1000"] / must.get("max_cost_per_1000", 5.0)),
    }


def weighted_total(scores: dict, weights: dict) -> float:
    return sum(weights[c] * scores[c] for c in CRITERIA)


def rank(candidates: list[dict], weights: dict) -> list[tuple[str, float]]:
    """[(model, total), ...] best first. Ties are broken by name, so the order never jumps around."""
    totals = [(c["model"], weighted_total(c["scores"], weights)) for c in candidates]
    return sorted(totals, key=lambda pair: (-round(pair[1], 9), pair[0]))


def build_candidates(results: list[dict], cards: dict, prices: dict, must: dict) -> tuple[list[dict], list[dict]]:
    """Measure every model. Returns (kept, dropped); dropped ones carry their reasons."""
    kept, dropped = [], []
    for result in results:
        model = result["meta"]["model"]
        m = measure(result, prices.get(model))
        card = cards.get(model)
        problems = must_have_problems(m, card, must)
        entry = {"model": model, "metrics": m, "card": card, "problems": problems}
        if problems:
            dropped.append(entry)
        else:
            entry["scores"] = criterion_scores(m, must)
            kept.append(entry)
    return kept, dropped


# ---------------------------------------------------------------------------
# 4. Explain a change of ranking in one sentence (step 9)
# ---------------------------------------------------------------------------

def versus(criterion: str, m1: dict, m2: dict) -> str:
    """'pass rate 90% against 70%' - the measured values behind one criterion, for two models."""
    if criterion == "quality":
        return f"pass rate {m1['quality']:.0%} against {m2['quality']:.0%}"
    if criterion == "speed":
        return f"median {m1['median_ms']:,.0f} ms against {m2['median_ms']:,.0f} ms"
    return f"${m1['cost_per_1000']:.2f} against ${m2['cost_per_1000']:.2f} per 1,000 questions"


def explain_change(candidates: list[dict], name_a: str, weights_a: dict, name_b: str, weights_b: dict) -> str:
    """One sentence: does the leader change between two profiles, and why?"""
    by_model = {c["model"]: c for c in candidates}
    ranking_a, ranking_b = rank(candidates, weights_a), rank(candidates, weights_b)
    if not ranking_a:
        return "No model passes your must-haves, so there is nothing to rank."
    if len(ranking_a) == 1:
        return f"Only {ranking_a[0][0]} passes the must-haves, so no weights can change the ranking."

    lead_a, lead_b = ranking_a[0][0], ranking_b[0][0]
    def score(model, c):
        return by_model[model]["scores"][c]

    def vs(c, first, second):
        return versus(c, by_model[first]["metrics"], by_model[second]["metrics"])

    if lead_a == lead_b:
        rival = ranking_b[1][0]
        better = [c for c in CRITERIA if score(rival, c) > score(lead_a, c)]
        if not better:
            return (f"The ranking holds: {lead_a} leads with both profiles. It is at least as good as {rival} "
                    f"on quality, speed and cost, so no weights can put {rival} first.")
        c = max(better, key=lambda c: weights_b[c])
        totals = dict(ranking_b)
        return (f"The ranking holds: {lead_a} leads with both profiles. {rival} is better on {c} "
                f"({vs(c, rival, lead_a)}), but with {name_b} {c} counts for "
                f"{weights_b[c]:.0%} of the score, not enough to close the gap "
                f"({totals[lead_a]:.2f} against {totals[rival]:.2f}).")

    a_better = [c for c in CRITERIA if score(lead_a, c) > score(lead_b, c)]
    b_better = [c for c in CRITERIA if score(lead_b, c) > score(lead_a, c)]
    if not a_better or not b_better:
        return f"The ranking changes from {lead_a} to {lead_b}, because their totals are almost equal."
    ca = max(a_better, key=lambda c: weights_a[c])
    cb = max(b_better, key=lambda c: weights_b[c] - weights_a[c])
    return (f"The ranking flips. With {name_a}, {ca} counts for {weights_a[ca]:.0%} and {lead_a} is better there "
            f"({vs(ca, lead_a, lead_b)}). With {name_b}, {cb} counts for {weights_b[cb]:.0%} "
            f"instead of {weights_a[cb]:.0%}, and {lead_b} is better there "
            f"({vs(cb, lead_b, lead_a)}), so {lead_b} moves to first.")


# ---------------------------------------------------------------------------
# 5. Notes: everything that makes the comparison less fair or less fresh
# ---------------------------------------------------------------------------

def notes_for(results: list[dict], cards: dict, prices: dict, today: date) -> list[str]:
    notes = []
    counts = sorted({len(r["rows"]) for r in results})
    if len(counts) > 1:
        notes.append(f"The result files cover different numbers of questions ({', '.join(map(str, counts))}). "
                     "Compare like with like: run run_eval.py without --first for every model.")
    judges = {r["meta"].get("judged_by") for r in results if r["meta"].get("judged_by")}
    if len(judges) > 1:
        notes.append(f"Different judges marked the open answers ({', '.join(sorted(judges))}). Use one judge for all files.")
    for r in results:
        meta = r["meta"]
        waiting = sum(1 for row in r["rows"] if row["passed"] is None and not row.get("error"))
        if waiting:
            notes.append(f"{meta['model']}: {waiting} open answers are waiting for the judge and are not counted. "
                         f"Run: python judge.py")
        if meta.get("judged_by") == f"{meta['provider']}:{meta['model']}":
            notes.append(f"{meta['model']} judged its own open answers (self-preference bias).")
    for model, card in cards.items():
        if any(r["meta"]["model"] == model for r in results):
            if card["business_use"] == "check the terms":
                notes.append(f"{model}: read the terms before business use ({card['licence']}).")
            age = (today - date.fromisoformat(str(card["checked"]))).days
            if age > CARD_STALE_DAYS:
                notes.append(f"{model}: the model card was checked {age} days ago. Check it again.")
    for model, price in prices.items():
        if not any(r["meta"]["model"] == model for r in results):
            continue
        if price.get("valid_until") and today > price["valid_until"]:
            notes.append(f"{model}: the price expired on {price['valid_until']}. Check {price['source']}")
        elif (today - price["checked"]).days > PRICE_STALE_DAYS:
            notes.append(f"{model}: the price was checked on {price['checked']}. Check {price['source']}")
        if price.get("note"):
            notes.append(f"{model}: {price['note']}")
    return notes


# ---------------------------------------------------------------------------
# 6. Reports
# ---------------------------------------------------------------------------

def money(value) -> str:
    return "-" if value is None else f"${value:.2f}"


def ms(value) -> str:
    return "-" if value is None else f"{value:,.0f}"


def tokens_text(m: dict) -> str:
    if m["avg_in"] is None or m["avg_out"] is None:
        return "-"
    return f"{m['avg_in']:,.0f} / {m['avg_out']:,.0f}"


def free_tier_use(card: dict | None, tier: str) -> str:
    if card is None:
        return "unknown"
    return "may be used to improve products" if card["used_to_improve_products"].get(tier, True) else "not used"


def markdown_report(kept, dropped, ranking, profile_text, must, notes, results, today, practice=False) -> str:
    by_model = {c["model"]: c for c in kept}
    tier = must.get("tier", "free")
    lines = ["# Practice scorecard (invented data, not a real model)" if practice else "# Model scorecard", ""]
    lines.append(f"Made by scorecard.py on {today.isoformat()} from {len(results)} result file(s). "
                 f"Profile: {profile_text}.")
    lines.append(f"Re-test by {(today + timedelta(days=RETEST_AFTER_DAYS)).isoformat()}, "
                 "or sooner if a provider changes a model or a price.")
    lines += ["", "## Ranking", "",
              "| Rank | Model | Quality | Median latency (ms) | Median first token (ms) | Tokens in / out per question "
              f"| Est. cost per 1,000 questions (paid tier) | Open or closed | Licence | Data use ({tier} tier) | Score |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for place, (model, total) in enumerate(ranking, start=1):
        c = by_model[model]
        m, card = c["metrics"], c["card"]
        lines.append(f"| {place} | {model} | {m['passed']}/{m['scored']} ({m['quality']:.0%}) | {ms(m['median_ms'])} "
                     f"| {ms(m['median_ttft_ms'])} | {tokens_text(m)} | {money(m['cost_per_1000'])} | {card['weights']} "
                     f"| {card['licence']} | {free_tier_use(card, tier)} | {total:.2f} |")
    if not ranking:
        lines.append("| - | no model passes the must-haves | | | | | | | | | |")

    lines += ["", "## Criterion scores (0 = just acceptable, 1 = perfect)", "", "| Model | Quality | Speed | Cost |",
              "|---|---|---|---|"]
    for model, _ in ranking:
        s = by_model[model]["scores"]
        lines.append(f"| {model} | {s['quality']:.2f} | {s['speed']:.2f} | {s['cost']:.2f} |")

    everyone = kept + dropped
    tags = list(dict.fromkeys(tag for c in everyone for tag in c["metrics"]["by_tag"]))
    lines += ["", "## Quality by tag", "", "| Tag | " + " | ".join(c["model"] for c in everyone) + " |",
              "|---|" + "---|" * len(everyone)]
    for tag in tags:
        cells = []
        for c in everyone:
            t = c["metrics"]["by_tag"].get(tag)
            cells.append(f"{t['passed']}/{t['scored']}" + (f" (+{t['waiting']} not judged)" if t["waiting"] else "")
                         if t else "-")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    lines += ["", "## Must-haves", ""]
    lines.append(f"Tier {tier}; private data: {'yes' if must.get('private_data') else 'no'}; "
                 f"business use needed: {'yes' if must.get('business_use') else 'no'}; "
                 f"tools needed: {'yes' if must.get('needs_tools') else 'no'}; quality at least "
                 f"{must.get('min_quality', 0):.0%}; median at most {must.get('max_median_ms', 10000):,} ms; "
                 f"cost at most ${must.get('max_cost_per_1000', 5.0):.2f} per 1,000 questions.")
    lines.append("")
    if dropped:
        for c in dropped:
            lines.append(f"- Dropped **{c['model']}**: " + "; ".join(c["problems"]))
    else:
        lines.append("- No model was dropped.")

    lines += ["", "## Notes", ""] + [f"- {note}" for note in notes or ["None."]]
    lines += ["", "## Sources", ""]
    for c in everyone:
        if c["card"]:
            lines.append(f"- {c['model']}: model card {c['card']['source']} (checked {c['card']['checked']})")
    lines.append("")
    return "\n".join(lines)


def html_answers(results: list[dict], title: str) -> str:
    """Every answer of every model in one HTML page. EVERY piece of model text goes through escape_for_html."""
    models = [r["meta"]["model"] for r in results]
    answers = {r["meta"]["model"]: {row["id"]: row for row in r["rows"]} for r in results}
    ids = list(dict.fromkeys(row["id"] for r in results for row in r["rows"]))
    questions = {row["id"]: row["question"] for r in results for row in r["rows"]}
    head = "".join(f"<th>{escape_for_html(model)}</th>" for model in models)
    body = []
    for item_id in ids:
        cells = []
        for model in models:
            row = answers[model].get(item_id)
            if row is None:
                cells.append("<td>-</td>")
                continue
            mark = {True: "PASS", False: "FAIL", None: "not judged"}[row["passed"]]
            cells.append(f"<td><b>{mark}</b><br>{escape_for_html(row.get('answer') or row.get('error') or '')}</td>")
        body.append(f"<tr><td><b>{escape_for_html(item_id)}</b><br>{escape_for_html(questions[item_id])}</td>"
                    + "".join(cells) + "</tr>")
    return ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>" + escape_for_html(title) + "</title>\n"
            "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}"
            "td,th{border:1px solid #bbb;padding:6px;vertical-align:top;max-width:420px}</style></head>\n<body>\n"
            f"<h1>{escape_for_html(title)}</h1>\n<table>\n<tr><th>Question</th>{head}</tr>\n"
            + "\n".join(body) + "\n</table>\n</body></html>\n")


# ---------------------------------------------------------------------------
# 7. Terminal output and main
# ---------------------------------------------------------------------------

def print_ranking(kept, ranking, profile_text, out=print) -> None:
    by_model = {c["model"]: c for c in kept}
    out(f"Ranking with {profile_text}")
    out(f"  {'rank':<5}{'model':<24}{'quality':>14}{'median ms':>11}{'$ per 1,000 Q':>15}{'score':>8}")
    for place, (model, total) in enumerate(ranking, start=1):
        m = by_model[model]["metrics"]
        quality = f"{m['passed']}/{m['scored']} {m['quality']:.0%}"
        out(f"  {place:<5}{model:<24}{quality:>14}{ms(m['median_ms']):>11}{money(m['cost_per_1000']):>15}{total:>8.2f}")
    if not ranking:
        out("  (no model passes your must-haves)")


def print_scores(kept, ranking, out=print) -> None:
    by_model = {c["model"]: c for c in kept}
    out("\nCriterion scores (0 = just acceptable, 1 = perfect)")
    out(f"  {'model':<24}{'quality':>8}{'speed':>8}{'cost':>8}")
    for model, _ in ranking:
        c = by_model[model]
        s = c["scores"]
        out(f"  {c['model']:<24}{s['quality']:>8.2f}{s['speed']:>8.2f}{s['cost']:>8.2f}")


def main(argv=None, today: date | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank models from their result files.")
    parser.add_argument("--profile", help="a profile from weights.toml (default: the one after use =)")
    parser.add_argument("--compare", nargs=2, metavar=("PROFILE_A", "PROFILE_B"), help="explain the change between two profiles")
    parser.add_argument("--practice", action="store_true", help="use the invented practice data in practice/")
    parser.add_argument("--weights", default=str(WEIGHTS_FILE), help="the weights file (default: weights.toml)")
    args = parser.parse_args(argv)
    today = today or date.today()

    base = PRACTICE_DIR if args.practice else HERE
    results_dir = PRACTICE_DIR / "results" if args.practice else RESULTS_DIR
    results = load_all_results(results_dir)
    if not results:
        print("No result files yet. Run:  python run_eval.py   (or try:  python scorecard.py --practice)")
        return 1

    cards, card_problems = load_cards(base / "model_cards.json")
    for problem in card_problems:
        print(f"FIX  model_cards.json: {problem}")
    if card_problems:
        return 1
    prices = load_toml(base / "prices.toml").get("models", {})
    weights_file = load_toml(args.weights)
    must = weights_file.get("must_have", {})
    try:
        names = args.compare or [args.profile or weights_file.get("use", "quality_first")]
        weights = {name: profile_weights(weights_file, name) for name in names}
    except ValueError as error:
        print(error)
        return 1

    kept, dropped = build_candidates(results, cards, prices, must)
    where = "practice/results" if args.practice else "results"
    if args.practice:
        print("PRACTICE DATA: invented by hand to show the maths. It is not a measurement of any real model.\n")
    print(f"Result files: {len(results)} in {where}/   ({', '.join(str(len(r['rows'])) for r in results)} questions)")

    for name in names:
        print()
        print_ranking(kept, rank(kept, weights[name]), describe_weights(name, weights[name]))
    if kept:
        print_scores(kept, rank(kept, weights[names[0]]))
    print("\nDropped by the must-haves:" + ("" if dropped else " none"))
    for c in dropped:
        print(f"  {c['model']}: " + "; ".join(c["problems"]))

    if args.compare:
        a, b = names
        print("\n" + explain_change(kept, a, weights[a], b, weights[b]))
        return 0

    tier = must.get("tier", "free")
    print("\nLicence and data use")
    for c in kept + dropped:
        card = c["card"]
        if card:
            licence = card["licence"] if len(card["licence"]) <= 44 else card["licence"].split(" (")[0][:44]  # drop a long link
            print(f"  {c['model']:<24}{card['weights']:<8}{licence:<46}{tier} tier: {free_tier_use(card, tier)}")
    notes = notes_for(results, cards, prices, today)
    if notes:
        print("\nNotes")
        for note in notes:
            print(f"  - {note}")

    REPORTS_DIR.mkdir(exist_ok=True)
    prefix = "practice_" if args.practice else ""
    name = names[0]
    md = markdown_report(kept, dropped, rank(kept, weights[name]), describe_weights(name, weights[name]), must, notes,
                         results, today, practice=args.practice)
    (REPORTS_DIR / f"{prefix}scorecard.md").write_text(md, encoding="utf-8")
    title = "Practice answers (invented data)" if args.practice else "Eval answers"
    (REPORTS_DIR / f"{prefix}answers.html").write_text(html_answers(results, title), encoding="utf-8")
    print(f"\nSaved reports/{prefix}scorecard.md and reports/{prefix}answers.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
