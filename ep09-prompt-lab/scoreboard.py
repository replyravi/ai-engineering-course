"""
scoreboard.py - measure five prompt techniques on your ten notes (Steps 13 and 14).

One question for every technique: "who owes whom?". A cell is right when the answer leaves
everyone at zero, checked by settle.py against your hand-checked answer key. Same notes,
same check, so the scores can be compared. No eyeballing.

    technique    prompt file           how it answers
    zero-shot    balance_zero_v1       the bare question, free text
    rtc          balance_rtc_v1        role, task, context, rules and a format
    few-shot     balance_fewshot_v1    rtc + two worked examples as message pairs
    cot          balance_cot_v2        step-by-step working in a hidden field, then final_answer
    structured   expenses_v1           typed Expense forms -> guard.py -> settle.py does the maths

Why --repeats instead of a temperature dial? Gemini 3.5 Flash-Lite and newer ignore temperature,
so you cannot "turn the randomness down". Run the same test set a few times and look at what moves.

Every row shows its prompt version, and each prompt's cells are saved to results/<version>.csv.

Run:  python scoreboard.py                              (needs your free Gemini key: 50 model calls)
      python scoreboard.py --repeats 3                  (150 calls: shows which notes vary between runs)
      python scoreboard.py --only zero-shot,structured  (just two rows)
      python scoreboard.py --prompt rtc=balance_rtc_v2  (Step 14: score your new prompt version)
      python scoreboard.py --pause 4                    (wait 4 s between calls if you hit the free-tier limit)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass, field

from langchain_core.callbacks import get_usage_metadata_callback

from common import RESULTS_DIR, friendly_error, is_rate_limit, load_notes, make_llm, model_label
from llm import MissingAPIKeyError
from models import TripNote
from prompt_files import estimate_tokens, read_prompt, note_inputs
from settle import same_result, settle
from techniques import SCHEMAS, Answer, ask_transfers, short

# technique -> (label on the scoreboard, prompt file). Change a prompt with --prompt rtc=balance_rtc_v2.
TECHNIQUES = {
    "zero-shot": ("zero-shot", "balance_zero_v1"),
    "rtc": ("role-task-context", "balance_rtc_v1"),
    "few-shot": ("few-shot", "balance_fewshot_v1"),
    "cot": ("chain of thought", "balance_cot_v2"),
    "structured": ("structured + Python", "expenses_v1"),
}

MARKS = {True: "+", False: "x", None: "?"}


@dataclass
class Row:
    """One technique's results: for each note, one True / False / None per run."""

    technique: str
    prompt: str
    cells: dict[int, list] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def score(self) -> tuple[int, int]:
        """(right, total) over every note and every run. 'No answer' counts as not right."""
        results = [r for runs in self.cells.values() for r in runs]
        return sum(r is True for r in results), len(results)

    def varied(self) -> list[int]:
        """Notes where the runs disagreed (some right, some not)."""
        return [note for note, runs in self.cells.items()
                if any(r is True for r in runs) and not all(r is True for r in runs)]


def estimated_tokens(prompt_name: str, notes: list[TripNote]) -> int:
    """About how many tokens one call sends (messages + the form, if any). A rule of thumb, offline."""
    prompt = read_prompt(prompt_name)
    per_note = [estimate_tokens(prompt.template().format_messages(**note_inputs(n.note))) for n in notes]
    form = len(json.dumps(SCHEMAS[prompt.output].model_json_schema())) // 4 if prompt.output in SCHEMAS else 0
    return round(sum(per_note) / len(per_note)) + form


def check_one(prompt_name: str, note: TripNote, llm) -> tuple[bool | None, Answer, dict]:
    """Ask one note, return (right?, the answer, token usage of that call)."""
    with get_usage_metadata_callback() as usage:
        try:
            answer = ask_transfers(prompt_name, note, llm)
        except Exception as error:
            if is_rate_limit(error):
                raise                     # stop the whole run: more calls would fail too
            answer = Answer(None, error=f"{type(error).__name__}: {str(error)[:120]}")
    tokens = {"in": sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values()),
              "out": sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())}
    right = None if answer.transfers is None else same_result(answer.transfers, note.expenses)
    return right, answer, tokens


def score_technique(technique: str, prompt_name: str, notes: list[TripNote], llm, repeats: int = 1,
                    pause: float = 0, results_dir=RESULTS_DIR) -> Row:
    """Run one technique over every note, `repeats` times, and save the cells to results/<version>.csv."""
    row = Row(technique, read_prompt(prompt_name).version_id)
    records = []
    for run in range(1, repeats + 1):
        for note in notes:
            right, answer, tokens = check_one(prompt_name, note, llm)
            row.cells.setdefault(note.id, []).append(right)
            row.input_tokens += tokens["in"]
            row.output_tokens += tokens["out"]
            row.calls += 1
            records.append({"model": model_label(), "technique": technique, "prompt": row.prompt, "note": note.id,
                            "run": run, "result": {True: "right", False: "wrong", None: "no answer"}[right],
                            "answer": short(answer.transfers), "error": answer.error})
            if pause:
                time.sleep(pause)
    save_csv(records, results_dir / f"{row.prompt}.csv")
    return row


def save_csv(records: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def cell(runs: list, repeats: int) -> str:
    """One run: + right, x wrong, ? no answer. Several runs: how many were right."""
    if repeats == 1:
        return MARKS[runs[0]]
    return str(sum(r is True for r in runs))


def print_table(rows: list[Row], notes: list[TripNote], repeats: int, estimates: dict) -> None:
    ids = [n.id for n in notes]
    header = f"{'technique':<21}{'prompt version':<20}" + "".join(f"{i:>3}" for i in ids) + "   score    tokens/call"
    print(header)
    for row in rows:
        right, total = row.score()
        if row.calls and row.input_tokens:
            tokens = f"{row.input_tokens // row.calls} in, {row.output_tokens // row.calls} out"
        else:
            tokens = f"~{estimates[row.prompt]} in (estimate)"
        cells = "".join(f"{cell(row.cells[i], repeats):>3}" for i in ids)
        print(f"{row.technique:<21}{row.prompt:<20}{cells}   {f'{right}/{total}':<8} {tokens}")
    python_right = sum(same_result(settle(n.expenses).transfers, n.expenses) for n in notes)
    print(f"{'Python check':<21}{'settle(), no model':<20}" + "".join(f"{'+':>3}" for _ in ids)
          + f"   {python_right}/{len(notes)}")
    if repeats == 1:
        print("\n+ right   x wrong   ? no usable answer")
    else:
        print(f"\ncells: how many of the {repeats} runs were right")
        for row in rows:
            print(f"  {row.technique}: varied between runs on notes {row.varied() or 'none'}")


def print_plan(chosen: dict, notes: list[TripNote], repeats: int) -> None:
    """What the scoreboard WILL measure (offline): prompt versions and estimated tokens per call."""
    print(f"{'technique':<21}{'prompt version':<20}{'tokens/call (estimate)':<24}score")
    for technique, (label, prompt_name) in chosen.items():
        version = read_prompt(prompt_name).version_id
        print(f"{label:<21}{version:<20}{'~' + str(estimated_tokens(prompt_name, notes)):<24}needs a live run")
    python_right = sum(same_result(settle(n.expenses).transfers, n.expenses) for n in notes)
    print(f"{'Python check':<21}{'settle(), no model':<20}{'0':<24}{python_right}/{len(notes)}")
    print(f"\nA live run makes {len(chosen) * len(notes) * repeats} model calls.")


def choose(only: str | None, overrides: list[str]) -> dict:
    """The techniques to run, with any --prompt technique=file overrides applied."""
    chosen = dict(TECHNIQUES)
    for override in overrides:
        technique, _, prompt_name = override.partition("=")
        if technique not in chosen or not prompt_name:
            raise ValueError(f"--prompt {override!r}: use one of {', '.join(TECHNIQUES)}, like rtc=balance_rtc_v2")
        read_prompt(prompt_name)                          # fails early with a clear message
        chosen[technique] = (chosen[technique][0], prompt_name)
    if only:
        wanted = [name.strip() for name in only.split(",")]
        unknown = [name for name in wanted if name not in chosen]
        if unknown:
            raise ValueError(f"--only: unknown technique(s) {unknown}. Use: {', '.join(TECHNIQUES)}")
        chosen = {name: chosen[name] for name in wanted}
    return chosen


def main(argv=None, llm=None, results_dir=RESULTS_DIR) -> int:
    parser = argparse.ArgumentParser(description="Score prompt techniques on your ten notes.")
    parser.add_argument("--repeats", type=int, default=1, help="run the whole test set this many times")
    parser.add_argument("--notes", default="all", help="'all' or note numbers like 1,3,8")
    parser.add_argument("--only", help="techniques to run, e.g. zero-shot,structured")
    parser.add_argument("--prompt", action="append", default=[], help="use another prompt file, e.g. rtc=balance_rtc_v2")
    parser.add_argument("--pause", type=float, default=0, help="seconds to wait between calls")
    args = parser.parse_args(argv)

    try:
        chosen = choose(args.only, args.prompt)
        notes = load_notes()
    except ValueError as error:        # a bad option, an unknown prompt file, or a typo in the notes file
        print(error)
        return 1
    if args.notes != "all":
        wanted = {part.strip() for part in args.notes.split(",")}
        notes = [n for n in notes if str(n.id) in wanted or n.slug in wanted]

    print(f"Prompt technique scoreboard: {len(notes)} notes x {args.repeats} run(s), model {model_label()}\n")
    try:
        llm = llm or make_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print_plan(chosen, notes, args.repeats)
        print(str(error).strip())
        return 1

    estimates = {read_prompt(p).version_id: estimated_tokens(p, notes) for _, p in chosen.values()}
    rows = []
    try:
        for technique, (label, prompt_name) in chosen.items():
            print(f"  running {label} ({prompt_name}) ...", flush=True)
            rows.append(score_technique(label, prompt_name, notes, llm, args.repeats, args.pause, results_dir))
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        if not rows:
            return 1
        print("Scores for the techniques that finished:\n")
    print()
    print_table(rows, notes, args.repeats, estimates)
    print(f"\nEvery cell is saved in {results_dir.name}/<prompt version>.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
