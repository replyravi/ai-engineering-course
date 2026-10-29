"""
eval_set.py - load and check your eval set (step 2).

An eval set is your own small exam for a model: questions from YOUR use case, each with
the answer you expect, or a rubric that says what a good answer looks like.
The file has one question per line, written as JSON. That format is called JSON Lines (.jsonl).

Three kinds of check:
    exact   the reply must equal the answer (or one of a list of answers)
    json    the reply must be a JSON object with these keys and values
    rubric  an open answer; a judge model grades it against the rubric (step 6)

Run:  python eval_set.py                            (checks eval_set.jsonl)
      python eval_set.py --first 5                  (only the first five lines)
      python eval_set.py data/broken_eval.jsonl     (see what mistakes look like)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

HERE = Path(__file__).resolve().parent
EVAL_FILE = HERE / "eval_set.jsonl"


class EvalItem(BaseModel):
    """One question of the eval set."""

    id: str = Field(min_length=1, description="a short unique name, e.g. arith-01")
    tag: str = Field(min_length=1, description="the group it belongs to, e.g. arithmetic")
    check: Literal["exact", "json", "rubric"]
    question: str = Field(min_length=5)
    answer: str | list[str] | None = None   # exact: the accepted answer(s)
    expect: dict | None = None              # json: keys and the values they must have
    rubric: str | None = None               # rubric: what a passing answer looks like

    @model_validator(mode="after")
    def has_what_its_check_needs(self):
        needed = {"exact": "answer", "json": "expect", "rubric": "rubric"}[self.check]
        if getattr(self, needed) in (None, "", [], {}):
            raise ValueError(f'check "{self.check}" needs an "{needed}" field')
        return self

    def accepted_answers(self) -> list[str]:
        """The exact answers we accept, always as a list."""
        return [self.answer] if isinstance(self.answer, str) else list(self.answer or [])


class EvalSetError(ValueError):
    """The eval set file has one or more problems."""


def explain(error: ValidationError) -> str:
    """Turn pydantic's error list into one short line."""
    parts = []
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"])
        message = re.sub(r"^Value error, ", "", item["msg"])
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


def check_eval_set(path=EVAL_FILE, first: int | None = None) -> tuple[list[EvalItem], list[str]]:
    """Read and check every line. Returns (good items, problems). No problems = the file is fine."""
    path = Path(path)
    if not path.is_file():
        return [], [f"{path.name} not found (looked in {path.parent})"]

    items, problems, first_line_of = [], [], {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if first is not None and number > first:
            break
        if not line.strip():
            continue  # empty lines are allowed
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {number}: not valid JSON ({error.msg}, column {error.colno})")
            continue
        try:
            item = EvalItem.model_validate(raw)
        except ValidationError as error:
            problems.append(f"line {number}: {explain(error)}")
            continue
        if item.id in first_line_of:
            problems.append(f"line {number}: the id {item.id} is used twice (first on line {first_line_of[item.id]})")
            continue
        first_line_of[item.id] = number
        items.append(item)
    if not items and not problems:
        problems.append(f"{path.name} has no questions yet")
    return items, problems


def load_eval_set(path=EVAL_FILE, first: int | None = None) -> list[EvalItem]:
    """The checked items. Stops with a clear message when the file has a problem."""
    items, problems = check_eval_set(path, first)
    if problems:
        raise EvalSetError(f"{Path(path).name} has problems:\n  " + "\n  ".join(problems))
    return items


def short(text: str, width: int = 52) -> str:
    """One line, cut to `width` characters."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check an eval set file.")
    parser.add_argument("file", nargs="?", default=str(EVAL_FILE), help="the .jsonl file (default: eval_set.jsonl)")
    parser.add_argument("--first", type=int, help="only read the first N lines")
    args = parser.parse_args(argv)

    path = Path(args.file)
    items, problems = check_eval_set(path, args.first)
    for item in items:
        print(f"OK   {item.id:<10} {item.tag:<17} {item.check:<7} {short(item.question)}")
    for problem in problems:
        print(f"FIX  {problem}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) in {path.name}. Fix them and run again.")
        return 1
    tags = Counter(item.tag for item in items)
    checks = Counter(item.check for item in items)
    print(f"{len(items)} items loaded from {path.name}")
    print("  by tag:   " + ", ".join(f"{tag} {n}" for tag, n in tags.items()))
    print("  by check: " + ", ".join(f"{check} {n}" for check, n in checks.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
