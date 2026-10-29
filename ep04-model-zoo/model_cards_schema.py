"""
model_cards_schema.py - check that model_cards.json is complete and believable (step 10).

A model card is like the label on a food packet: size, ingredients, best-before date,
and who should not use it. Every fact needs a source and the date you checked it,
because model facts go stale fast.

Run:  python model_cards_schema.py                  (checks model_cards.json)
      python model_cards_schema.py other_cards.json   (checks another file)
"""

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from common import HERE, in_millions

CARDS_FILE = HERE / "model_cards.json"
STALE_AFTER_DAYS = 180   # re-check a card that is older than about six months

Family = Literal["encoder", "decoder", "encoder-decoder", "contrastive image-text",
                 "multimodal", "diffusion", "speech-to-text"]


class ModelCard(BaseModel):
    """The facts we keep about one model. Field(...) means: you must fill it in (null is allowed where shown)."""

    name: str = Field(min_length=2)
    id: str = Field(min_length=2, description="what you type in code, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    maker: str = Field(min_length=2)
    family: Family
    generates: bool = Field(description="does it write new text, images or sound?")
    inputs: list[str] = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)
    parameters: int | None = Field(..., gt=0, description="null when the maker does not publish it")
    context_window: int | None = Field(..., gt=0, description="how many tokens it reads at once")
    weights: Literal["open", "closed"]
    licence: str = Field(min_length=2)
    business_use: Literal["yes", "check the terms", "research only"]
    runs_on: str = Field(min_length=2)
    download_mb: int | None = Field(..., gt=0)
    released: str = Field(pattern=r"^\d{4}(-\d{2})?$", description="2024 or 2024-09")
    source: str
    checked: date
    notes: str = ""

    @model_validator(mode="after")
    def facts_fit_together(self):
        if not self.source.startswith("https://"):
            raise ValueError("source must be a web link that starts with https://")
        if self.checked > date.today():
            raise ValueError(f"checked date {self.checked} is in the future")
        if self.weights == "closed" and self.download_mb is not None:
            raise ValueError("closed weights cannot be downloaded, so download_mb must be null")
        if self.weights == "open" and self.download_mb is None:
            raise ValueError("open weights are a file you download: give download_mb")
        return self


def explain(error: ValidationError) -> list[str]:
    """Turn pydantic's error list into short lines like 'weights: Input should be 'open' or 'closed''."""
    lines = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "card"
        message = re.sub(r"^Value error, ", "", item["msg"])
        lines.append(f"{where}: {message}")
    return lines


def check_cards(path=CARDS_FILE) -> tuple[list[ModelCard], list[str]]:
    """Read and check every card. Returns (good cards, problems). No problems = the file is fine."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [], [f"{Path(path).name} is not valid JSON: {error}"]
    if not isinstance(raw, list):
        return [], ["the file must hold a list of cards: [ {...}, {...} ]"]

    cards, problems, seen = [], [], set()
    for number, item in enumerate(raw, start=1):
        label = item.get("name", f"card {number}") if isinstance(item, dict) else f"card {number}"
        try:
            card = ModelCard.model_validate(item)
        except ValidationError as error:
            problems += [f"{label}: {line}" for line in explain(error)]
            continue
        if card.id in seen:
            problems.append(f"{label}: the id {card.id} appears twice")
        seen.add(card.id)
        cards.append(card)
    return cards, problems


def load_cards(path=CARDS_FILE) -> list[ModelCard]:
    """The checked cards. Stops with a clear message when the file has a problem."""
    cards, problems = check_cards(path)
    if problems:
        raise ValueError("model_cards.json has problems:\n  " + "\n  ".join(problems))
    return cards


def is_stale(card: ModelCard, today: date | None = None) -> bool:
    """True when the card was checked more than STALE_AFTER_DAYS ago."""
    return ((today or date.today()) - card.checked).days > STALE_AFTER_DAYS


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else CARDS_FILE
    cards, problems = check_cards(path)

    for card in cards:
        size = in_millions(card.parameters) + " parameters" if card.parameters else "size not published"
        stale = "  <- older than six months, check it again" if is_stale(card) else ""
        print(f"OK   {card.name:<24} {card.family:<23} {card.weights:<7} {size:<25} checked {card.checked}{stale}")
    for problem in problems:
        print(f"FIX  {problem}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) in {path.name}. Fix them and run again.")
        return 1
    print(f"{len(cards)} cards in {path.name}, all passed the schema check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
