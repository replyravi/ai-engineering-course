"""
common.py - small things every file of our prompt lab shares.

  - where the data, prompt and result files are
  - load_notes() / get_note(): read data/trip_notes.jsonl (your ten notes and their answers)
  - make_llm(): the model from llm.py, with an optional Gemini thinking level
  - friendly_error(): a failed model call explained in two lines, never a long traceback

You do not run this file. The other files import it.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import json
import os
from pathlib import Path

from pydantic import ValidationError

from llm import get_llm, get_model_name, get_provider
from models import TripNote

HERE = Path(__file__).resolve().parent
NOTES_FILE = HERE / "data" / "trip_notes.jsonl"
PROMPTS_DIR = HERE / "prompts"
RESULTS_DIR = HERE / "results"          # made when you run the lab (git ignores it)


class NotesFileError(ValueError):
    """data/trip_notes.jsonl has a problem (a typo in the JSON, a missing field...)."""


def load_notes(path=NOTES_FILE) -> list[TripNote]:
    """Read every line of the notes file and check it against the TripNote form."""
    path = Path(path)
    notes = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue                                     # blank lines are fine
        try:
            notes.append(TripNote.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as error:
            first_problem = str(error).splitlines()[0:3]
            raise NotesFileError(f"{path.name} line {number}: {' / '.join(first_problem)}") from None
    return notes


def get_note(which, path=NOTES_FILE) -> TripNote:
    """One note by number (1) or by name (coorg)."""
    for note in load_notes(path):
        if str(which).strip().lower() in (str(note.id), note.slug):
            return note
    raise NotesFileError(f"No note called {which!r} in {Path(path).name}. Use a number from 1 to 10 or a name like coorg.")


def make_llm(thinking: str | None = None):
    """The chat model from llm.py. thinking="low" / "medium" / "high" sets Gemini's thinking level.

    Gemini 3.5 Flash-Lite thinks a little by default (minimal). Gemini 3.8 Flash thinks more
    (medium) and does NOT accept "minimal". Other providers ignore this switch here.
    """
    if thinking and get_provider() == "gemini":
        return get_llm(thinking_level=thinking)
    return get_llm()


def model_label() -> str:
    """'gemini / gemini-3.5-flash-lite' - printed on every result, so you know what was measured."""
    try:
        provider = get_provider()
        return f"{provider} / {get_model_name(provider)}"
    except ValueError:
        return "unknown model"


def friendly_error(error: Exception) -> str:
    """A short explanation of a failed model call, with the fix."""
    text = f"{type(error).__name__}: {error}"
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "ollama" and ("Connect" in type(error).__name__ or "refused" in text.lower()):
        return ("Could not reach Ollama. It runs models on your own computer and is not running yet:\n"
                "  install it from https://ollama.com/download, run  ollama pull qwen3.5:4b , then try again.")
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (f"The free-tier limit was reached ({text[:100]}...).\n"
                "Wait a minute and try again, or add --pause 4 to slow down. "
                "Your limits: https://aistudio.google.com/rate-limit")
    if "thinking" in text.lower() and "minimal" in text.lower():
        return f"{text[:160]}\nGemini 3.8 Flash does not accept thinking level minimal. Use low, medium or high."
    return f"The model call failed: {text[:300]}"


def is_rate_limit(error: Exception) -> bool:
    """True for the free-tier 'slow down' error (HTTP 429)."""
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or " 429" in text or text.startswith("429")
