"""
common.py - small helpers that every part of Tiffin Toolbox shares.

  - where things live (my_notes/, photos/, outputs/)
  - which three local models we use
  - reading your notes, finding your photos
  - "is this model already downloaded?" and a friendly message when it is not
  - a stopwatch, and how much memory this program uses right now

Import order matters: we import llm.py FIRST. It reads the course .env file, so a line like
HF_HUB_OFFLINE=1 in .env is set before the Hugging Face libraries start (they read it once).
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import llm  # noqa: F401  (reads the root .env before any Hugging Face import)
from llm import DEFAULT_LOCAL_EMBEDDINGS_MODEL

HERE = Path(__file__).resolve().parent
NOTES_DIR = HERE / "my_notes"
PHOTOS_DIR = HERE / "photos"
OUTPUTS_DIR = HERE / "outputs"
PHOTO_TYPES = (".jpg", ".jpeg", ".png", ".webp")

# The three local models of this episode (sizes and licences: python models.py).
CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "").strip() or "Qwen/Qwen2.5-0.5B-Instruct"
VISION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
EMBEDDINGS_MODEL = DEFAULT_LOCAL_EMBEDDINGS_MODEL      # sentence-transformers/all-MiniLM-L6-v2

SAMPLE_NOTE = NOTES_DIR / "jaipur_trip.txt"
SAMPLE_PHOTO = PHOTOS_DIR / "desk.jpg"


@dataclass
class Note:
    name: str     # the file name without .txt, e.g. "jaipur_trip"
    text: str


def load_notes(folder=NOTES_DIR) -> list[Note]:
    """Every .txt file in the folder, sorted by name, as Note(name, text)."""
    notes = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = " ".join(path.read_text(encoding="utf-8").split())   # one line, single spaces
        if text:
            notes.append(Note(path.stem, text))
    return notes


def read_note(path) -> str:
    """The text of one note file, on one line."""
    return " ".join(Path(path).read_text(encoding="utf-8").split())


def find_photos(folder=PHOTOS_DIR) -> list[Path]:
    """Every picture in the folder, sorted by name."""
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in PHOTO_TYPES)


def device() -> str:
    """Where the models run. CPU by default (works everywhere, same answers every run).

    Set LOCAL_DEVICE=mps (Apple Silicon GPU) or LOCAL_DEVICE=cuda (NVIDIA GPU) in .env to try a GPU.
    """
    return os.getenv("LOCAL_DEVICE", "cpu").strip().lower() or "cpu"


def is_downloaded(model_id: str) -> bool:
    """True when the model's config file is already in your Hugging Face cache folder."""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def not_downloaded_help(model_id: str) -> str:
    """What to do when a model is missing and the Hub cannot be reached."""
    return (f"{model_id} is not on this computer yet, and the Hugging Face Hub cannot be reached.\n"
            "Connect to the internet and run the same command once (it downloads the model).\n"
            "If HF_HUB_OFFLINE=1 is set, remove it for that first run.")


def offline_mode() -> bool:
    """True when HF_HUB_OFFLINE=1: Hugging Face only uses models already downloaded."""
    return os.getenv("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


class Stopwatch:
    """with Stopwatch() as watch: ...   then watch.seconds holds how long it took."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False


def memory_mb() -> float:
    """How much memory (RAM) this Python program holds right now, in megabytes."""
    import psutil

    return psutil.Process().memory_info().rss / 1_000_000


def word_count(text: str) -> int:
    return len(text.split())


def shorten(text: str, width: int = 70) -> str:
    """One line, cut with ... when longer than width characters."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."
