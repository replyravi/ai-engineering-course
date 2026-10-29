"""
skills/summarise.py - a one-sentence summary of a note, made on your laptop (step 4).

Two kinds of summary:
  highlighter (extractive)  copies the ONE sentence of the note that best matches the whole note
  writer (abstractive)      the local chat model retells the note in its own words

The highlighter can never invent a fact, but it can miss things. The writer reads better,
but a small model sometimes gets a detail wrong. Always check a summary against the note.

Run (inside ep16-local-ai-toolbox):
    python -m skills.summarise                          (the sample note my_notes/jaipur_trip.txt)
    python -m skills.summarise my_notes/laptop_repair.txt
"""

import re
import sys
from pathlib import Path

import numpy as np

from common import CHAT_MODEL, HERE, SAMPLE_NOTE, Stopwatch, device, is_downloaded, \
    not_downloaded_help, offline_mode, read_note, word_count
from skills.chat_model import Reply, ask_local, chat_model, count_tokens

SYSTEM = ("You summarise personal notes. Reply with ONE short sentence (at most 25 words). "
          "Use only facts from the note.")
MAX_NEW_TOKENS = 60      # a hard limit on the reply, counted in TOKENS, not words


def summarise(text: str) -> Reply:
    """The writer: the local chat model's one-sentence summary."""
    return ask_local(SYSTEM, f"Summarise this note:\n\n{text}", MAX_NEW_TOKENS)


def split_sentences(text: str) -> list[str]:
    """Cut after . ! or ? followed by a space."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def highlight(text: str) -> str:
    """The highlighter: the sentence whose meaning is closest to the whole note (embeddings, step 7)."""
    from skills.search import encode

    sentences = split_sentences(text)
    vectors = encode(sentences + [text])          # last row = the whole note
    scores = vectors[:-1] @ vectors[-1]
    return sentences[int(np.argmax(scores))]


def show_path(path: Path) -> str:
    full = path.resolve()
    return full.relative_to(HERE).as_posix() if full.is_relative_to(HERE) else str(path)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else SAMPLE_NOTE
    if not path.is_file():
        print(f"No note at {path}. Try: python -m skills.summarise my_notes/jaipur_trip.txt")
        return 1
    if offline_mode() and not is_downloaded(CHAT_MODEL):
        print(not_downloaded_help(CHAT_MODEL))
        return 1
    text = read_note(path)

    try:
        first_time = not is_downloaded(CHAT_MODEL)
        if first_time:
            print(f"First run: downloading {CHAT_MODEL} (about 1 GB, only once)...")
        with Stopwatch() as load:
            chat_model()                                   # load (or download) the model
        print(f"Note: {show_path(path)} ({word_count(text)} words, {count_tokens(text)} tokens for the chat model)\n")

        print("Highlighter (extractive: copies the best sentence)")
        print(f'  "{highlight(text)}"\n')

        reply = summarise(text)
    except OSError:
        print(not_downloaded_help(CHAT_MODEL))
        return 1

    print(f"Writer (abstractive: {CHAT_MODEL} on {device()})")
    print(f'  "{reply.text}"')
    print(f"  model ready in {load.seconds:.1f} s | read {reply.tokens_in} tokens, "
          f"wrote {reply.tokens_out} (limit {MAX_NEW_TOKENS}) in {reply.seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
