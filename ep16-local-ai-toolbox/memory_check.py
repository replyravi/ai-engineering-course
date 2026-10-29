"""
memory_check.py - load a model only when it is needed, and give the memory back after (step 13).

A laptop has limited memory (RAM). Each local model takes hundreds of megabytes while it is
loaded. Tiffin Toolbox loads a model the first time a skill needs it (lazy loading, skills/lazy.py)
and release() lets it go. Watch the numbers go up and come back down.

Run:  python memory_check.py
"""

import sys

from common import CHAT_MODEL, SAMPLE_NOTE, SAMPLE_PHOTO, VISION_MODEL, is_downloaded, memory_mb, \
    not_downloaded_help, read_note
from skills.lazy import loaded_names, release


def show(moment: str, previous: float | None = None) -> float:
    """Print the memory this program holds now, and the change since the last reading."""
    now = memory_mb()
    change = "" if previous is None else f"{now - previous:+8,.0f} MB"
    loaded = ", ".join(loaded_names()) or "none"
    print(f"{moment:<40}{now:>8,.0f} MB {change:>11}   models loaded: {loaded}")
    return now


def main() -> int:
    for model in (CHAT_MODEL, VISION_MODEL):
        if not is_downloaded(model):
            print(not_downloaded_help(model))
            return 1
    from skills.ask_photo import ask, open_photo
    from skills.summarise import summarise

    print(f"{'moment':<40}{'memory':>11} {'change':>11}")
    before = show("start (no model, no PyTorch loaded yet)")

    summarise(read_note(SAMPLE_NOTE))                          # the first call loads the chat model
    before = show("after one summary", before)

    release("chat")                                            # forget the chat model, free its memory
    before = show("after release('chat')", before)

    ask(open_photo(SAMPLE_PHOTO), "What colour is the mug?")   # the first call loads the vision model
    before = show("after one photo question", before)

    release()                                                  # forget every model
    show("after release()", before)
    return 0


if __name__ == "__main__":
    sys.exit(main())
