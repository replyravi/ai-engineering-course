"""
skills/translate.py - translate a note into Hindi on your laptop, then back to English (step 6).

Can't read Hindi? The trip back is your judge: if the English that comes back has lost the
meaning, the Hindi was wrong too. Even if you do read Hindi, a round trip is a quick sanity check.

The same small chat model does it, with a different system message. There is no separate
"translation" pipeline in transformers v5 any more (step 3).

Run (inside ep16-local-ai-toolbox):
    python -m skills.translate                                (the sample note my_notes/plant_care.txt)
    python -m skills.translate my_notes/amma_birthday.txt
    python -m skills.translate --text "Good morning, how are you?"
    python -m skills.translate --to Tamil
"""

import argparse
import sys
from pathlib import Path

from common import CHAT_MODEL, NOTES_DIR, Stopwatch, device, is_downloaded, not_downloaded_help, \
    offline_mode, read_note
from skills.chat_model import Reply, ask_local, chat_model

SAMPLE = NOTES_DIR / "plant_care.txt"
MAX_NEW_TOKENS = 200     # Hindi needs more tokens than English for the same sentence

# Telling the model the script helps it write Hindi letters instead of Latin ones.
SCRIPTS = {"Hindi": "Devanagari", "Marathi": "Devanagari", "Tamil": "Tamil", "Telugu": "Telugu",
           "Bengali": "Bengali", "Gujarati": "Gujarati", "Kannada": "Kannada"}


def system_message(language: str) -> str:
    script = f", in {SCRIPTS[language]} script" if language in SCRIPTS else ""
    return (f"You are a translator. Translate the user's text into {language}. "
            f"Reply with the translation only{script}.")


def translate(text: str, language: str = "Hindi") -> Reply:
    """The local chat model's translation of text into language."""
    return ask_local(system_message(language), text, max_new_tokens=MAX_NEW_TOKENS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m skills.translate",
                                     description="Translate a note there and back with the local model.")
    parser.add_argument("note", nargs="?", default=str(SAMPLE), help="a .txt note (default my_notes/plant_care.txt)")
    parser.add_argument("--text", help="translate this text instead of a note")
    parser.add_argument("--to", default="Hindi", help="the language to translate into (default Hindi)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.text:
        text = " ".join(args.text.split())
    elif Path(args.note).is_file():
        text = read_note(args.note)
    else:
        print(f"No note at {args.note}. Try: python -m skills.translate --text \"Good morning\"")
        return 1
    if offline_mode() and not is_downloaded(CHAT_MODEL):
        print(not_downloaded_help(CHAT_MODEL))
        return 1

    try:
        with Stopwatch() as load:
            chat_model()
        there = translate(text, args.to)
        back = translate(there.text, "English")
    except OSError:
        print(not_downloaded_help(CHAT_MODEL))
        return 1

    print(f"Local model: {CHAT_MODEL} on {device()} (ready in {load.seconds:.1f} s)\n")
    print(f"English:\n  {text}\n")
    limit = f", stopped at the {MAX_NEW_TOKENS}-token limit" if there.hit_limit else ""
    print(f"{args.to} ({there.seconds:.1f} s, {there.tokens_out} tokens{limit}):\n  {there.text}\n")
    print(f"Back to English ({back.seconds:.1f} s):\n  {back.text}\n")
    print("Judge it: does the trip back keep the meaning? Compare with Gemini: python compare.py translate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
