"""
app.py - Tiffin Toolbox: four local AI skills behind one command (step 12).

    python app.py summarise [note]                       a one-sentence summary of a note
    python app.py translate [note] [--text "..."]        a note into Hindi and back
    python app.py ask-photo [photo] ["question" ...]     questions about your own photo
    python app.py search "train ticket"                  find notes by meaning (also --matrix, --shape)
    python app.py                                        a small menu that keeps the models loaded

Each command hands the rest of the line to its skill file in skills/, so
"python app.py search --matrix" is the same as "python -m skills.search --matrix".
Every skill runs on your laptop. Want Gemini next to it? python compare.py
"""

import importlib
import sys
from pathlib import Path

from common import SAMPLE_PHOTO, find_photos

COMMANDS = {
    "summarise": ("skills.summarise", "a one-sentence summary of a note"),
    "translate": ("skills.translate", "translate a note into Hindi and back"),
    "ask-photo": ("skills.ask_photo", "ask questions about your photo"),
    "search": ("skills.search", "find notes by meaning"),
}


def usage() -> str:
    lines = ["Tiffin Toolbox: local AI on your laptop", "", "Commands:"]
    lines += [f"  {i}. {name:<10} {about}" for i, (name, (_, about)) in enumerate(COMMANDS.items(), start=1)]
    lines += ["", "Examples:", "  python app.py summarise my_notes/jaipur_trip.txt",
              '  python app.py search "train ticket"', "  python app.py ask-photo photos/desk.jpg \"What colour is the mug?\""]
    return "\n".join(lines)


def run(command: str, args: list[str]) -> int:
    """Import the skill (and its libraries) only now, then run its main()."""
    module = importlib.import_module(COMMANDS[command][0])
    return module.main(args)


def first_photo() -> Path:
    """The first picture in photos/ (your own, if you added one), else the sample desk."""
    return (find_photos() or [SAMPLE_PHOTO])[0]


def menu_args(command: str, answer: str) -> list[str]:
    """Turn what the user typed in the menu into the skill's command-line arguments."""
    answer = answer.strip()
    if command == "search":
        return [answer] if answer else ["--matrix"]
    if command == "ask-photo":
        return [str(first_photo()), answer] if answer else []
    if command == "translate" and answer and not Path(answer).is_file():
        return ["--text", answer]
    return [answer] if answer else []


def menu(read=input, write=print, runner=run) -> int:
    """A tiny menu. The models stay loaded between choices, so the second time is fast."""
    prompts = {
        "summarise": "Note file (Enter = my_notes/jaipur_trip.txt): ",
        "translate": "Text to translate, or a note file (Enter = my_notes/plant_care.txt): ",
        "ask-photo": f"Question about {first_photo().name} (Enter = two sample questions): ",
        "search": "Search your notes for (Enter = show the similarity matrix): ",
    }
    names = list(COMMANDS)
    write(usage())
    while True:
        try:
            choice = read("\nPick 1-4, or q to quit: ").strip().lower()
            if choice in ("q", "quit", "exit"):
                break
            if choice not in {"1", "2", "3", "4", *names}:
                write("Please type 1, 2, 3, 4 or q.")
                continue
            command = names[int(choice) - 1] if choice.isdigit() else choice
            runner(command, menu_args(command, read(prompts[command])))
        except (EOFError, KeyboardInterrupt):
            break
    write("Bye!")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return menu()
    command, rest = argv[0], argv[1:]
    if command in ("-h", "--help", "help"):
        print(usage())
        return 0
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n\n{usage()}")
        return 2
    return run(command, rest)


if __name__ == "__main__":
    sys.exit(main())
