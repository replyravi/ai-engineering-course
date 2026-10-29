"""
app.py - the Chai Kitty prompt lab, end to end (Step 1 and the finished program).

Chai Kitty is an invented name: the shared trip money of our invented group of friends.

    messy trip note
      -> typed expenses        (model + with_structured_output, prompts/expenses_v1.toml)
      -> guard.py              (names in the group? amounts really in the note?)
      -> settle.py             (Python does the maths)
      -> group message         (model writes it, rules.py checks it, retry.py retries)

The bot only computes and reminds. It never promises a payment or a refund.

Run:  python app.py --check        (Step 1: settings, data and prompt files, offline)
      python app.py                (note 1, needs your free Gemini key)
      python app.py --note 10      (the note with a hidden order to the AI)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse

from common import NOTES_FILE, friendly_error, get_note, load_notes, make_llm, model_label
from guard import check_expenses
from llm import ENV_FILE, MissingAPIKeyError, check_api_key, get_model_name, get_provider
from prompt_files import list_prompts
from retry import write_message
from settle import same_result, settle
from techniques import expense_table, extract_expenses


def check_setup() -> int:
    """Step 1: everything this lab needs, checked without calling a model."""
    try:
        provider = get_provider()
    except ValueError as error:          # a typo in LLM_PROVIDER
        print(error)
        return 1
    model = get_model_name(provider)
    print(f"Settings file: {ENV_FILE or 'no .env found'}")

    notes = load_notes()
    key_ok = sum(same_result(n.transfers, n.expenses) for n in notes)
    print(f"Trip notes:    {len(notes)} in {NOTES_FILE.parent.name}/{NOTES_FILE.name} "
          f"(answer key adds up on {key_ok} of {len(notes)})")
    prompts = list_prompts()
    print(f"Prompt files:  {len(prompts)} in prompts/")

    try:
        check_api_key(provider)
    except MissingAPIKeyError as error:
        print(str(error).strip())
        return 1
    print(f"Provider: {provider}, model: {model}, OK")
    return 0


def run_note(note, llm, say=print) -> dict:
    """The whole pipeline for one note. Returns what it found (handy for tests)."""
    say(f"Note {note.id} ({note.slug}):\n  " + note.note.replace("\n", "\n  ") + "\n")

    result = extract_expenses(note.note, llm)
    say(f"1. Typed expenses (prompts/expenses_v1.toml, {len(result.expenses)} Expense objects)")
    say("\n".join(expense_table(result.expenses)))

    kept, held_back = check_expenses(result.expenses, note.note, note.people)
    say(f"\n2. Guard: kept {len(kept)}, held back {len(held_back)}")
    for expense, reason in held_back:
        say(f"   held back for a person to check: {expense.payer} {expense.amount:g} ({reason})")

    settlement = settle(kept)
    say("\n3. Settle-up (Python)")
    for line in settlement.lines():
        say(f"   {line}")
    matches = same_result(settlement.transfers, note.expenses)
    say(f"   (lab check) same as your answer key: {'yes' if matches else 'NO'}")

    say("\n4. Group message (prompts/message_v1.toml, checked by rules.py)")
    message = write_message(note.trip, settlement, llm, say=lambda line: say(f"   {line}"))
    say(f"\n{message.text}")
    return {"expenses": result.expenses, "held_back": held_back, "settlement": settlement,
            "matches_key": matches, "message": message}


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="The Chai Kitty prompt lab (Chai Kitty is an invented name).")
    parser.add_argument("--check", action="store_true", help="check settings, notes and prompts (no model call)")
    parser.add_argument("--note", default="1", help="note number or name (default 1)")
    args = parser.parse_args(argv)

    try:
        if args.check:
            return check_setup()
        note = get_note(args.note)
    except ValueError as error:        # a typo in data/trip_notes.jsonl, or an unknown note
        print(error)
        return 1
    try:
        llm = llm or make_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    print(f"Model: {model_label()}\n")
    try:
        run_note(note, llm)
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
