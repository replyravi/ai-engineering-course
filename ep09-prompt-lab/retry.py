"""
retry.py - write the group message, check it in code, and retry with the failures (Steps 10 and 11).

    1. settle.py works out who owes whom (the numbers never come from the model)
    2. the model writes a short message with prompts/message_v1.toml
    3. rules.py checks every rule
    4. if a rule fails, we send the model its own message AND the list of failures, and ask again
    5. at most MAX_RETRIES retries; then we give up gracefully with a plain message written by code

    Prompt v1 -> Model -> Output -> check_rules() -> all pass? show it
                                                  -> some fail? append the failures -> retry (max 2)
                                                  -> still failing? give up gracefully

Run:  python retry.py              (note 1, needs your free Gemini key)
      python retry.py --note 6
      python retry.py --show-prompt (the filled prompt only, offline)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage

from common import friendly_error, get_note, make_llm, model_label
from llm import MissingAPIKeyError
from models import Settlement, TripNote
from prompt_files import read_prompt
from rules import check_rules, failed_rules, fallback_message
from settle import settle

MAX_RETRIES = 2            # so at most 3 attempts: 1 try + 2 retries


@dataclass
class MessageResult:
    text: str                         # the message we show the group
    attempts: int                     # how many times we asked the model
    passed: bool                      # False = we gave up and used the code-written message
    log: list[str] = field(default_factory=list)


def message_inputs(trip: str, settlement: Settlement) -> dict:
    """The slots of prompts/message_v1.toml: the trip name and the transfer lines from settle()."""
    return {"trip": trip, "transfers": "\n".join(settlement.lines())}


def feedback(failures) -> str:
    """The note we send back to the model when rules fail. It ends the conversation with a human turn."""
    lines = [f"- {f.rule}: {f.detail}" for f in failures]
    return ("Your message broke these rules:\n" + "\n".join(lines) +
            "\nWrite the whole message again and follow every rule. Write only the message.")


def write_message(trip: str, settlement: Settlement, llm=None, max_retries: int = MAX_RETRIES,
                  say=print) -> MessageResult:
    """Ask, check, retry with the failure list, give up gracefully after max_retries."""
    llm = llm or make_llm()
    amounts = [t.amount for t in settlement.transfers]
    messages = read_prompt("message_v1").template().format_messages(**message_inputs(trip, settlement))
    log = []

    for attempt in range(1, max_retries + 2):
        text = llm.invoke(messages).text.strip()
        results = check_rules(text, amounts)
        failures = failed_rules(results)
        if not failures:
            log.append(f"attempt {attempt}: all rules pass")
            say(log[-1])
            return MessageResult(text, attempt, True, log)
        log.append(f"attempt {attempt}: {len(results) - len(failures)} of {len(results)} rules pass "
                   f"(failed: {'; '.join(f'{f.rule} - {f.detail}' for f in failures)})")
        say(log[-1])
        # Keep the conversation: our prompt, the model's message, then the list of what went wrong.
        messages = [*messages, AIMessage(text), HumanMessage(feedback(failures))]

    log.append(f"gave up after {max_retries + 1} attempts: showing a plain message written by code instead")
    say(log[-1])
    return MessageResult(fallback_message(trip, settlement), max_retries + 1, False, log)


def message_for_note(note: TripNote, llm=None, say=print) -> MessageResult:
    """The whole Step 11 for one note: settle with Python, then the checked message."""
    return write_message(note.trip, settle(note.expenses), llm, say=say)


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="A rule-checked group message with automatic retries.")
    parser.add_argument("--note", default="1", help="note number or name (default 1)")
    parser.add_argument("--show-prompt", action="store_true", help="print the filled prompt only (no key needed)")
    args = parser.parse_args(argv)

    try:
        note = get_note(args.note)
    except ValueError as error:
        print(error)
        return 1
    settlement = settle(note.expenses)
    if args.show_prompt:
        for message in read_prompt("message_v1").template().format_messages(**message_inputs(note.trip, settlement)):
            print(f"----- {message.type} -----\n{message.text}")
        return 0

    try:
        llm = llm or make_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    print(f"Model: {model_label()}   Note {note.id} ({note.slug})")
    print("Transfers from settle(): " + " | ".join(settlement.lines()) + "\n")
    try:
        result = write_message(note.trip, settlement, llm)
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1
    print(f"\nMessage for the group ({'model, checked' if result.passed else 'written by code'}):\n{result.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
