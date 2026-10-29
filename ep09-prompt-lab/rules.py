"""
rules.py - check the format rules of a message IN CODE, not by eye (Step 10).

Language models are poor at counting: "at most 3 sentences" or "exactly one emoji" often
comes back wrong, and a quick look misses it. So we never trust our eyes. check_rules()
reads the message and reports pass or fail for each rule:

    at most 3 sentences     split at . ! ? and count
    exactly one emoji       count emoji characters (a family like 👨‍👩‍👧 counts as one)
    ends with a question    the last character (after any emoji) is ?
    mentions every amount   every amount from settle() appears in the text
    no promises             no "refund", "guarantee", "will pay" ... (the bot only reminds)

Run:  python rules.py                                      (two fixed sample messages, offline)
      python rules.py "Kabir owes Meera 740 rupees..."     (your own message, amounts of note 1)
      python rules.py "..." --note 3                       (amounts of note 3)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re
from dataclasses import dataclass

from common import get_note
from guard import numbers_in
from models import Settlement, rupees
from settle import settle

MAX_SENTENCES = 3

# One emoji = one picture character, plus optional skin tone or joiner pieces glued to it.
_PICTURE = "[\U0001F300-\U0001F3FA\U0001F400-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF☀-➿]"
_GLUE = "[\U0001F3FB-\U0001F3FF️]*"
EMOJI = re.compile(f"(?:[\U0001F1E6-\U0001F1FF]{{2}}|{_PICTURE}{_GLUE}(?:‍{_PICTURE}{_GLUE})*)")

PROMISES = re.compile(r"\b(refund|guarantee|promise|will pay|'ll pay|will transfer|'ll transfer|will send|'ll send)",
                      re.IGNORECASE)


@dataclass
class RuleResult:
    rule: str
    passed: bool
    detail: str


def emojis(text: str) -> list[str]:
    return EMOJI.findall(text)


def sentences(text: str) -> list[str]:
    """Split a message into sentences (simple: at . ! or ? followed by a space)."""
    plain = EMOJI.sub(" ", text)
    plain = re.sub(r"\b(Rs|Mr|Mrs|Dr|vs)\.", r"\1", plain, flags=re.IGNORECASE)   # "Rs. 740" is not an end
    parts = re.split(r"(?<=[.!?])\s+", plain.strip())
    return [part for part in parts if re.search(r"\w", part)]


def check_rules(message: str, amounts: list[float]) -> list[RuleResult]:
    """Check every rule. amounts = the transfer amounts from settle(), e.g. [740, 230]."""
    found_sentences = sentences(message)
    found_emojis = emojis(message)
    without_emojis = EMOJI.sub("", message).strip()
    numbers = numbers_in(message)
    missing = [rupees(a) for a in amounts if round(a, 2) not in numbers]
    promise = PROMISES.search(message)

    return [
        RuleResult(f"at most {MAX_SENTENCES} sentences", len(found_sentences) <= MAX_SENTENCES,
                   f"found {len(found_sentences)}"),
        RuleResult("exactly one emoji", len(found_emojis) == 1,
                   f"found {len(found_emojis)}" + (f" ({' '.join(found_emojis)})" if found_emojis else "")),
        RuleResult("ends with a question", without_emojis.endswith("?"),
                   f"last character: {without_emojis[-1:] or 'none'!r}"),
        RuleResult("mentions every amount", not missing,
                   "all there" if not missing else f"missing: {', '.join(missing)}"),
        RuleResult("no promises", promise is None,
                   "none found" if promise is None else f"found {promise.group(0)!r}"),
    ]


def all_pass(results: list[RuleResult]) -> bool:
    return all(result.passed for result in results)


def failed_rules(results: list[RuleResult]) -> list[RuleResult]:
    return [result for result in results if not result.passed]


def report(results: list[RuleResult]) -> str:
    """One line per rule: PASS or FAIL, the rule, and what we found."""
    lines = [f"  {'PASS' if r.passed else 'FAIL'}  {r.rule:<24}{r.detail}" for r in results]
    passed = sum(r.passed for r in results)
    lines.append(f"  {passed} of {len(results)} rules pass")
    return "\n".join(lines)


def fallback_message(trip: str, settlement: Settlement) -> str:
    """A plain message written by CODE. It always passes every rule (a test checks that).

    Used when the model keeps breaking rules (Step 11): we give up gracefully instead of
    sending a broken message.
    """
    if not settlement.transfers:
        return f"{trip}: everyone is square, nobody owes anything. Shall we plan the next trip? 🙂"
    lines = "; ".join(t.line() for t in settlement.transfers)
    return f"{trip} settle-up: {lines}. Can everyone settle up by this weekend? 🙂"


# Two messages written by hand for this demo (not model output).
SAMPLE_MESSAGES = {
    "looks fine at a glance": "Hi all! Kabir owes Meera 740 rupees and Arjun owes Meera 230 rupees. "
                              "Meera gets her refund by Sunday. Thanks for a lovely Coorg trip 😊🌄",
    "follows every rule": "Coorg settle-up: Kabir owes Meera 740 rupees and Arjun owes Meera 230 rupees. "
                          "Can you both send it by Sunday? 🙏",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check a message against the rules, in code.")
    parser.add_argument("message", nargs="?", help="a message to check (default: the two samples)")
    parser.add_argument("--note", default="1", help="whose amounts to look for (default note 1)")
    args = parser.parse_args(argv)

    try:
        note = get_note(args.note)
    except ValueError as error:
        print(error)
        return 1
    amounts = [t.amount for t in settle(note.expenses).transfers]
    print(f"Amounts from settle() for note {note.id} ({note.slug}): {', '.join(rupees(a) for a in amounts)}\n")

    messages = {"your message": args.message} if args.message else SAMPLE_MESSAGES
    for label, message in messages.items():
        print(f"Message ({label}):\n  {message}")
        print(report(check_rules(message, amounts)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
