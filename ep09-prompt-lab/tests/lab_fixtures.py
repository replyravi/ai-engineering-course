"""
tests/lab_fixtures.py - shared test helpers for this episode.

Test files import what they need, for example:
    from lab_fixtures import no_keys, key_reply, lines_reply  # noqa: F401

- no_keys:         no provider settings and no keys (even if your real .env has one)
- key_reply(note): a scripted structured reply that copies the note's answer key (a "perfect" model)
- lines_reply(...): a scripted text reply with "X owes Y N" lines
- GOOD_MESSAGE / BAD_MESSAGE: group messages for note 1 that pass / break the rules
"""

import pytest

from models import ExpenseList, TripNote
from settle import settle
from testing.fakes import structured_reply

GOOD_MESSAGE = ("Coorg settle-up: Kabir owes Meera 740 rupees and Arjun owes Meera 230 rupees. "
                "Can you both send it by Sunday? 🙏")
BAD_MESSAGE = "Kabir owes Meera 740 rupees. Arjun owes Meera 230. Thanks all 😊🌄"   # 2 emoji, no question


@pytest.fixture
def no_keys(monkeypatch):
    for name in ["LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(name, raising=False)


def key_reply(note: TripNote):
    """What a perfect extractor would answer: exactly the expenses of the answer key."""
    return structured_reply(ExpenseList(expenses=note.expenses))


def lines_reply(note: TripNote) -> str:
    """A correct free-text answer: settle()'s lines, written the way our prompts ask."""
    return "\n".join(f"{t.from_person} owes {t.to_person} {t.amount:g}" for t in settle(note.expenses).transfers)
