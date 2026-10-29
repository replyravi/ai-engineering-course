"""
tests/test_retry.py - the validate-and-retry loop with scripted fakes (Step 11).

The fake model answers with messages WE wrote, so we know exactly which rules fail.
"""

import retry
from common import get_note
from lab_fixtures import BAD_MESSAGE, GOOD_MESSAGE, no_keys  # noqa: F401
from retry import MAX_RETRIES, message_for_note, write_message
from rules import fallback_message
from settle import settle
from testing.fakes import fake_llm, gemini_style_reply

NOTE = get_note("coorg")


def test_fails_once_then_passes():
    llm = fake_llm(BAD_MESSAGE, GOOD_MESSAGE)
    result = message_for_note(NOTE, llm, say=lambda line: None)

    assert result.passed and result.attempts == 2
    assert result.text == GOOD_MESSAGE
    assert result.log[0].startswith("attempt 1: 3 of 5 rules pass (failed: exactly one emoji - found 2")
    assert result.log[-1] == "attempt 2: all rules pass"

    second_call = llm.received[1]
    assert second_call[-2].type == "ai" and second_call[-2].text == BAD_MESSAGE    # its own message
    assert second_call[-1].type == "human"                                          # Gemini needs a human turn last
    assert "exactly one emoji: found 2" in second_call[-1].text
    assert "ends with a question" in second_call[-1].text


def test_passes_first_time_without_a_retry():
    llm = fake_llm(GOOD_MESSAGE)
    result = message_for_note(NOTE, llm, say=lambda line: None)
    assert result.attempts == 1 and result.passed
    assert len(llm.received) == 1


def test_gives_up_gracefully_after_the_cap():
    llm = fake_llm(*[BAD_MESSAGE] * (MAX_RETRIES + 1))       # a 4th call would raise "ran out of replies"
    result = message_for_note(NOTE, llm, say=lambda line: None)

    assert not result.passed
    assert result.attempts == MAX_RETRIES + 1 == 3
    assert len(llm.received) == 3
    assert result.text == fallback_message(NOTE.trip, settle(NOTE.expenses))
    assert result.log[-1] == "gave up after 3 attempts: showing a plain message written by code instead"


def test_the_amounts_come_from_settle_not_from_the_model():
    llm = fake_llm(GOOD_MESSAGE)
    write_message(NOTE.trip, settle(NOTE.expenses), llm, say=lambda line: None)
    human = llm.received[0][-1]
    assert human.type == "human"
    assert "Kabir owes Meera 740 rupees" in human.text and "Arjun owes Meera 230 rupees" in human.text


def test_works_with_gemini_style_replies():
    llm = fake_llm(gemini_style_reply(GOOD_MESSAGE))
    assert message_for_note(NOTE, llm, say=lambda line: None).text == GOOD_MESSAGE


def test_the_command_line_without_a_key_shows_the_help(no_keys, capsys):
    assert retry.main(["--note", "1"]) == 1
    output = capsys.readouterr().out
    assert "https://aistudio.google.com/apikey" in output and "Traceback" not in output


def test_the_command_line_with_a_fake_model(capsys):
    assert retry.main(["--note", "1"], llm=fake_llm(BAD_MESSAGE, GOOD_MESSAGE)) == 0
    output = capsys.readouterr().out
    assert "attempt 2: all rules pass" in output
    assert "Message for the group (model, checked):" in output
