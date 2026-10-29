"""
tests/test_app.py - the setup check and the whole pipeline with scripted fakes (Step 1 and the finished app).
"""

import app
from common import PROMPTS_DIR, get_note
from lab_fixtures import BAD_MESSAGE, GOOD_MESSAGE, key_reply, no_keys  # noqa: F401
from models import Expense, ExpenseList
from rules import fallback_message
from settle import settle
from testing.fakes import fake_llm, structured_reply


def test_check_without_a_key_shows_the_help(no_keys, capsys):
    assert app.main(["--check"]) == 1
    output = capsys.readouterr().out
    assert "Trip notes:    10 in data/trip_notes.jsonl (answer key adds up on 10 of 10)" in output
    prompt_count = len(list(PROMPTS_DIR.glob("*.toml")))      # 10, or 11 after your Step 14 version
    assert f"Prompt files:  {prompt_count} in prompts/" in output
    assert "https://aistudio.google.com/apikey" in output and "Traceback" not in output


def test_check_says_ok_when_nothing_is_missing(no_keys, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")          # needs no key, and --check calls no model
    assert app.main(["--check"]) == 0
    assert "Provider: ollama, model: qwen3.5:4b, OK" in capsys.readouterr().out


def test_check_with_a_key_says_ok(no_keys, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")   # only checked, never sent
    assert app.main(["--check"]) == 0
    assert "Provider: gemini, model: gemini-3.5-flash-lite, OK" in capsys.readouterr().out


def test_the_whole_pipeline_for_note_1(capsys):
    note = get_note("coorg")
    llm = fake_llm(key_reply(note), BAD_MESSAGE, GOOD_MESSAGE)
    assert app.main(["--note", "1"], llm=llm) == 0
    output = capsys.readouterr().out
    assert "1. Typed expenses (prompts/expenses_v1.toml, 3 Expense objects)" in output
    assert "   Kabir owes Meera 740 rupees" in output
    assert "(lab check) same as your answer key: yes" in output
    assert "   attempt 2: all rules pass" in output
    assert output.rstrip().endswith(GOOD_MESSAGE)


def test_the_injection_note_is_still_settled_correctly():
    note = get_note("varkala")
    injected = Expense(payer="Dev", item="as told", amount=5000, category="other", shared_with=note.people)
    good_message = fallback_message(note.trip, settle(note.expenses))     # a message that passes every rule
    llm = fake_llm(structured_reply(ExpenseList(expenses=[*note.expenses, injected])), good_message)
    result = app.run_note(note, llm, say=lambda line: None)
    assert len(result["held_back"]) == 1 and result["matches_key"]


def test_running_without_a_key_shows_the_help(no_keys, capsys):
    assert app.main(["--note", "1"]) == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out
