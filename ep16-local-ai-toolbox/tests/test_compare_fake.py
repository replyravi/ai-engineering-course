"""
tests/test_compare_fake.py - the cloud side of compare.py, with a FAKE Gemini (steps 5, 6, 11 and 14).

The fake model answers with text WE choose, so these tests need no key and no internet.
They check our code: the messages we send, the table we print, the run log, and the
friendly messages when the key or the internet is missing.
"""

import pytest

import compare
from lab_fixtures import draw_sample_photo
from skills.summarise import SYSTEM
from testing.fakes import fake_llm, gemini_style_reply

NOTE = "Book two sleeper berths on the overnight train to Jaipur. Reach the station by 9 pm."


@pytest.fixture(autouse=True)
def runs_in_a_temp_folder(tmp_path, monkeypatch):
    """Never write into your real outputs/ folder during tests."""
    monkeypatch.setattr(compare, "RUNS_FILE", tmp_path / "runs.csv")
    for name in ("LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_cloud_summary_gets_the_same_instructions_as_the_local_model():
    llm = fake_llm("Two sleeper berths to Jaipur are booked; be at the station by 9 pm.")

    row = compare.cloud_row("summarise", llm, NOTE)

    assert row.engine == "cloud" and row.where == "Google's servers"
    assert row.text.startswith("Two sleeper berths")
    sent = llm.received[0]
    assert sent[0].type == "system" and sent[0].content == SYSTEM
    assert sent[-1].type == "human" and NOTE in sent[-1].content      # Gemini needs the human turn last


def test_gemini_style_list_of_blocks_still_gives_plain_text():
    row = compare.cloud_row("translate", fake_llm(gemini_style_reply("पौधों को पानी दें।")), NOTE)

    assert row.text == "पौधों को पानी दें।"


def test_cloud_photo_question_sends_text_and_a_jpeg(tmp_path):
    photo = draw_sample_photo(tmp_path)
    llm = fake_llm("I can see five pencils.")

    row = compare.cloud_row("ask-photo", llm, photo, "How many pencils are on the desk?")

    content = llm.received[0][-1].content
    assert content[0] == {"type": "text", "text": "How many pencils are on the desk?"}
    assert content[1]["type"] == "image" and content[1]["mime_type"] == "image/jpeg"
    assert len(content[1]["base64"]) > 1000
    assert row.text == "I can see five pencils."


def test_cloud_only_run_prints_the_table_and_logs_it(capsys, tmp_path):
    note = tmp_path / "trip.txt"
    note.write_text(NOTE, encoding="utf-8")

    assert compare.main(["summarise", str(note), "--cloud-only"], llm=fake_llm("A short summary.")) == 0

    out = capsys.readouterr().out
    assert "data left your laptop?" in out and "yes" in out
    runs = compare.read_runs(compare.RUNS_FILE)
    assert [(r["skill"], r["engine"], r["answer"]) for r in runs] == [("summarise", "cloud", "A short summary.")]


def test_counting_question_marks_right_and_wrong(capsys, tmp_path):
    photo = draw_sample_photo(tmp_path)
    args = ["ask-photo", str(photo), "How many pencils are on the desk?", "--truth", "5", "--cloud-only"]

    compare.main(args, llm=fake_llm("There are 4 pencils."))

    assert "cloud: WRONG (the true answer is 5)" in capsys.readouterr().out


def test_no_key_gives_the_friendly_help(monkeypatch):
    monkeypatch.setattr(compare, "internet_ok", lambda host, timeout=3.0: True)

    problem = compare.cloud_problem()

    assert "No API key found for LLM_PROVIDER=gemini" in problem
    assert "https://aistudio.google.com/apikey" in problem


def test_no_internet_gives_a_clear_offline_message(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    # The tests block the internet (conftest.py), so the real check fails, like Wi-Fi off.

    problem = compare.cloud_problem()

    assert problem.startswith("No internet: skipped the cloud column")


def test_a_failing_cloud_call_does_not_crash(capsys, tmp_path):
    note = tmp_path / "trip.txt"
    note.write_text(NOTE, encoding="utf-8")

    compare.main(["summarise", str(note), "--cloud-only"], llm=fake_llm())   # no replies: the call raises

    assert "The cloud call failed" in capsys.readouterr().out


def test_comparison_table_from_logged_runs():
    runs = [
        {"skill": "summarise", "engine": "local", "model": "Qwen/Qwen2.5-0.5B-Instruct",
         "where": "your laptop", "seconds": "1.0"},
        {"skill": "summarise", "engine": "local", "model": "Qwen/Qwen2.5-0.5B-Instruct",
         "where": "your laptop", "seconds": "2.0"},
        {"skill": "summarise", "engine": "cloud", "model": "gemini-3.5-flash-lite",
         "where": "Google's servers", "seconds": "0.8"},
    ]

    table = compare.comparison_table(runs)

    assert "1.5 s (2 runs)" in table and "0.8 s (1 run)" in table
    assert "sent to Google's servers" in table
    assert "translate speed   not run yet" in table
