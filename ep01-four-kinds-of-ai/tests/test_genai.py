"""
tests/test_genai.py - the Gemini steps (5, 11, 12, 13), tested with FAKE models.

No key, no internet, same result every time. We script what the "AI" answers and
check OUR code: the prompts we send, how we read the answers, and the friendly
help when a key is missing.
"""

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import compare
import d_genai
import e_reply_and_probe
import first_call
from common import load_and_split
from d_genai import FeedbackLabel, classify
from testing.fakes import fake_embeddings, fake_llm, gemini_style_reply, structured_reply


@pytest.fixture
def no_keys(monkeypatch):
    """Start with no provider settings and no keys (even if your real .env has one)."""
    for name in ["LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Step 5: first call
# ---------------------------------------------------------------------------

def test_first_call_prints_text_and_tokens(capsys):
    reply = AIMessage(
        content=[{"type": "text", "text": "An LLM predicts the next word. It learned from lots of text.",
                  "extras": {"signature": "fake"}}],                     # Gemini 3 style: a list of blocks
        usage_metadata={"input_tokens": 31, "output_tokens": 18, "total_tokens": 49},
    )
    assert first_call.main(llm=fake_llm(reply)) == 0
    output = capsys.readouterr().out
    assert "An LLM predicts the next word." in output
    assert "[{" not in output                                            # we printed .text, not .content
    assert "Tokens: 31 in, 18 out, 49 total" in output


def test_first_call_without_a_key_shows_where_to_get_one(no_keys, capsys):
    assert first_call.main() == 1
    output = capsys.readouterr().out
    assert "Provider: gemini" in output
    assert "https://aistudio.google.com/apikey" in output
    assert "Traceback" not in output


def test_the_one_line_switch_to_openai_without_a_key(no_keys, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert first_call.main() == 1
    output = capsys.readouterr().out
    assert "Provider: openai" in output
    assert "https://platform.openai.com/api-keys" in output


def test_a_typo_in_the_provider_gets_a_clear_message(no_keys, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "gemni")
    assert first_call.main() == 1
    assert "LLM_PROVIDER='gemni' is not supported" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Step 11: zero-shot labels with structured output
# ---------------------------------------------------------------------------

def test_classifier_reads_the_structured_label():
    llm = fake_llm(structured_reply(FeedbackLabel(label="complaint", reason="sarcastic about a cold dosa")))
    (answer, ms), = classify(["Great job, the dosa was cold again."], llm)
    assert answer.label == "complaint"
    assert answer.reason == "sarcastic about a cold dosa"
    assert ms >= 0


def test_classifier_prompt_defines_the_labels_and_includes_the_feedback():
    llm = fake_llm(structured_reply(FeedbackLabel(label="praise", reason="happy")))
    classify(["The chai is great."], llm)
    system, human = llm.received[0]
    assert all(word in system.content for word in ["praise", "complaint", "suggestion"])
    assert human.content == "Feedback: The chai is great."


def test_only_our_three_labels_are_allowed():
    with pytest.raises(ValidationError):
        FeedbackLabel(label="angry", reason="not one of our labels")


def test_genai_script_scores_the_10_test_rows_and_saves_them(tmp_path, capsys):
    _, test = load_and_split()
    replies = [structured_reply(FeedbackLabel(label=label, reason="scripted")) for label in test["label"]]
    replies[0] = structured_reply(FeedbackLabel(label="suggestion", reason="scripted miss"))
    saved = tmp_path / "gemini.csv"

    assert d_genai.main(llm=fake_llm(*replies), results_file=saved) == 0
    output = capsys.readouterr().out
    assert "Training rows used: 0" in output
    assert "Zero-shot accuracy on the 10 test rows: 90% (9 of 10)" in output
    assert compare.saved_gemini_score(test, saved)[0] == 9    # compare.py can reuse the saved run


def test_genai_script_without_a_key_shows_the_help(no_keys, capsys):
    assert d_genai.main() == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Steps 12 and 13: replies and the hallucination probe
# ---------------------------------------------------------------------------

def test_reply_drafts_are_returned_as_plain_text():
    llm = fake_llm("Sorry about the oily biryani, I will check with the kitchen today.",
                   gemini_style_reply("Sorry about the wait, I will look at the queue."))
    drafts = e_reply_and_probe.draft_replies(["The biryani was oily.", "Waited 25 minutes."], llm)
    assert drafts == ["Sorry about the oily biryani, I will check with the kitchen today.",
                      "Sorry about the wait, I will look at the queue."]
    system, human = llm.received[0]
    assert "Do not promise refunds" in system.content
    assert human.content == "Complaint: The biryani was oily."


def test_probe_pastes_the_fact_only_the_second_time():
    llm = fake_llm("It opens at 8 am on Sundays.", "It is closed on Sundays.")
    question = e_reply_and_probe.QUESTION
    e_reply_and_probe.ask(question, llm)
    e_reply_and_probe.ask(question, llm, fact=e_reply_and_probe.CANTEEN_FACT)
    first, second = llm.received
    assert len(first) == 1 and first[0].content == question                 # no facts at all
    assert e_reply_and_probe.CANTEEN_FACT in second[0].content              # the fact is in the system message


def test_reply_and_probe_script_end_to_end(capsys):
    llm = fake_llm("Draft one.", "Draft two.", "Draft three.",
                   "Byte Bites probably opens around 9 am on Sundays.",
                   "Byte Bites is closed on Sundays.")
    assert e_reply_and_probe.main(llm=llm) == 0
    output = capsys.readouterr().out
    assert output.count("Draft:") == 3
    assert "nothing is sent" in output
    assert "uses the pasted fact (says 'closed'): yes" in output


def test_reply_and_probe_without_a_key_shows_the_help(no_keys, capsys):
    assert e_reply_and_probe.main() == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The cold open: four answers for one sentence
# ---------------------------------------------------------------------------

def test_four_answers_for_one_sentence():
    llm = fake_llm(structured_reply(FeedbackLabel(label="complaint", reason="sarcasm")))
    answers = compare.four_answers("Great job, the dosa was cold again.", embeddings=fake_embeddings(), llm=llm)
    assert answers["A rules"] == "praise"
    assert answers["B trained model"] in {"praise", "complaint", "suggestion"}
    assert answers["C clusters"].startswith("group ")
    assert answers["D Gemini"] == "complaint (sarcasm)"


def test_four_answers_without_a_key(no_keys):
    answers = compare.four_answers("The sambar was cold.", embeddings=fake_embeddings())
    assert "needs your free Gemini key" in answers["D Gemini"]
