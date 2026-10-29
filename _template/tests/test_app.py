"""
tests/test_app.py - how an episode test uses the fakes.

The fake model answers with text WE choose, so the test:
  - needs no API key and no internet
  - gives the same result every time
  - checks OUR code (the prompt and the chain), not the AI

Run:  python -m pytest
"""

from app import explain
from testing.fakes import fake_llm, gemini_style_reply


def test_explain_returns_the_model_answer():
    llm = fake_llm("Embeddings turn text into lists of numbers.")

    answer = explain("embeddings", llm=llm)

    assert answer == "Embeddings turn text into lists of numbers."


def test_the_prompt_contains_the_topic():
    llm = fake_llm("Any answer")

    explain("vector databases", llm=llm)

    # llm.received keeps every prompt the fake model was sent
    messages_sent = llm.received[0]
    assert messages_sent[0].type == "system"
    assert "vector databases" in messages_sent[-1].content


def test_works_with_gemini_style_answers():
    # Gemini 3+ returns a list of content blocks; StrOutputParser turns it into text.
    llm = fake_llm(gemini_style_reply("Plain text, please."))

    assert explain("anything", llm=llm) == "Plain text, please."
