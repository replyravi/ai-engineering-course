"""
tests/test_explain.py - the Gemini step (step 14) with a FAKE model: no key, no internet, same answer every time.
"""

import explain
from testing.fakes import fake_llm, gemini_style_reply

FACTS = {
    "sentence": "The cook smiled at the students because they loved his samosas.",
    "from_word": "they", "to_word": "students", "layer": 3, "head": 11,
    "top": [("students", 0.68), ("[SEP]", 0.08), ("the", 0.06)],
    "heads_agreeing": 7, "total_heads": 72,
}
REPLY = "The word they mostly looks at students. That is how the model links them. Other heads do other jobs."


def test_prompt_carries_our_real_numbers():
    system, human = explain.build_messages(FACTS)
    assert "fresher" in system.content
    for text in ["layer 3, head 11", "students 0.68", "[SEP] 0.08", "7 of the 72 heads", "exactly three"]:
        assert text in human.content


def test_explain_returns_the_model_text():
    llm = fake_llm(REPLY)
    assert explain.explain(FACTS, llm=llm) == REPLY
    assert len(llm.received) == 1                     # one call, with our two messages
    assert len(llm.received[0]) == 2


def test_gemini_style_content_blocks_still_print_as_text():
    assert explain.explain(FACTS, llm=fake_llm(gemini_style_reply(REPLY))) == REPLY


def test_main_prints_prompt_and_answer(capsys):
    assert explain.main(llm=fake_llm(REPLY), facts=FACTS) == 0
    out = capsys.readouterr().out
    assert "students 0.68" in out and REPLY in out


def test_no_key_gives_friendly_help(monkeypatch, capsys):
    for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "LLM_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    assert explain.main(facts=FACTS) == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out
