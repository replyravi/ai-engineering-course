"""
tests/test_ask.py - the answer step with a FAKE chat model: no key, no internet, the same result every time.

The fake model replies with text WE wrote, so these tests check our code (retrieval, the prompt,
the citations, the printing), never the AI's wording.
"""

from lab_fixtures import breadcrumb_docs, fake_store  # noqa: F401

from chunklab.ask import build_chain, cited_tags, format_context, format_reply, main
from testing.fakes import fake_llm, gemini_style_reply

ANSWER = "Report it within 6 hours of the end of your slot [refunds.md > Missed or late meal]."


def test_chain_returns_the_answer_and_its_sources(fake_store):
    chain = build_chain(store=fake_store, llm=fake_llm(ANSWER), k=2)

    result = chain.invoke("What is the refund window for a missed lunch?")

    assert result["answer"] == ANSWER
    assert len(result["sources"]) == 2
    assert set(result["sources"][0]) == {"tag", "breadcrumb", "distance"}


def test_the_prompt_carries_every_tag_and_ends_with_the_question(fake_store):
    llm = fake_llm(ANSWER)
    build_chain(store=fake_store, llm=llm, k=3).invoke("Can I pause my plan?")

    system, human = llm.received[0]
    assert "[refunds.md > Missed or late meal] (Refunds Policy > Missed or late meal)" in system.content
    assert "[plans.md > Pausing your plan]" in system.content
    assert "I don't know based on the handbook." in system.content
    assert human.type == "human" and human.content == "Can I pause my plan?"     # Gemini needs a human turn last


def test_context_block_starts_each_extract_with_its_tag(breadcrumb_docs):
    context = format_context(breadcrumb_docs[:1])

    assert context.splitlines()[0] == "[refunds.md > Missed or late meal] (Refunds Policy > Missed or late meal)"


def test_cited_tags_and_the_reply_layout():
    answer = ("Report it within 6 hours [refunds.md > Missed or late meal]. "
              "You can pause for 10 days [plans.md > Pausing your plan].")
    result = {"answer": answer, "sources": [
        {"tag": "refunds.md > Missed or late meal", "breadcrumb": "Refunds Policy > Missed or late meal", "distance": 0.31},
        {"tag": "support.md > Quick answers", "breadcrumb": "Help > Quick answers", "distance": 0.42},
    ]}

    assert cited_tags(answer) == ["refunds.md > Missed or late meal", "plans.md > Pausing your plan"]
    lines = format_reply(result).splitlines()
    assert lines[0] == f"Answer: {answer}"
    assert lines[2].startswith("  cited  [refunds.md > Missed or late meal]  distance 0.310")
    assert lines[3].startswith("         [support.md > Quick answers]")


def test_gemini_style_replies_become_plain_text(fake_store):
    chain = build_chain(store=fake_store, llm=fake_llm(gemini_style_reply(ANSWER)), k=1)

    assert chain.invoke("refund window?")["answer"] == ANSWER


def test_without_a_key_ask_prints_the_free_key_help(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert main(["What is the refund window?"]) == 1
    printed = capsys.readouterr().out
    assert "No API key found for LLM_PROVIDER=gemini" in printed
    assert "https://aistudio.google.com/apikey" in printed
