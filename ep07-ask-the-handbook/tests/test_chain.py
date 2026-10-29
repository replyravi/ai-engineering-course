"""
tests/test_chain.py - the question pipeline: prompt, tags, chain wiring, sources, ask.py loop.

The chat model is always a FAKE (fake_llm), so no key and no internet are needed.
Two tests also use the real local embeddings model on the real handbook.
"""

from langchain_core.documents import Document
from rag_fixtures import fake_store, mini_handbook, real_embeddings, real_store  # noqa: F401

from ask import chat, format_reply
from rag_chain import (
    PROMPT,
    REFUSAL,
    build_chain,
    build_simple_chain,
    cited_tags,
    format_docs,
    show_prompt,
)
from search import search
from testing.fakes import fake_llm, gemini_style_reply

QUESTION = "What is the refund window for a missed lunch?"


def test_format_docs_adds_tags_with_section_or_page():
    docs = [
        Document(page_content="Report it within 6 hours.",
                 metadata={"source": "refunds.md", "chunk": 4, "section": "Missed or late meal"}),
        Document(page_content="Our kitchen is not nut-free.",
                 metadata={"source": "food_safety.pdf", "chunk": 6, "page": 2}),
    ]

    text = format_docs(docs)

    assert "[refunds.md#4] (section: Missed or late meal)\nReport it within 6 hours." in text
    assert "[food_safety.pdf#6] (page 2)\nOur kitchen is not nut-free." in text


def test_prompt_puts_rules_context_and_question_in_place():
    messages = PROMPT.format_messages(context="[plans.md#6] (section: Pausing)\nUp to 10 days.",
                                      question="How long can I pause?")

    system, human = messages
    assert system.type == "system" and human.type == "human"   # last turn is human: Gemini needs that
    assert "ONLY the handbook extracts" in system.content
    assert REFUSAL in system.content
    assert "[plans.md#6]" in system.content
    assert human.content == "How long can I pause?"


def test_simple_chain_retrieves_fills_the_prompt_and_answers(fake_store):
    llm = fake_llm("Report it within 6 hours [refunds.md#2].")

    answer = build_simple_chain(fake_store, llm=llm, k=2).invoke(QUESTION)

    assert answer == "Report it within 6 hours [refunds.md#2]."
    system, human = llm.received[0]
    extracts = system.content.split("Handbook extracts:")[1]
    assert sum(line.startswith("[") for line in extracts.splitlines()) == 2   # k=2 tagged chunks
    assert human.content == QUESTION


def test_chain_returns_answer_and_sources_together(fake_store):
    llm = fake_llm("Report it within 6 hours [refunds.md#2].")

    result = build_chain(fake_store, llm=llm, k=3, max_distance=2.0).invoke(QUESTION)

    assert list(result) == ["answer", "sources"]
    assert result["answer"] == "Report it within 6 hours [refunds.md#2]."
    assert len(result["sources"]) == 3
    for source in result["sources"]:
        assert set(source) == {"tag", "where", "distance"}
        assert f"[{source['tag']}]" in llm.received[0][0].content   # the model saw every source


def test_gemini_style_replies_become_plain_text(fake_store):
    llm = fake_llm(gemini_style_reply("You can pause for 10 days [plans.md#1]."))

    result = build_chain(fake_store, llm=llm, max_distance=2.0).invoke("How long can I pause?")

    assert result["answer"] == "You can pause for 10 days [plans.md#1]."


def test_cited_tags_reads_the_tags_in_an_answer():
    answer = "Report within 6 hours [support.md#6]. A missed meal is 45 minutes late [refunds.md#3]."
    assert cited_tags(answer) == ["support.md#6", "refunds.md#3"]


def test_show_prompt_prints_the_messages_and_the_budget(fake_store, capsys):
    show_prompt("How long can I pause?", store=fake_store, k=2)

    printed = capsys.readouterr().out
    assert "----- system message -----" in printed
    assert "----- human message -----\nHow long can I pause?" in printed
    assert "prompt tokens + up to ~100 reply tokens" in printed


def test_real_search_finds_the_missed_lunch_rule(real_store):
    results = search(QUESTION, k=3, store=real_store)

    distances = [distance for _, distance in results]
    assert distances == sorted(distances)                       # closest first
    assert any("6 hours" in doc.page_content for doc, _ in results)


def test_real_chain_sends_the_right_chunks_to_the_model(real_store):
    llm = fake_llm("You have 6 hours after your slot ends to report it [delivery.md#8].")

    result = build_chain(real_store, llm=llm).invoke(QUESTION)

    assert "6 hours" in llm.received[0][0].content              # the answer text reached the model
    assert {s["tag"].split("#")[0] for s in result["sources"]} <= {"refunds.md", "support.md", "delivery.md"}


def test_chat_loop_answers_until_exit():
    class ScriptedChain:
        def __init__(self):
            self.questions = []

        def invoke(self, question):
            self.questions.append(question)
            return {"answer": "₹350 for two boxes [delivery.md#6].",
                    "sources": [{"tag": "delivery.md#6", "where": "section: The steel meal box",
                                 "distance": 0.301}]}

    chain = ScriptedChain()
    typed = iter(["What is the box deposit?", "", "exit", "never asked"])
    printed = []

    chat(chain, read=lambda _prompt: next(typed), write=printed.append)

    assert chain.questions == ["What is the box deposit?"]      # the empty line was skipped
    assert printed[1].startswith("Bot: ₹350 for two boxes [delivery.md#6].")
    assert "cited  delivery.md#6" in printed[1]
    assert printed[-1] == "Bye!"


def test_format_reply_marks_only_the_cited_sources():
    result = {"answer": "Up to 10 days [plans.md#6].",
              "sources": [{"tag": "plans.md#6", "where": "section: Pausing your plan", "distance": 0.198},
                          {"tag": "support.md#7", "where": "section: Quick answers", "distance": 0.474}]}

    lines = format_reply(result).splitlines()

    assert "cited  plans.md#6" in lines[1]
    assert "cited" not in lines[2]
