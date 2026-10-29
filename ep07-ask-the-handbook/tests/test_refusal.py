"""
tests/test_refusal.py - the "I don't know" path.

Two safety nets:
  1. the gate: if even the best chunk is far away, we refuse WITHOUT calling the model
  2. the prompt rule: if the chunks do not hold the answer, the model must reply with REFUSAL
"""

from rag_fixtures import fake_store, mini_handbook, real_embeddings, real_store  # noqa: F401

from ask import format_reply
from rag_chain import MAX_DISTANCE, REFUSAL, SYSTEM_RULES, build_chain
from testing.fakes import fake_llm


def test_weak_retrieval_refuses_without_calling_the_model(fake_store):
    llm = fake_llm()  # no scripted replies: calling it would raise an error

    result = build_chain(fake_store, llm=llm, max_distance=-1.0).invoke("Anything at all?")

    assert result == {"answer": REFUSAL, "sources": []}
    assert llm.received == []


def test_when_the_model_refuses_no_sources_are_shown(fake_store):
    llm = fake_llm(REFUSAL)

    result = build_chain(fake_store, llm=llm, max_distance=2.0).invoke("Who founded Dabbaloop?")

    assert result == {"answer": REFUSAL, "sources": []}
    assert len(llm.received) == 1  # this time the model was asked


def test_the_prompt_tells_the_model_the_exact_refusal():
    assert REFUSAL in SYSTEM_RULES
    assert "Never use outside knowledge" in SYSTEM_RULES


def test_off_topic_question_is_stopped_by_the_gate(real_store):
    llm = fake_llm()      # no scripted replies: calling the model would raise an error

    result = build_chain(real_store, llm=llm).invoke("What is the best laptop for coding?")

    assert result["answer"] == REFUSAL
    assert llm.received == []       # the model was never called


def test_handbook_question_passes_the_gate(real_store):
    best = real_store.similarity_search_with_score("Is the kitchen nut-free?", k=1)[0][1]
    llm = fake_llm("No, the kitchen is not nut-free [food_safety.pdf#6].")

    result = build_chain(real_store, llm=llm).invoke("Is the kitchen nut-free?")

    assert best < MAX_DISTANCE
    assert len(llm.received) == 1
    assert result["sources"]


def test_ask_prints_a_refusal_without_sources():
    assert format_reply({"answer": REFUSAL, "sources": []}) == f"Bot: {REFUSAL}"
