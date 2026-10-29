"""
tests/test_rag.py - the policy path: formatting with tags, selective retrieval, the refusal gate.
"""

from desk_fixtures import fake_store, real_embeddings, real_store  # noqa: F401

from rag import (MAX_DISTANCE, QUESTION_1, QUESTION_2, REFUSAL, SYSTEM_RULES, TINY_LIBRARY, build_policy_chain,
                 default_k, describe_sources, format_docs, index_handbook, retrieve, tiny_library_ids)
from testing.fakes import fake_llm


def test_format_docs_puts_a_tag_on_every_chunk(fake_store):
    docs = retrieve(fake_store, QUESTION_1)
    text = format_docs(docs)
    for doc in docs:
        assert f"[{doc.metadata['source']}#{doc.metadata['chunk']}]" in text
    assert "metadata=" not in text and "page_content=" not in text      # plain text, not Python objects
    assert [s["tag"] for s in describe_sources(docs)] == [f"{d.metadata['source']}#{d.metadata['chunk']}" for d in docs]


def test_the_prompt_has_the_refusal_rule():
    assert f"reply with exactly: {REFUSAL}" in SYSTEM_RULES


def test_index_is_54_chunks_of_384_numbers(real_embeddings, tmp_path):
    report = index_handbook(real_embeddings, index_dir=tmp_path / "index")
    assert (report["files"], report["documents"], report["chunks"], report["dim"]) == (5, 7, 54, 384)
    again = index_handbook(real_embeddings, index_dir=tmp_path / "index")
    assert (again["added"], again["removed"], again["chunks"]) == (0, 0, 54)       # no duplicates


def test_two_questions_get_different_chunks(real_store):
    first = [f"{d.metadata['source']}#{d.metadata['chunk']}" for d in retrieve(real_store, QUESTION_1)]
    second = [f"{d.metadata['source']}#{d.metadata['chunk']}" for d in retrieve(real_store, QUESTION_2)]
    assert first == ["delivery.md#8", "refunds.md#3", "refunds.md#12"]
    assert second == ["food_safety.pdf#6", "food_safety.pdf#5", "food_safety.pdf#4"]
    assert not set(first) & set(second)


def test_the_tiny_library_returns_everything(real_embeddings):
    found = tiny_library_ids(real_embeddings, [QUESTION_1, QUESTION_2])
    assert default_k() == 4
    for ids in found.values():
        assert sorted(ids) == [f"note{i}" for i in range(1, len(TINY_LIBRARY) + 1)]


def test_off_topic_question_is_refused_without_the_model(real_store):
    llm = fake_llm()                     # no scripted reply: calling the model would fail

    result = build_policy_chain(real_store, llm).invoke("What is the best laptop for coding?")

    assert result == {"answer": REFUSAL, "sources": []}
    assert llm.received == []
    assert retrieve(real_store, "What is the best laptop for coding?", k=1)[0].metadata["distance"] > MAX_DISTANCE


def test_answer_comes_with_sources(fake_store):
    llm = fake_llm("Report it in the app within 6 hours [delivery.md#8].")

    result = build_policy_chain(fake_store, llm, max_distance=2.0).invoke(QUESTION_1)

    assert result["answer"].endswith("[delivery.md#8].")
    assert len(result["sources"]) == 3
    assert "[" in llm.received[0][0].content                    # the system prompt carries tagged chunks


def test_when_the_model_refuses_no_sources_are_shown(fake_store):
    llm = fake_llm(REFUSAL)

    result = build_policy_chain(fake_store, llm, max_distance=2.0).invoke("Who founded Dabbaloop?")

    assert result == {"answer": REFUSAL, "sources": []}
