"""
tests/test_scoring.py - the scoreboard maths, the metadata filter, the saved collections and FAISS save/load.

Most tests use FAKE embeddings (same text -> same made-up vector), so they check our code, not MiniLM.
The last test runs the real local model and checks the scoreboard numbers shown in the README.
"""

from langchain_core.documents import Document
from lab_fixtures import breadcrumb_docs, fake_store, minilm_tokenizer, real_embeddings, tiktoken_ready  # noqa: F401

from chunklab.index import CollectionsMissingError, build_collection, saved_counts
from chunklab.score import (
    check_questions, holds_answer, in_memory_store, pick_winner, rank_of_answer, rule_rank, score_store, summarise,
)
from chunklab.search import file_filter, search
from chunklab.strategies import STRATEGIES, load_handbook, load_questions, make_chunks
from testing.fakes import fake_embeddings

PAUSE = {"question": "How long can I pause?", "sources": ["plans.md", "support.md"],
         "section": "Pausing your plan", "answer_contains": "10 days"}


def doc(text, source):
    return Document(page_content=text, metadata={"source": source, "chunk": 1})


def test_a_hit_needs_the_right_file_and_the_answer_phrase():
    assert holds_answer(doc("Pause for up to 10 DAYS a month.", "plans.md"), PAUSE)      # case does not matter
    assert not holds_answer(doc("Pause for up to 10 days a month.", "refunds.md"), PAUSE)  # wrong file
    assert not holds_answer(doc("Pause from the app.", "plans.md"), PAUSE)                # no answer phrase


def test_rank_is_the_first_chunk_that_holds_the_answer():
    ranked = [doc("Pause from the app.", "plans.md"), doc("Up to 10 days.", "refunds.md"),
              doc("Up to 10 days a month.", "support.md")]

    assert rank_of_answer(ranked, PAUSE) == 3
    assert rank_of_answer(ranked[:2], PAUSE) is None


def test_summary_and_the_tie_rule():
    rows = [{"hit": True, "first": True, "context_tokens": 100}, {"hit": True, "first": False, "context_tokens": 300},
            {"hit": False, "first": False, "context_tokens": 200}]
    assert summarise(rows) == {"hits": 2, "first": 1, "context": 200}

    summaries = {"a": {"hits": 9, "first": 9, "context": 200}, "b": {"hits": 10, "first": 7, "context": 400},
                 "c": {"hits": 10, "first": 9, "context": 300}, "d": {"hits": 10, "first": 9, "context": 250}}
    assert pick_winner(summaries) == "d"       # most hits, then most found first, then the least context


def test_scoring_with_fake_embeddings_finds_an_exact_copy(breadcrumb_docs):
    # Fake embeddings give the SAME vector to the SAME text, so a question that copies a chunk ranks it 1st.
    question = {"question": breadcrumb_docs[1].page_content, "sources": ["plans.md"],
                "section": "Pausing your plan", "answer_contains": "10 days"}
    store = in_memory_store("copy", breadcrumb_docs, fake_embeddings())

    rows = score_store(store, [question], count=len)      # count characters: no tokenizer needed

    assert rows[0]["rank"] == 1 and rows[0]["hit"] and rows[0]["first"]
    assert rows[0]["top"] == "plans.md#5"
    assert len(rows[0]["ranked"]) == 3          # every chunk is ranked, so we know how far a miss is


def test_rule_rank_finds_the_refund_rule(breadcrumb_docs):
    rule = Document(page_content="Report a missed lunch or dinner within 6 hours of the end of your slot.",
                    metadata={"source": "refunds.md", "chunk": 4})
    rows = [{"ranked": [breadcrumb_docs[1], breadcrumb_docs[0], rule]}]

    assert rule_rank(rows) == 3


def test_filter_keeps_only_the_chosen_files(fake_store):
    assert file_filter(None) is None
    assert file_filter(["a.md"]) == {"source": "a.md"}
    assert file_filter(["a.md", "b.md"]) == {"source": {"$in": ["a.md", "b.md"]}}

    results = search(fake_store, "Is it safe to eat?", k=3, files=["food_safety.md"])
    assert [found.metadata["source"] for found, _ in results] == ["food_safety.md"]

    results = search(fake_store, "Is it safe to eat?", k=3, files=["food_safety.md", "plans.md"])
    assert sorted(found.metadata["source"] for found, _ in results) == ["food_safety.md", "plans.md"]


def test_saved_collections_reopen_with_the_same_counts(breadcrumb_docs, tmp_path):
    for strategy in STRATEGIES:
        build_collection(strategy, breadcrumb_docs, fake_embeddings(), tmp_path)
    build_collection("headers", breadcrumb_docs, fake_embeddings(), tmp_path)   # building again: no duplicates

    assert saved_counts(tmp_path) == {strategy: 3 for strategy in STRATEGIES}


def test_searching_before_building_gives_a_clear_message(tmp_path):
    try:
        saved_counts(tmp_path)
    except CollectionsMissingError as error:
        assert "python -m chunklab index" in str(error)
    else:
        raise AssertionError("expected CollectionsMissingError")


def test_faiss_saves_and_loads_on_a_temp_folder(breadcrumb_docs, tmp_path):
    from langchain_community.vectorstores import FAISS

    embeddings = fake_embeddings()
    texts = [d.page_content for d in breadcrumb_docs]
    store = FAISS.from_embeddings(list(zip(texts, embeddings.embed_documents(texts))), embeddings,
                                  metadatas=[d.metadata for d in breadcrumb_docs], ids=["a", "b", "c"])
    store.save_local(str(tmp_path / "faiss_index"))

    loaded = FAISS.load_local(str(tmp_path / "faiss_index"), embeddings, allow_dangerous_deserialization=True)

    assert loaded.index.ntotal == 3
    best, distance = loaded.similarity_search_with_score(texts[2], k=1)[0]
    assert best.id == "c" and distance < 1e-6 and best.metadata["source"] == "food_safety.md"


def test_every_answer_phrase_is_in_its_section():
    questions = load_questions()

    assert len(questions) == 10
    assert check_questions(questions)


def test_real_scoreboard_matches_the_readme(real_embeddings, minilm_tokenizer, tiktoken_ready):
    questions = load_questions()
    docs = load_handbook()
    four = ["character", "recursive", "tokens", "headers"]        # the README's four strategies
    results = {strategy: score_store(in_memory_store(f"real_{strategy}", make_chunks(strategy, docs),
                                                     real_embeddings), questions)
               for strategy in four}
    summaries = {strategy: summarise(rows) for strategy, rows in results.items()}

    assert [summaries[s]["hits"] for s in four] == [10, 10, 8, 10]          # hit@3
    assert [summaries[s]["first"] for s in four] == [9, 9, 7, 10]           # found first
    assert [summaries[s]["context"] for s in four] == [244, 236, 293, 305]  # MiniLM tokens per question
    assert [rule_rank(results[s]) for s in four] == [8, 8, 1, 3]            # Episode 07's refund rule
    assert pick_winner(summaries) == "headers"
