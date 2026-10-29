"""
tests/test_ingest.py - chunking and indexing the fixture crawl (Step 12) and the search_docs tool (Step 13).

Most tests use FAKE embeddings (fast, offline). One test uses the real local MiniLM model, which is
already on your computer after Step 12; it is skipped if the model was never downloaded.
"""

import pytest
from mandi_fixtures import fake_store  # noqa: F401  (pytest fixture)

import ingest
from ingest import (FIXTURE, IndexNotBuiltError, build_index, chunk_tag, count_chunks, load_index, load_pages,
                    page_name, split_pages)
from testing.fakes import fake_embeddings
from tools.docs import make_search_tool


def test_the_fixture_loads_three_pages_with_their_urls():
    docs, data = load_pages(FIXTURE)
    assert [page_name(d.metadata["source"]) for d in docs] == ["site/", "fees.html", "units.html"]
    assert docs[1].metadata["title"] == "Buyer charges - Mandi Tool Belt"
    assert "4 percent of the goods value" in docs[1].page_content


def test_split_into_ten_chunks_numbered_per_page():
    chunks = split_pages(load_pages(FIXTURE)[0])
    assert len(chunks) == 10
    assert all(len(c.page_content) <= ingest.CHUNK_SIZE for c in chunks)
    assert [chunk_tag(c) for c in chunks if "fees" in c.metadata["source"]] == \
        ["fees.html#1", "fees.html#2", "fees.html#3"]


def test_indexing_twice_adds_nothing_twice(tmp_path):
    chunks = split_pages(load_pages(FIXTURE)[0])
    _, first = build_index(chunks, fake_embeddings(), index_dir=tmp_path)
    store, second = build_index(chunks, fake_embeddings(), index_dir=tmp_path)
    assert (first["added"], second["added"], count_chunks(store)) == (10, 0, 10)


def test_searching_an_empty_index_says_how_to_build_it(tmp_path):
    with pytest.raises(IndexNotBuiltError, match="python ingest.py"):
        load_index(fake_embeddings(), index_dir=tmp_path)


def test_search_docs_returns_passages_with_source_urls(fake_store):
    result = make_search_tool(fake_store).invoke({"query": "loading charge"})
    assert result.startswith("[1] ")
    assert result.count("source: https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/") == 3


def test_search_docs_without_an_index_returns_the_error_as_text(monkeypatch):
    def not_built():
        raise IndexNotBuiltError("The document index is empty. Build it first with:  python ingest.py")
    monkeypatch.setattr("tools.docs.load_index", not_built)
    assert make_search_tool().invoke({"query": "commission"}).startswith("Error: The document index is empty")


def test_ingest_main_prints_chunks_and_dim(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ingest, "INDEX_DIR", tmp_path)          # never touch the real index/
    assert ingest.main(["--source", str(FIXTURE)], embeddings=fake_embeddings()) == 0
    out = capsys.readouterr().out
    assert "Split into 10 chunks (chunk_size=400, chunk_overlap=60)" in out
    assert "10 chunks indexed, dim 384 (new this run: 10, removed: 0)" in out


def test_real_local_embeddings_find_the_loading_rule(tmp_path):
    try:
        from llm import get_embeddings
        embeddings = get_embeddings()
        embeddings.embed_query("test")
    except (OSError, ImportError):
        pytest.skip("the local MiniLM model is not downloaded yet: run python ingest.py once with internet")
    store, _ = build_index(split_pages(load_pages(FIXTURE)[0]), embeddings, index_dir=tmp_path)
    best, distance = store.similarity_search_with_score("loading charge per bag", k=1)[0]
    assert chunk_tag(best) == "fees.html#2"
    assert distance < 0.5
