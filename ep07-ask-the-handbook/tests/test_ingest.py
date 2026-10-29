"""
tests/test_ingest.py - the indexing pipeline: load, split, ids, and no duplicates.

Uses a tiny handbook in a temporary folder and FAKE embeddings, so it is fast and offline.
"""

import pytest
from rag_fixtures import mini_handbook  # noqa: F401  (pytest fixture)

from ingest import (
    HANDBOOK_DIR,
    IndexNotBuiltError,
    build_index,
    chunk_id,
    count_chunks,
    load_handbook,
    load_index,
    open_index,
    shared_text,
    split_documents,
)
from testing.fakes import fake_embeddings


def test_real_handbook_loads_markdown_files_and_pdf_pages():
    docs = load_handbook(HANDBOOK_DIR)
    markdown = [d for d in docs if d.metadata["source"].endswith(".md")]
    pdf_pages = [d for d in docs if d.metadata["source"] == "food_safety.pdf"]

    assert len(markdown) >= 4                                   # one Document per Markdown file
    assert [d.metadata["page"] for d in pdf_pages] == [1, 2, 3]  # one Document per PDF page
    assert "not nut-free" in pdf_pages[1].page_content          # the PDF text is readable


def test_split_counts_and_metadata_on_a_fixture(mini_handbook):
    chunks = split_documents(load_handbook(mini_handbook), chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 5                                     # 1 from plans.md + 4 from refunds.md
    assert all(len(c.page_content) <= 100 for c in chunks)
    refunds = [c for c in chunks if c.metadata["source"] == "refunds.md"]
    assert [c.metadata["chunk"] for c in refunds] == [1, 2, 3, 4]  # numbered per file, from 1
    six_hours = next(c for c in chunks if "6 hours" in c.page_content)
    assert six_hours.metadata["section"] == "Missed meal"     # the heading travels with the chunk


def test_neighbouring_chunks_share_the_overlap(mini_handbook):
    chunks = split_documents(load_handbook(mini_handbook), chunk_size=100, chunk_overlap=20)
    first, second = [c for c in chunks if c.metadata["source"] == "refunds.md"][1:3]

    shared = shared_text(first.page_content, second.page_content)

    assert shared == "full credit in your"
    assert len(shared) <= 20


def test_shared_text_is_empty_when_nothing_overlaps():
    assert shared_text("pause your plan", "plan ahead") == "plan"
    assert shared_text("missed meal", "wallet credit") == ""


def test_chunk_ids_are_stable_and_unique(mini_handbook):
    first_run = [chunk_id(c) for c in split_documents(load_handbook(mini_handbook), 100, 20)]
    second_run = [chunk_id(c) for c in split_documents(load_handbook(mini_handbook), 100, 20)]

    assert first_run == second_run
    assert len(set(first_run)) == len(first_run)


def test_running_ingest_twice_does_not_duplicate(mini_handbook, tmp_path):
    chunks = split_documents(load_handbook(mini_handbook), 100, 20)
    embeddings = fake_embeddings()

    _, first = build_index(chunks, embeddings, tmp_path / "index")
    _, second = build_index(chunks, embeddings, tmp_path / "index")

    assert first == {"added": 5, "removed": 0, "total": 5}
    assert second == {"added": 0, "removed": 0, "total": 5}


def test_changed_file_replaces_its_old_chunks(mini_handbook, tmp_path):
    embeddings = fake_embeddings()
    build_index(split_documents(load_handbook(mini_handbook), 100, 20), embeddings, tmp_path / "index")

    plans = mini_handbook / "plans.md"
    plans.write_text(plans.read_text(encoding="utf-8").replace("10 days", "12 days"), encoding="utf-8")
    new_chunks = split_documents(load_handbook(mini_handbook), 100, 20)
    store, report = build_index(new_chunks, embeddings, tmp_path / "index")

    assert report == {"added": 1, "removed": 1, "total": len(new_chunks)}
    texts = store.get()["documents"]
    assert any("12 days" in t for t in texts)
    assert not any("10 days" in t for t in texts)


def test_index_is_saved_to_disk(mini_handbook, tmp_path):
    embeddings = fake_embeddings()
    build_index(split_documents(load_handbook(mini_handbook), 100, 20), embeddings, tmp_path / "index")

    reopened = open_index(embeddings, tmp_path / "index")

    assert count_chunks(reopened) == 5


def test_searching_before_ingest_gives_a_clear_message(tmp_path):
    with pytest.raises(IndexNotBuiltError, match="python ingest.py"):
        load_index(fake_embeddings(), tmp_path / "empty_index")
