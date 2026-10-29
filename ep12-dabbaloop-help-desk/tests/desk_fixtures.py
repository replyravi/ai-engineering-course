"""
tests/desk_fixtures.py - shared test helpers for this episode.

Test files import the fixtures they need, for example:
    from desk_fixtures import db_path, fake_store  # noqa: F401

- db_path:          a fresh dabbaloop.db made by seed_db.py in a temporary folder (the real file is never touched)
- fake_store:       the real handbook indexed with FAKE embeddings (tests the plumbing, fully offline)
- real_embeddings:  the free local model all-MiniLM-L6-v2 (meaningful search, no internet)
- real_store:       the real handbook indexed with the real local model
- semantic:         the SemanticRouter built on the real local model
- JAIN_SQL:         a correct SQL answer to "How many Jain thalis were sold last week?"
"""

import pytest

from ingest import build_index, load_handbook, split_documents
from seed_db import create_database
from testing.fakes import fake_embeddings

JAIN_SQL = ("SELECT SUM(o.quantity) AS jain_thalis_sold FROM orders o "
            "JOIN menu_items m ON m.item_id = o.item_id "
            "WHERE m.name = 'Jain Thali' AND o.order_date BETWEEN '2026-10-16' AND '2026-10-22'")


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "dabbaloop.db"
    create_database(path)
    return path


@pytest.fixture(scope="session")
def fake_store(tmp_path_factory):
    chunks = split_documents(load_handbook())
    store, _ = build_index(chunks, fake_embeddings(), tmp_path_factory.mktemp("fake_index"))
    return store


@pytest.fixture(scope="session")
def real_embeddings():
    """The real local embeddings model. Skips (with the fix) if it was never downloaded."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from llm import DEFAULT_LOCAL_EMBEDDINGS_MODEL

    try:
        return HuggingFaceEmbeddings(model_name=DEFAULT_LOCAL_EMBEDDINGS_MODEL)
    except Exception as error:  # not downloaded yet, and tests run offline
        pytest.skip(f"Local model not downloaded yet: run python rag.py index once online ({error!r})")


@pytest.fixture(scope="session")
def real_store(real_embeddings, tmp_path_factory):
    chunks = split_documents(load_handbook())
    store, _ = build_index(chunks, real_embeddings, tmp_path_factory.mktemp("real_index"))
    return store


@pytest.fixture(scope="session")
def semantic(real_embeddings):
    from embed_router import SemanticRouter

    return SemanticRouter(real_embeddings)
