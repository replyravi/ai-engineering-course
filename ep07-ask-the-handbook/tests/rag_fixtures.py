"""
tests/rag_fixtures.py - shared test helpers for this episode.

Test files import the fixtures they need, for example:
    from rag_fixtures import mini_handbook, real_embeddings  # noqa: F401

- mini_handbook:   a tiny 2-file handbook in a temporary folder (fast, fully offline)
- fake_store:      that tiny handbook indexed with FAKE embeddings (tests the plumbing)
- real_embeddings: the free local model all-MiniLM-L6-v2 (meaningful search, no internet)
- real_store:      the REAL handbook/ indexed with the real local model
"""

import pytest

from ingest import HANDBOOK_DIR, build_index, load_handbook, split_documents
from testing.fakes import fake_embeddings

MINI_REFUNDS = """# Mini Refunds

## Missed meal

A meal is missed when it is 45 minutes late. Report it within 6 hours to get a full credit in your wallet. Late reports are checked one by one.

## Cancelling

Cancel any time from the app. Refunds reach your bank in 7 working days.
"""

MINI_PLANS = """# Mini Plans

## Pausing

You can pause your plan for up to 10 days a month.
"""


@pytest.fixture
def mini_handbook(tmp_path):
    folder = tmp_path / "handbook"
    folder.mkdir()
    (folder / "refunds.md").write_text(MINI_REFUNDS, encoding="utf-8")
    (folder / "plans.md").write_text(MINI_PLANS, encoding="utf-8")
    return folder


@pytest.fixture
def fake_store(mini_handbook, tmp_path):
    chunks = split_documents(load_handbook(mini_handbook), chunk_size=100, chunk_overlap=20)
    store, _ = build_index(chunks, fake_embeddings(), tmp_path / "index")
    return store


@pytest.fixture(scope="session")
def real_embeddings():
    """The real local embeddings model. Skips (with the fix) if it was never downloaded."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from llm import DEFAULT_LOCAL_EMBEDDINGS_MODEL

    try:
        return HuggingFaceEmbeddings(model_name=DEFAULT_LOCAL_EMBEDDINGS_MODEL)
    except Exception as error:  # not downloaded yet, and tests run offline
        pytest.skip(f"Local model not downloaded yet: run python step05_embed_three.py once online ({error!r})")


@pytest.fixture(scope="session")
def real_store(real_embeddings, tmp_path_factory):
    chunks = split_documents(load_handbook(HANDBOOK_DIR))
    store, _ = build_index(chunks, real_embeddings, tmp_path_factory.mktemp("real_index"))
    return store
