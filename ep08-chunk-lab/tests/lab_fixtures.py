"""
tests/lab_fixtures.py - shared test helpers for this episode.

Test files import the ones they need, for example:
    from lab_fixtures import minilm_tokenizer, tiktoken_ready  # noqa: F401

- MINI_MD:          a tiny Markdown file with exact, hand-countable pieces
- minilm_tokenizer: MiniLM's tokenizer from your disk (skips, with the fix, if it was never downloaded)
- tiktoken_ready:   the cl100k_base file from tiktoken_cache/ (skips, with the fix, if it was never downloaded)
- real_embeddings:  the real local MiniLM model (skips if it was never downloaded)

The fixtures are function-scoped on purpose: then conftest.py has already blocked the internet when they
run, so a missing file makes the test SKIP instead of quietly downloading it.
- breadcrumb_docs:  three hand-made chunks with header metadata (no model needed)
- fake_store:       those chunks in an in-memory Chroma collection with FAKE embeddings
"""

import os
from functools import lru_cache

import pytest
from langchain_core.documents import Document

# FAISS and PyTorch each bring their own copy of the OpenMP library. Without this setting, using both in
# one process can stop Python with "OMP: Error #15" (seen on macOS). scikit-learn sets the same value
# when it is imported, and chunklab/faiss_compare.py sets it too.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from testing.fakes import fake_embeddings

# Six pieces separated by blank lines (3 headings, 3 sentences), 19, 33, 7, 30, 8 and 19 characters long.
MINI_MD = """# Night Shift Notes

Riders start at 7 pm, no earlier.

## Rain

Slots can shift by 60 minutes.

## Boxes

Rinse the box only.
"""


@pytest.fixture
def minilm_tokenizer():
    from chunklab.strategies import tokenizer

    try:
        return tokenizer()
    except Exception as error:  # not downloaded yet, and tests run offline
        pytest.skip(f"MiniLM not downloaded yet: run  python check_env.py  once online ({error!r})")


@pytest.fixture
def tiktoken_ready():
    import tiktoken

    import chunklab.strategies  # noqa: F401  (sets TIKTOKEN_CACHE_DIR to tiktoken_cache/)

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as error:  # not downloaded yet, and tests run offline
        pytest.skip(f"tiktoken file not downloaded yet: run  python -m chunklab show tokens  once online ({error!r})")


@lru_cache(maxsize=1)
def _load_minilm():
    from langchain_huggingface import HuggingFaceEmbeddings

    from chunklab.strategies import EMBEDDER

    return HuggingFaceEmbeddings(model_name=EMBEDDER)


@pytest.fixture
def real_embeddings(minilm_tokenizer):
    try:
        return _load_minilm()           # loaded once, then reused by every test
    except Exception as error:
        pytest.skip(f"MiniLM not downloaded yet: run  python check_env.py  once online ({error!r})")


@pytest.fixture
def breadcrumb_docs():
    """Three chunks as the header strategy makes them: text plus source, h1, h2 and chunk number."""
    return [
        Document(page_content="## Missed or late meal\nReport a missed lunch within 6 hours of your slot.",
                 metadata={"source": "refunds.md", "h1": "Refunds Policy", "h2": "Missed or late meal", "chunk": 3}),
        Document(page_content="## Pausing your plan\nPause for up to 10 days a month.",
                 metadata={"source": "plans.md", "h1": "Meal Plans", "h2": "Pausing your plan", "chunk": 5}),
        Document(page_content="## 7. If something seems wrong\nIf the food smells sour, do not eat it.",
                 metadata={"source": "food_safety.md", "h1": "Food Safety Guide", "h2": "7. If something seems wrong",
                           "chunk": 8}),
    ]


@pytest.fixture
def fake_store(breadcrumb_docs):
    from chunklab.score import in_memory_store

    return in_memory_store("test", breadcrumb_docs, fake_embeddings())
