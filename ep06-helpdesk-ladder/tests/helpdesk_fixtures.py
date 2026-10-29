"""
tests/helpdesk_fixtures.py - shared test helpers for this episode.

Test files import what they need, for example:
    from helpdesk_fixtures import KeywordEmbeddings, fixed_today, no_keys  # noqa: F401

- no_keys:            no provider settings and no keys (even if your real .env has one)
- fixed_today:        the holiday tool thinks today is Friday 23 October 2026
- mini_handbook:      a tiny handbook file with a heading and a comment (to test the splitter)
- KeywordEmbeddings:  tiny offline embeddings that count a few words, so ranking MEANS something
- real_embeddings:    the free local all-MiniLM-L6-v2 (skips with the fix if never downloaded)
"""

import re

import pytest
from langchain_core.embeddings import Embeddings

FIXED_TODAY = "2026-10-23"

MINI_HANDBOOK = """# Mini handbook

<!-- a comment that must be skipped -->

## Meals

Cut-off: skip lunch before 9:30 am.

Pausing: pause up to
10 days a month.
"""


@pytest.fixture
def no_keys(monkeypatch):
    for name in ["LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fixed_today(monkeypatch):
    monkeypatch.setenv("HELPDESK_TODAY", FIXED_TODAY)
    return FIXED_TODAY


@pytest.fixture
def mini_handbook(tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text(MINI_HANDBOOK, encoding="utf-8")
    return path


class KeywordEmbeddings(Embeddings):
    """Offline embeddings for tests: one number per keyword (how often it appears).

    Not a real model, but texts that share our keywords really do point the same way,
    so a ranking test can check that the RIGHT paragraph comes first.
    """

    KEYWORDS = ["skip", "lunch", "cut", "pause", "holiday", "deposit", "box", "chat", "missed", "zone"]

    def _vector(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        return [0.01 + sum(w.startswith(k) for w in words) for k in self.KEYWORDS]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture(scope="session")
def real_embeddings():
    """The real local embeddings model. Skips (with the fix) if it was never downloaded."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from llm import DEFAULT_LOCAL_EMBEDDINGS_MODEL

    try:
        return HuggingFaceEmbeddings(model_name=DEFAULT_LOCAL_EMBEDDINGS_MODEL)
    except Exception as error:  # not downloaded yet, and tests run offline
        pytest.skip(f"Local model not downloaded yet: run python rung3_context.py once online ({error!r})")
