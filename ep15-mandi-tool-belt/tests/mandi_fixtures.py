"""
tests/mandi_fixtures.py - shared test helpers for Episode 15 (import the fixtures you need).

    temp_db      a fresh mandi.db in a temporary folder (the real one is never touched)
    fake_store   the fixture crawl indexed with FAKE embeddings (fast, offline, meaning not tested)
    no_keys      no API keys and no .env values, so get_llm() shows the friendly help
"""

from pathlib import Path

import pytest

import seed_db
from ingest import FIXTURE, build_index, load_pages, split_pages
from testing.fakes import fake_embeddings

EPISODE = Path(__file__).resolve().parents[1]
COPY_MARKER = "# Everything below the next line is the original file, unchanged.\n"


def without_header(path):
    """A copied file minus its header note (everything up to the dashed line after the marker)."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert COPY_MARKER in lines, "the copy lost its header note"
    return "".join(lines[lines.index(COPY_MARKER) + 2:])


KEY_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER",
             "LLM_MODEL", "APIFY_API_TOKEN", "CRAWL_START_URL"]


@pytest.fixture
def temp_db(tmp_path):
    path = tmp_path / "mandi.db"
    seed_db.seed(path)
    return path


@pytest.fixture
def fake_store(tmp_path):
    docs, _ = load_pages(FIXTURE)
    store, _ = build_index(split_pages(docs), fake_embeddings(), index_dir=tmp_path / "index")
    return store


@pytest.fixture
def no_keys(monkeypatch):
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
