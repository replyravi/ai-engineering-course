"""
Grounding tests (Step 14): the loaded Documents go into the index, keep their metadata, and come back
as citation tags. The search tests use the REAL local embeddings model (already downloaded, no internet);
the chat model is always a fake.
"""

import pytest
from langchain_core.documents import Document

import cafe_tools
from cafe_index import build_index, chunk_id, citation, search, split
from common import CAFE_RULES, GROUNDING_RULES
from loaders import load_folder
from memory_modes import build_bot
from testing.fakes import fake_llm, tool_call

KACHORI = "Is the kachori nut-free?"


@pytest.fixture(scope="module")
def chunks():
    docs, _ = load_folder()
    return split(docs)


@pytest.fixture(scope="module")
def real_store(chunks, tmp_path_factory):
    """The cafe index, built in a temp folder with the real MiniLM model."""
    from llm import get_embeddings

    try:
        embeddings = get_embeddings()
        embeddings.embed_query("test")
    except OSError:
        pytest.skip("Local model not downloaded yet: run  python cafe_index.py  once with internet.")
    return build_index(chunks, embeddings, tmp_path_factory.mktemp("index"))


def test_citation_tags_come_from_the_loader_metadata():
    assert citation(Document("x", metadata={"file": "menu.csv", "row": 7})) == "menu.csv row 7"
    assert citation(Document("x", metadata={"file": "policy.pdf", "page": 0, "page_label": "1"})) == "policy.pdf page 1"
    assert citation(Document("x", metadata={"file": "about.html", "title": "About"})) == "about.html"


def test_splitting_keeps_every_label(chunks):
    by_file = {}
    for chunk in chunks:
        by_file.setdefault(chunk.metadata["file"], []).append(chunk)

    assert {name: len(items) for name, items in by_file.items()} == {"about.html": 3, "menu.csv": 12, "policy.pdf": 7}
    assert [c.metadata["row"] for c in by_file["menu.csv"]] == list(range(12))     # a row is never cut
    assert all("page_label" in c.metadata for c in by_file["policy.pdf"])
    assert len({chunk_id(c) for c in chunks}) == len(chunks)                       # every id is unique


def test_the_kachori_question_finds_the_menu_row_and_the_kitchen_rule(real_store):
    hits = search(real_store, KACHORI, k=cafe_tools.TOOL_K)

    tags = [citation(doc) for doc, _ in hits]
    assert tags[0] == "menu.csv row 7"                                             # the kachori row, first
    assert any("not nut-free" in doc.page_content for doc, _ in hits)             # the policy rule is in the top 4


def test_building_twice_does_not_duplicate_chunks(chunks, real_store, tmp_path):
    from llm import get_embeddings

    embeddings = get_embeddings()
    build_index(chunks, embeddings, tmp_path)
    store = build_index(chunks, embeddings, tmp_path)
    assert len(store.get(include=[])["ids"]) == len(chunks) == 22


def test_the_search_tool_returns_tagged_text(real_store, monkeypatch):
    monkeypatch.setattr(cafe_tools, "_store", lambda: real_store)

    text = cafe_tools.search_cafe_docs.invoke({"question": KACHORI})

    assert text.startswith("[menu.csv row 7]\nitem: Khasta kachori")
    assert text.count("[") == cafe_tools.TOOL_K


def test_a_failing_search_returns_an_error_message_instead_of_crashing(monkeypatch):
    def broken():
        raise RuntimeError("index folder missing")

    monkeypatch.setattr(cafe_tools, "_store", broken)
    assert cafe_tools.search_cafe_docs.invoke({"question": KACHORI}).startswith("Error: could not search")


def test_todays_specials_reads_the_kitchen_file(tmp_path):
    assert "Methi thepla with curd: Rs 45" in cafe_tools.read_specials()
    assert cafe_tools.read_specials(tmp_path / "missing.csv").startswith("Error: could not read")


def test_the_grounded_bot_answers_from_the_tool_result(real_store, monkeypatch):
    monkeypatch.setattr(cafe_tools, "_store", lambda: real_store)
    llm = fake_llm(
        tool_call("search_cafe_docs", {"question": KACHORI}),
        "The kachori recipe has no nuts [menu.csv row 7], but our kitchen is not nut-free [policy.pdf page 1].",
    )
    bot = build_bot("buffer", llm=llm, tools=cafe_tools.TOOLS, system_prompt=CAFE_RULES + GROUNDING_RULES)

    result = bot.invoke({"messages": [{"role": "user", "content": KACHORI}]}, {"configurable": {"thread_id": "asha"}})

    second_call = llm.received[1]                     # what the model read before writing its answer
    tool_result = [m for m in second_call if m.type == "tool"][0]
    assert tool_result.text.startswith("[menu.csv row 7]")
    assert "not nut-free" in tool_result.text
    assert "[menu.csv row 7]" in result["messages"][-1].text
    assert "search_cafe_docs first" in second_call[0].text          # the grounding rules are in the system prompt
