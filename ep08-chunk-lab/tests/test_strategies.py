"""
tests/test_strategies.py - the four splitters, on a tiny fixture with counts you can check by hand,
and on the real handbook with the exact numbers shown in the README.
"""

import inspect
import logging

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter
from lab_fixtures import MINI_MD, minilm_tokenizer, tiktoken_ready  # noqa: F401

from chunklab.strategies import (
    HANDBOOK_DIR, STRATEGIES, breadcrumb, citation, count_tokens, load_handbook, make_chunks, recursive_splitter,
    shared_text, split_by_headers, split_headers_then_recursive,
)


def test_character_splitter_cuts_only_at_blank_lines_then_glues():
    chunks = CharacterTextSplitter(separator="\n\n", chunk_size=40, chunk_overlap=0).split_text(MINI_MD)

    # Pieces of 19, 33, 7, 30, 8 and 19 characters: "## Rain" (7) + "\n\n" + 30 = 39 fits in 40, so they are glued.
    assert chunks == [
        "# Night Shift Notes",
        "Riders start at 7 pm, no earlier.",
        "## Rain\n\nSlots can shift by 60 minutes.",
        "## Boxes\n\nRinse the box only.",
    ]


def test_character_splitter_keeps_a_long_piece_whole_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        chunks = CharacterTextSplitter(separator="\n\n", chunk_size=25, chunk_overlap=0).split_text(MINI_MD)

    assert "Riders start at 7 pm, no earlier." in chunks           # 33 characters, bigger than 25, kept whole
    assert "Created a chunk of size 33, which is longer than the specified 25" in caplog.text


def test_recursive_splitter_respects_the_size_and_repeats_the_overlap():
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = recursive_splitter(size=20, overlap=8).split_text(text)

    assert all(len(chunk) <= 20 for chunk in chunks)
    # The last word or two of each chunk is repeated at the start of the next one (the overlap).
    assert chunks == ["one two three four", "four five six seven", "seven eight nine", "nine ten eleven",
                      "eleven twelve"]
    assert shared_text(chunks[0], chunks[1], minimum=4) == "four"


def test_token_splitter_default_encoding_is_gpt2():
    # Why we always pass encoding_name: the default is the old GPT-2 tokenizer.
    assert inspect.signature(TokenTextSplitter.__init__).parameters["encoding_name"].default == "gpt2"


def test_token_splitter_counts_tiktoken_tokens(tiktoken_ready):
    chunks = make_chunks("tokens")

    assert len(chunks) == 44
    assert max(len(tiktoken_ready.encode(chunk.page_content)) for chunk in chunks) == 100


def test_header_splitter_puts_headings_into_the_keys_you_chose():
    headers = [("#", "h1"), ("##", "h2")]
    kept = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False).split_text(MINI_MD)
    stripped = MarkdownHeaderTextSplitter(headers_to_split_on=headers).split_text(MINI_MD)

    assert [doc.metadata for doc in kept] == [
        {"h1": "Night Shift Notes"},
        {"h1": "Night Shift Notes", "h2": "Rain"},
        {"h1": "Night Shift Notes", "h2": "Boxes"},
    ]
    assert kept[1].page_content.startswith("## Rain")                   # strip_headers=False keeps the heading
    assert stripped[1].page_content == "Slots can shift by 60 minutes."   # the default removes it
    assert "header_type" not in kept[1].metadata                        # a key that does not exist


def test_split_by_headers_adds_the_file_name(tmp_path):
    (tmp_path / "notes.md").write_text(MINI_MD, encoding="utf-8")
    sections = split_by_headers(load_handbook(tmp_path))

    assert [section.metadata["source"] for section in sections] == ["notes.md"] * 3
    assert breadcrumb(sections[1]) == "Night Shift Notes > Rain"
    assert citation(Document(page_content="x", metadata={**sections[1].metadata, "chunk": 2})) == "notes.md > Rain"


def test_long_sections_are_split_and_every_part_keeps_its_metadata(minilm_tokenizer):
    long_section = "# Guide\n\n## Rules\n\n" + "\n".join(f"Rule {i}: wash your hands before packing box {i}."
                                                         for i in range(60))
    chunks = split_headers_then_recursive([Document(page_content=long_section, metadata={"source": "g.md"})])

    assert len(chunks) > 1
    assert all(chunk.metadata["h2"] == "Rules" and chunk.metadata["source"] == "g.md" for chunk in chunks)
    assert chunks[0].metadata["part"] == f"1 of {len(chunks)}"
    assert all(count_tokens(chunk.page_content) <= 200 for chunk in chunks)


def test_real_handbook_counts_match_the_readme(minilm_tokenizer, tiktoken_ready):
    docs = load_handbook()
    expected = {"character": 51, "recursive": 54, "tokens": 44, "headers": 43}   # the README's numbers

    assert [doc.metadata["source"] for doc in docs] == [
        "delivery.md", "food_safety.md", "plans.md", "refunds.md", "support.md"]
    assert {strategy: len(make_chunks(strategy, docs)) for strategy in expected} == expected
    assert len(split_by_headers(docs)) == 42


def test_no_chunk_is_longer_than_the_embedder_reads(minilm_tokenizer, tiktoken_ready):
    docs = load_handbook()
    for strategy in STRATEGIES:
        assert max(count_tokens(chunk.page_content) for chunk in make_chunks(strategy, docs)) <= 254


def test_the_refund_rule_lost_its_heading_in_the_size_based_split():
    chunks = make_chunks("recursive")
    rule = next(chunk for chunk in chunks if "Report a missed lunch or dinner" in chunk.page_content)

    assert rule.metadata["source"] == "refunds.md" and rule.metadata["chunk"] == 4
    assert "Missed or late meal" not in rule.page_content     # its heading went into refunds.md#3


def test_pdf_converter_rebuilds_the_saved_markdown():
    from tools.pdf_to_markdown import pdf_to_markdown

    assert pdf_to_markdown() == (HANDBOOK_DIR / "food_safety.md").read_text(encoding="utf-8")
