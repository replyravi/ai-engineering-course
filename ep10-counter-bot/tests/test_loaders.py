"""
Loader tests (Steps 3 to 8): tiny fixture files in tests/fixtures, no key, no internet.
Each test checks one thing a loader does: how many Documents, which metadata keys, which text.
"""

import inspect
import json
import re
from pathlib import Path

import pytest
from langchain_core.documents import Document

from loaders import (
    BSHTMLLoader, CSVLoader, PyPDFLoader, export_jsonl, lazy_load_folder, load_about, load_folder, load_menu,
    load_menu_keep_text, load_menu_named, load_policy, tidy,
)

FIXTURES = Path(__file__).parent / "fixtures"
TINY_MENU = str(FIXTURES / "tiny_menu.csv")
EXCEL_MENU = str(FIXTURES / "tiny_menu_excel.csv")
TINY_PAGE = str(FIXTURES / "tiny_page.html")
TWO_PAGES = str(FIXTURES / "two_pages.pdf")


# --- CSV (Steps 3 and 4) ---

def test_csv_gives_one_document_per_row():
    docs = CSVLoader(TINY_MENU, encoding="utf-8").load()

    assert len(docs) == 3                                     # 3 rows; the header line is not a Document
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "item: Tea\nprice_rs: 10\nveg: yes"
    assert docs[1].metadata == {"source": TINY_MENU, "row": 1}   # rows count from 0


def test_the_cafe_menu_has_12_rows_and_row_7_is_the_kachori():
    docs = load_menu()

    assert len(docs) == 12
    assert docs[7].metadata["row"] == 7
    assert docs[7].page_content.startswith("item: Khasta kachori")


def test_source_column_replaces_the_file_name_and_metadata_columns_leave_the_text():
    named = load_menu_named()[0]
    assert named.metadata == {"source": "Masala chai", "row": 0, "veg": "yes"}
    assert "veg:" not in named.page_content                   # a metadata column is taken OUT of the text

    kept = load_menu_keep_text()[0]
    assert kept.metadata["veg"] == "yes"
    assert "veg: yes" in kept.page_content                    # content_columns keeps it in the text too


def test_a_csv_saved_from_excel_needs_utf_8_sig():
    plain = CSVLoader(EXCEL_MENU, encoding="utf-8").load()
    assert plain[0].page_content.startswith("\ufeffitem: Tea")    # the invisible BOM sticks to the first column

    with pytest.raises(RuntimeError) as error:
        CSVLoader(EXCEL_MENU, encoding="utf-8", source_column="item").load()
    assert "Source column 'item' not found" in str(error.value.__cause__)

    fixed = CSVLoader(EXCEL_MENU, encoding="utf-8-sig", source_column="item").load()
    assert fixed[0].metadata["source"] == "Tea"


# --- HTML (Step 5) ---

def test_html_default_separator_glues_words_and_a_space_fixes_it():
    glued = BSHTMLLoader(TINY_PAGE, open_encoding="utf-8").load()
    spaced = BSHTMLLoader(TINY_PAGE, open_encoding="utf-8", get_text_separator=" ").load()

    assert len(glued) == 1                                    # one Document for the whole page
    assert "TeaToast" in glued[0].page_content
    assert "Tea Toast" in spaced[0].page_content
    assert spaced[0].metadata == {"source": TINY_PAGE, "title": "Tiny page"}


def test_the_wrong_encoding_garbles_the_apostrophe():
    windows = BSHTMLLoader(TINY_PAGE, open_encoding="cp1252").load()[0]
    utf8 = BSHTMLLoader(TINY_PAGE, open_encoding="utf-8").load()[0]

    assert "It\u00e2\u20ac\u2122s hot" in windows.page_content   # what a typical Windows laptop shows
    assert "It\u2019s hot" in utf8.page_content


def test_the_about_page_has_its_title_and_tidy_removes_empty_lines():
    doc = load_about()[0]
    assert doc.metadata["title"] == "About Kadak Kettle"

    tidied = tidy(doc.page_content)
    assert "" not in tidied.splitlines()
    assert "Home Menu About us" in tidied


# --- PDF (Step 6) ---

def test_pdf_gives_one_document_per_page_counting_from_zero():
    docs = PyPDFLoader(TWO_PAGES).load()

    assert len(docs) == 2
    assert [doc.metadata["page"] for doc in docs] == [0, 1]
    assert [doc.metadata["page_label"] for doc in docs] == ["1", "2"]
    assert docs[0].metadata["total_pages"] == 2
    assert "Page one says hello." in docs[0].page_content
    assert "Page two says goodbye." in docs[1].page_content


def test_the_policy_pdf_has_two_clean_pages():
    docs = load_policy()

    assert len(docs) == 2
    text = " ".join(doc.page_content for doc in docs)
    assert "Our kitchen is not nut-free" in text
    assert re.findall(r"\bT [a-z]", text) == []               # no "T ell"-style gaps (see tools/make_policy_pdf.py)


def test_load_and_split_runs_a_splitter_it_does_not_give_pages():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader("data/policy.pdf")
    assert len(loader.load_and_split()) == 2                   # each page is under its hidden 4,000-character limit
    small = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0))
    assert len(small) > 2                                     # same call, more pieces: it is a splitter


# --- A whole folder (Steps 7 and 8) ---

def test_load_folder_reports_every_file_and_skips_the_ones_it_cannot_read():
    docs, report = load_folder(FIXTURES)

    counts = {name: row["documents"] for name, row in report.rows.items()}
    assert counts == {"notes.txt": 1, "tiny_menu.csv": 3, "tiny_menu_excel.csv": 3, "tiny_page.html": 1,
                      "two_pages.pdf": 2}
    assert report.rows["two_pages.pdf"]["loader"] == "PyPDFLoader"
    assert report.skipped == ["unknown.xyz"]                  # no loader for .xyz
    assert report.failed == ["broken.pdf (PdfStreamError)"]   # a .pdf that is not a PDF
    assert len(docs) == 10
    assert all("file" in doc.metadata for doc in docs)        # our own key, for citations
    excel = [doc for doc in docs if doc.metadata["file"] == "tiny_menu_excel.csv"]
    assert excel[0].page_content.startswith("item: Tea")      # the folder loader reads Excel CSVs with utf-8-sig


def test_the_cafe_folder_gives_15_documents():
    docs, report = load_folder()

    assert {name: row["documents"] for name, row in report.rows.items()} == {
        "about.html": 1, "menu.csv": 12, "policy.pdf": 2}
    assert len(docs) == 15
    assert report.skipped == [] and report.failed == []
    assert report.table()[0].split() == ["file", "loader", "documents", "characters"]


def test_lazy_load_folder_hands_over_one_document_at_a_time():
    stream = lazy_load_folder()

    assert inspect.isgenerator(stream)
    first = next(stream)
    assert first.metadata["file"] == "about.html"             # files are read in name order


def test_without_a_report_a_broken_file_stops_the_loop():
    with pytest.raises(Exception):
        list(lazy_load_folder(FIXTURES))


def test_export_jsonl_writes_one_document_per_line(tmp_path):
    docs, _ = load_folder()
    path = tmp_path / "documents.jsonl"

    count = export_jsonl(docs, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert count == len(lines) == 15
    row = json.loads(lines[1])
    assert row["metadata"]["file"] == "menu.csv" and row["metadata"]["row"] == 0
    assert row["page_content"].startswith("item: Masala chai")
