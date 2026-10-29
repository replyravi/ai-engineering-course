"""
inspect_docs.py - look inside the Documents each loader makes (Steps 2 to 8). No API key needed.

    python inspect_docs.py files            Step 2: the three cafe files in data/
    python inspect_docs.py csv              Step 3: the menu, one Document per row
    python inspect_docs.py csv-named        Step 4: source_column and metadata_columns
    python inspect_docs.py html             Step 5: the about page, loaded two ways
    python inspect_docs.py pdf              Step 6: the policy, one Document per page
    python inspect_docs.py folder           Step 7: the whole folder, with a report
    python inspect_docs.py folder --export  Step 7: ... and save every Document to outputs/documents.jsonl
    python inspect_docs.py folder --folder tests/fixtures
                                            Step 8: a messier folder (an Excel CSV, a .txt, a .xyz, a broken .pdf)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import codecs
import locale
from pathlib import Path

from common import DATA_DIR
from loaders import (
    ABOUT_HTML, EXPORT_FILE, LOADERS, MENU_CSV, POLICY_PDF, SUNSET_WARNING, FolderReport, PyPDFLoader, export_jsonl,
    lazy_load_folder, load_about, load_about_like_windows, load_folder, load_menu, load_menu_keep_text,
    load_menu_named, load_policy, short_path, tidy,
)

KINDS = {".csv": "a table (rows and columns)", ".html": "a web page", ".pdf": "a document with pages"}


def show_document(doc, label, max_lines=8):
    """Print the two parts of a Document: its text (page_content) and its labels (metadata)."""
    print(label)
    print("  page_content:")
    lines = doc.page_content.splitlines() or [""]
    for line in lines[:max_lines]:
        print(f"    {line}")
    if len(lines) > max_lines:
        print(f"    ... ({len(lines) - max_lines} more lines)")
    print(f"  metadata: {doc.metadata}")


def between(text, first, last):
    """The piece of `text` from the words `first` to the words `last`, spaces squeezed (to compare two loads)."""
    flat = " ".join(text.split())
    start = flat.find(first)
    end = flat.find(last, start)
    return flat[start:end + len(last)] if start >= 0 and end >= 0 else "(not found)"


def show_files():
    print(f"Files in {short_path(DATA_DIR)}/:")
    for path in sorted(DATA_DIR.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            print(f"  {path.name:<12} {path.stat().st_size:>7,} bytes   {KINDS.get(path.suffix, 'other')}")


def show_csv():
    if SUNSET_WARNING:
        print(f"langchain-community says: {SUNSET_WARNING.split('. ')[0]}.")
        print("(It still works. We pin version 0.4.2. More in the README, Step 3.)\n")
    docs = load_menu()
    print(f"CSVLoader made {len(docs)} Documents from {short_path(MENU_CSV)} (one per row)\n")
    show_document(docs[0], "Document 1:")
    print()
    show_document(docs[7], "Document 8 (row 7: rows count from 0):")


def show_csv_named():
    docs = load_menu_named()
    print('CSVLoader(..., source_column="item", metadata_columns=["veg"])\n')
    show_document(docs[0], "Document 1:")
    print("\nNotice: source is now the item name (the file path is gone), and veg moved OUT of the text.")
    keep = load_menu_keep_text()
    print('\nCSVLoader(..., source_column="item", metadata_columns=["veg"], content_columns=[all six columns])\n')
    show_document(keep[0], "Document 1:")
    veg_items = [doc.metadata["source"] for doc in docs if doc.metadata["veg"] == "no"]
    print(f"\nA filter on metadata: items with veg = no -> {veg_items}")


def show_html():
    default = codecs.lookup(locale.getpreferredencoding(False)).name     # e.g. utf-8 or cp1252
    print(f"Your computer's default text encoding: {default}")
    print("(Python uses it when you give no encoding. On Windows it is usually cp1252.)\n")
    ways = [
        ('Way 1: open_encoding="cp1252" (Windows default), get_text_separator="" (the default)',
         load_about_like_windows()[0]),
        ('Way 2: open_encoding="utf-8", get_text_separator=" "', load_about()[0]),
    ]
    for title, doc in ways:
        print(title)
        print(f"  apostrophe:  {between(doc.page_content, 'bus stop.', 'standing')}")
        print(f"  rupee sign:  {between(doc.page_content, 'less than', '50.')}")
        print(f"  list items:  {between(doc.page_content, 'Ginger crushed', 'twice a day')}")
        print(f"  menu bar:    {between(doc.page_content, 'Home', 'About us')}")
        print(f"  metadata:    {doc.metadata}")
        print(f"  characters:  {len(doc.page_content)}\n")
    tidied = tidy(ways[1][1].page_content)
    print(f"After tidy() (trim every line, drop empty lines): {len(tidied)} characters. The first 4 lines:")
    for line in tidied.splitlines()[:4]:
        print(f"    {line}")


def show_pdf():
    docs = load_policy()
    print(f"PyPDFLoader made {len(docs)} Documents from {short_path(POLICY_PDF)} (one per page)\n")
    for doc in docs:
        meta = doc.metadata
        first = doc.page_content.splitlines()[0]
        print(f"  page={meta['page']}  page_label={meta['page_label']!r}  total_pages={meta['total_pages']}  "
              f"{len(doc.page_content):,} characters  starts: {first[:45]}")
    print(f"\nAll metadata keys: {sorted(docs[0].metadata)}")
    show_document(docs[0], "\nDocument 1 (page 0):", max_lines=4)

    # The old shortcut. It is not "get the pages": it loads, then runs a text splitter.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(short_path(POLICY_PDF))
    default = loader.load_and_split()
    small = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0))
    print(f"\nload_and_split() gave {len(default)} pieces: one per page here only because each page is")
    print("shorter than its hidden splitter's chunk_size (4,000 characters).")
    print(f"load_and_split(RecursiveCharacterTextSplitter(chunk_size=400)) gave {len(small)} pieces.")


def show_folder(folder=DATA_DIR, export=False):
    print(f"Loaders by file type: { {ext: cls.__name__ for ext, (cls, _) in LOADERS.items()} }\n")
    docs, report = load_folder(folder)
    print(f"Folder: {short_path(folder)}/")
    for line in report.table():
        print(line)
    print("\nMetadata keys on each file's Documents ('file' is our own key, the loader added the rest):")
    for name, row in report.rows.items():
        print(f"  {name:<20} {', '.join(sorted(row['keys']))}")

    stream = lazy_load_folder(folder, FolderReport())   # a generator: nothing is read yet
    first = next(stream)                            # now ONE Document is read
    print(f"\nlazy_load_folder() is a {type(stream).__name__}. next() read one Document: "
          f"{first.metadata['file']}, {len(first.page_content)} characters.")
    if export:
        count = export_jsonl(lazy_load_folder(folder, FolderReport()))
        print(f"Saved {count} Documents to {short_path(EXPORT_FILE)} (one JSON object per line).")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Look inside the Documents each loader makes.")
    parser.add_argument("mode", choices=["files", "csv", "csv-named", "html", "pdf", "folder"])
    parser.add_argument("--export", action="store_true", help="folder mode: also save outputs/documents.jsonl")
    parser.add_argument("--folder", type=Path, default=DATA_DIR, help="folder mode: which folder (default data/)")
    args = parser.parse_args(argv)
    actions = {"files": show_files, "csv": show_csv, "csv-named": show_csv_named, "html": show_html, "pdf": show_pdf}
    if args.mode == "folder":
        if not args.folder.is_dir():
            print(f"No folder called {args.folder}. Run this from the ep10-counter-bot folder.")
            return 1
        show_folder(args.folder, args.export)
    else:
        actions[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
