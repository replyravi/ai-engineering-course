"""
loaders.py - turn the cafe's files into LangChain Documents (Steps 3 to 7).

A LOADER reads one kind of file and gives back a list of Documents.
A DOCUMENT has two parts:
    page_content   the text
    metadata       a dict of labels about that text, like {"source": "data/menu.csv", "row": 7}
The metadata travels with the text all the way into the index, so an answer can cite it later.

    CSV   CSVLoader     one Document per ROW
    HTML  BSHTMLLoader  one Document per FILE (tags removed)
    PDF   PyPDFLoader   one Document per PAGE

These loaders live in langchain-community. That package was archived in 2026 and prints a
"being sunset" warning, but it still works, and no standalone loader package replaces it yet.
So we pin langchain-community==0.4.2 and import the loaders in this ONE file: if they move
one day, only this file changes.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from common import DATA_DIR, OUTPUTS_DIR

# Import the loaders and keep the package's sunset warning, so inspect_docs.py can show it to you once.
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    from langchain_community.document_loaders import BSHTMLLoader, CSVLoader, PyPDFLoader, TextLoader
SUNSET_WARNING = next((str(w.message) for w in _caught if "sunset" in str(w.message)), "")

MENU_CSV = DATA_DIR / "menu.csv"
ABOUT_HTML = DATA_DIR / "about.html"
POLICY_PDF = DATA_DIR / "policy.pdf"
EXPORT_FILE = OUTPUTS_DIR / "documents.jsonl"


def short_path(path):
    """'data/menu.csv' instead of the full path: tidier metadata, and the same on every computer."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:                      # the file is not inside the folder you run Python from
        return path.as_posix()


# ---------------------------------------------------------------------------
# Step 3 and 4: the menu (CSV)
# ---------------------------------------------------------------------------

def load_menu(path=MENU_CSV):
    """Step 3: one Document per row. Its text is one 'column: value' line per column."""
    return CSVLoader(short_path(path), encoding="utf-8").load()


def load_menu_named(path=MENU_CSV):
    """Step 4: metadata["source"] becomes the item name, and veg is copied into the metadata.

    Careful: a metadata column is also taken OUT of the text.
    """
    return CSVLoader(short_path(path), encoding="utf-8", source_column="item", metadata_columns=["veg"]).load()


def load_menu_keep_text(path=MENU_CSV):
    """Step 4, last try: list the text columns yourself (content_columns), so veg is in BOTH places."""
    columns = ["item", "price_rs", "veg", "contains_nuts", "spice", "description"]
    return CSVLoader(short_path(path), encoding="utf-8", source_column="item", metadata_columns=["veg"],
                     content_columns=columns).load()


# ---------------------------------------------------------------------------
# Step 5: the about page (HTML)
# ---------------------------------------------------------------------------

def load_about_like_windows(path=ABOUT_HTML):
    """Step 5, way 1: the settings you get on Windows when you pass nothing.

    open_encoding=None means "your computer's default". On Windows that is usually cp1252
    (on macOS and Linux it is UTF-8), so we name cp1252 to show everyone what Windows users see.
    get_text_separator="" is the loader's default: text from neighbouring tags is glued together.
    """
    return BSHTMLLoader(short_path(path), open_encoding="cp1252").load()


def load_about(path=ABOUT_HTML):
    """Step 5, way 2: read the file as UTF-8 and put a space between the text of neighbouring tags."""
    return BSHTMLLoader(short_path(path), open_encoding="utf-8", get_text_separator=" ").load()


def tidy(text):
    """Remove the empty space BSHTMLLoader leaves behind: trim every line and drop the empty ones."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Step 6: the policy (PDF)
# ---------------------------------------------------------------------------

def load_policy(path=POLICY_PDF):
    """Step 6: one Document per page. metadata["page"] counts from 0, metadata["page_label"] is the printed number."""
    return PyPDFLoader(short_path(path)).load()


# ---------------------------------------------------------------------------
# Step 7: a whole folder, one loader per file type
# ---------------------------------------------------------------------------

# file extension -> (loader class, its settings)
LOADERS = {
    ".csv": (CSVLoader, {"encoding": "utf-8-sig"}),        # utf-8-sig also reads Excel's "CSV UTF-8" files
    ".html": (BSHTMLLoader, {"open_encoding": "utf-8", "get_text_separator": " "}),
    ".htm": (BSHTMLLoader, {"open_encoding": "utf-8", "get_text_separator": " "}),
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
}


@dataclass
class FolderReport:
    """What load_folder() found: one row per file, plus the files it skipped or could not read."""

    rows: dict = field(default_factory=dict)        # file name -> {"loader", "documents", "characters", "keys"}
    skipped: list = field(default_factory=list)     # files with no loader in LOADERS
    failed: list = field(default_factory=list)      # files whose loader raised an error

    def add(self, file_name, loader_name, doc):
        row = self.rows.setdefault(file_name, {"loader": loader_name, "documents": 0, "characters": 0, "keys": set()})
        row["documents"] += 1
        row["characters"] += len(doc.page_content)
        row["keys"].update(doc.metadata)

    def table(self):
        """The report as lines of text (printed by inspect_docs.py folder)."""
        width = max([12, *(len(name) for name in self.rows)])
        lines = [f"{'file':<{width}} {'loader':<13} {'documents':>9} {'characters':>11}"]
        for name, row in self.rows.items():
            lines.append(f"{name:<{width}} {row['loader']:<13} {row['documents']:>9} {row['characters']:>11,}")
        docs = sum(row["documents"] for row in self.rows.values())
        chars = sum(row["characters"] for row in self.rows.values())
        lines.append(f"{'total':<{width}} {'':<13} {docs:>9} {chars:>11,}")
        lines.append(f"skipped (no loader for these): {', '.join(self.skipped) if self.skipped else 'none'}")
        lines.append(f"failed (the loader raised an error): {', '.join(self.failed) if self.failed else 'none'}")
        return lines


def loader_for(path):
    """The right loader for one file, chosen by its extension."""
    loader_class, settings = LOADERS[Path(path).suffix.lower()]
    return loader_class(short_path(path), **settings)


def lazy_load_folder(folder=DATA_DIR, report=None):
    """Yield the Documents of every file in `folder`, ONE AT A TIME (a generator).

    lazy_load() reads one Document, hands it over, then reads the next. A folder of
    10,000 pages never has to sit in memory at once.
    report: a FolderReport that counts what was read. With a report, a file that cannot be read
    is written down in report.failed and the next file is read. Without one, its error stops the loop.
    """
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue                                   # skip sub-folders and hidden files like .DS_Store
        if path.suffix.lower() not in LOADERS:
            if report is not None:
                report.skipped.append(path.name)
            continue
        loader = loader_for(path)
        try:
            for doc in loader.lazy_load():
                doc.metadata["file"] = path.name       # our own key: a short file name for citations
                if isinstance(loader, BSHTMLLoader):
                    doc.page_content = tidy(doc.page_content)
                if report is not None:
                    report.add(path.name, type(loader).__name__, doc)
                yield doc
        except Exception as error:                     # one broken file must not stop the whole folder
            if report is None:
                raise
            report.failed.append(f"{path.name} ({type(error).__name__})")


def load_folder(folder=DATA_DIR):
    """Load every file in `folder`. Returns (documents, report)."""
    report = FolderReport()
    docs = list(lazy_load_folder(folder, report))
    return docs, report


def export_jsonl(docs, path=EXPORT_FILE):
    """Write one Document per line as JSON (JSON Lines), so you can open what was loaded in any editor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as file:
        for doc in docs:
            file.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False))
            file.write("\n")
            count += 1
    return count
