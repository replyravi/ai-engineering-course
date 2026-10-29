"""
ingest.py - the INDEXING pipeline: load -> split -> embed -> store.

Episode 12 note: copied from Episode 07 (ep07-ask-the-handbook/ingest.py). The code is
unchanged, so this folder builds the same 54-chunk index from the same handbook.
rag.py imports these functions instead of repeating them.

Run it once, and again whenever a file in handbook/ changes:

    python ingest.py

It reads the Dabbaloop handbook (4 Markdown files and 1 PDF), cuts it into
overlapping chunks, turns every chunk into 384 numbers with a free local model,
and saves everything in a Chroma index in the index/ folder.
Running it twice does NOT create duplicates: every chunk gets a stable id.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import hashlib
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma telemetry off (only matters before Chroma 1.5.4)

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from llm import get_embeddings  # noqa: E402

HERE = Path(__file__).resolve().parent
HANDBOOK_DIR = HERE / "handbook"
INDEX_DIR = HERE / "index"
COLLECTION = "dabbaloop_handbook"

CHUNK_SIZE = 400     # characters per chunk (at most)
CHUNK_OVERLAP = 60   # characters that neighbouring chunks may share


class IndexNotBuiltError(RuntimeError):
    """Raised when someone searches before running ingest.py."""


# ---------------------------------------------------------------------------
# Step 3 - Load: every file becomes Document(s) with a "source" label
# ---------------------------------------------------------------------------

def load_handbook(folder=HANDBOOK_DIR):
    """Read every .md file (one Document each) and every .pdf (one Document per page)."""
    docs = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": path.name}))
        elif path.suffix.lower() == ".pdf":
            for page_number, page in enumerate(PdfReader(path).pages, start=1):
                docs.append(Document(
                    page_content=page.extract_text(),
                    metadata={"source": path.name, "page": page_number},
                ))
    return docs


# ---------------------------------------------------------------------------
# Step 4 - Split: overlapping chunks that remember where they came from
# ---------------------------------------------------------------------------

def section_of(text, position):
    """Return the last Markdown heading (# or ##) that starts at or before `position`."""
    heading = ""
    for match in re.finditer(r"^#{1,3} (.+)$", text, flags=re.MULTILINE):
        if match.start() > position:
            break
        heading = match.group(1).strip()
    return heading


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Cut documents into chunks. Each chunk keeps its source and gets a chunk number.

    Markdown chunks also get the section heading they belong to; PDF chunks keep their page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,   # remember where each chunk starts in its document
    )
    chunks = []
    numbers = {}  # source file -> how many chunks so far
    for doc in docs:
        for chunk in splitter.split_documents([doc]):
            source = chunk.metadata["source"]
            numbers[source] = numbers.get(source, 0) + 1
            chunk.metadata["chunk"] = numbers[source]
            if source.endswith(".md"):
                chunk.metadata["section"] = section_of(doc.page_content, chunk.metadata["start_index"])
            chunks.append(chunk)
    return chunks


def shared_text(first, second):
    """Return the overlap: the longest end of `first` that is also the start of `second`."""
    for size in range(min(len(first), len(second)), 0, -1):
        if first[-size:] == second[:size]:
            return second[:size]
    return ""


# ---------------------------------------------------------------------------
# Labels used for citations, e.g. [refunds.md#3]
# ---------------------------------------------------------------------------

def source_tag(doc):
    """The short citation tag of a chunk: file name + chunk number."""
    return f"{doc.metadata['source']}#{doc.metadata['chunk']}"


def source_where(doc):
    """Where the chunk is in its file: the page for PDFs, the section for Markdown."""
    if "page" in doc.metadata:
        return f"page {doc.metadata['page']}"
    return f"section: {doc.metadata.get('section') or 'top of file'}"


# ---------------------------------------------------------------------------
# Step 6 - Embed + store: a persistent Chroma index with stable ids
# ---------------------------------------------------------------------------

def chunk_id(chunk):
    """A stable id: the same file + chunk number + text always gives the same id."""
    key = f"{chunk.metadata['source']}|{chunk.metadata['chunk']}|{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def open_index(embeddings=None, index_dir=INDEX_DIR, collection=COLLECTION):
    """Open (or create) the Chroma collection in index_dir.

    We ask for cosine distance: 0 means the same meaning, larger means less related.
    """
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(index_dir),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def count_chunks(store):
    """How many chunks are stored in the index."""
    return len(store.get(include=[])["ids"])


def build_index(chunks, embeddings=None, index_dir=INDEX_DIR):
    """Store the chunks. Only NEW chunks are embedded; chunks of changed files are replaced.

    Returns (store, report) where report = {"added": ..., "removed": ..., "total": ...}.
    """
    store = open_index(embeddings, index_dir)
    wanted = {chunk_id(chunk): chunk for chunk in chunks}
    existing = set(store.get(include=[])["ids"])

    old_ids = sorted(existing - set(wanted))              # chunks that are no longer in the files
    new_ids = [i for i in wanted if i not in existing]     # chunks we have not stored yet
    if old_ids:
        store.delete(ids=old_ids)
    if new_ids:
        store.add_documents([wanted[i] for i in new_ids], ids=new_ids)

    report = {"added": len(new_ids), "removed": len(old_ids), "total": count_chunks(store)}
    return store, report


def load_index(embeddings=None, index_dir=INDEX_DIR):
    """Open the saved index for searching. Stops with a clear message if it is empty."""
    store = open_index(embeddings, index_dir)
    if count_chunks(store) == 0:
        raise IndexNotBuiltError("The index is empty. Build it first with:  python ingest.py")
    return store


# ---------------------------------------------------------------------------
# Run the whole indexing pipeline and show what happens at each step
# ---------------------------------------------------------------------------

def show_one_overlap(chunks):
    """Print one pair of neighbouring chunks with their shared text marked « »."""
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata["source"] == b.metadata["source"]]
    sharing = [(a, b, shared_text(a.page_content, b.page_content)) for a, b in pairs]
    sharing = [(a, b, s) for a, b, s in sharing if len(s) >= 20]
    headings = [s for _, _, s in sharing if s.startswith("#")]
    print(f"Neighbouring chunks that share text: {len(sharing)} of {len(pairs)} pairs "
          f"({len(headings)} of them share just a heading line)")
    sentences = [(a, b, s) for a, b, s in sharing if not s.startswith("#")]
    if not sentences:
        return
    a, b, shared = max(sentences, key=lambda item: len(item[2]))  # the longest shared sentence
    before = a.page_content[:-len(shared)][-70:].replace("\n", " ")
    after = b.page_content[len(shared):][:70].replace("\n", " ")
    flat = shared.replace("\n", " ")
    print(f"  {source_tag(a)} ends with:    ...{before}«{flat}»")
    print(f"  {source_tag(b)} starts with:  «{flat}»{after}...")
    print(f"  Shared text: {len(shared)} characters")


def main():
    # Step 3 - load
    docs = load_handbook()
    files = sorted({doc.metadata["source"] for doc in docs})
    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"Loaded {len(files)} files as {len(docs)} documents ({total_chars:,} characters)")
    for name in files:
        parts = [doc for doc in docs if doc.metadata["source"] == name]
        chars = sum(len(doc.page_content) for doc in parts)
        print(f"  {name:<16} {len(parts)} document(s)  {chars:>5,} characters")

    # Step 4 - split
    chunks = split_documents(docs)
    print(f"\nSplit into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})")
    print(f"  Example chunk metadata: {chunks[2].metadata}")
    show_one_overlap(chunks)

    # Step 6 - embed + store
    print("\nEmbedding new chunks and saving them to index/ ...")
    _, report = build_index(chunks)
    print(f"Indexed {report['total']} chunks "
          f"(new this run: {report['added']}, removed: {report['removed']})")


if __name__ == "__main__":
    main()
