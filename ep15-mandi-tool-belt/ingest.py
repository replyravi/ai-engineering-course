"""
ingest.py - Step 12: turn the crawled pages into a search index (Episode 07's pipeline, reused).

    python ingest.py                       reads crawl.json if you crawled (Step 11), else fixtures/crawl.json
    python ingest.py --source fixtures/crawl.json

Same steps as Episode 07's ingest.py, now for web pages instead of handbook files:
    load   every page becomes a Document, labelled with its URL
    split  400-character chunks with 60 characters of overlap (and we SHOW the count)
    embed  384 numbers per chunk with the free local all-MiniLM-L6-v2 model (no key)
    store  a Chroma index in index/, with stable ids, so running it twice adds nothing twice

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import get_embeddings

HERE = Path(__file__).resolve().parent
CRAWL_FILE = HERE / "crawl.json"                       # made by crawl.py (Step 11, optional)
FIXTURE = HERE / "fixtures" / "crawl.json"             # the offline copy that ships with the repo
INDEX_DIR = HERE / "index"
COLLECTION = "mandi_rule_book"

CHUNK_SIZE = 400     # characters per chunk (at most)
CHUNK_OVERLAP = 60   # characters that neighbouring chunks may share


class IndexNotBuiltError(RuntimeError):
    """Raised when someone searches before running ingest.py."""


def default_source() -> Path:
    """Your own crawl if you made one, else the fixture."""
    return CRAWL_FILE if CRAWL_FILE.exists() else FIXTURE


# ---------------------------------------------------------------------------
# Load and split
# ---------------------------------------------------------------------------

def load_pages(path) -> tuple[list[Document], dict]:
    """Read a crawl file. Every page becomes one Document with its URL and title as metadata."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = [Document(page_content=page["text"], metadata={"source": page["url"], "title": page["title"]})
            for page in data["pages"]]
    return docs, data


def page_name(url: str) -> str:
    """A short name for a URL: '.../site/fees.html' -> 'fees.html', '.../site/' -> 'site/'."""
    parts = url.rstrip("/").split("/")
    return parts[-1] if not url.endswith("/") else parts[-1] + "/"


def split_pages(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> list[Document]:
    """Cut pages into chunks. Each chunk keeps its URL and title and gets a chunk number per page."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        for number, chunk in enumerate(splitter.split_documents([doc]), start=1):
            chunk.metadata["chunk"] = number
            chunks.append(chunk)
    return chunks


def chunk_tag(chunk) -> str:
    """The citation tag of a chunk: short page name + chunk number, like fees.html#2."""
    return f"{page_name(chunk.metadata['source'])}#{chunk.metadata['chunk']}"


# ---------------------------------------------------------------------------
# Embed + store (the same stable-id trick as Episode 07)
# ---------------------------------------------------------------------------

def chunk_id(chunk) -> str:
    """A stable id: the same URL + chunk number + text always gives the same id."""
    key = f"{chunk.metadata['source']}|{chunk.metadata['chunk']}|{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def open_index(embeddings=None, index_dir=None):
    """Open (or create) the Chroma collection, with cosine distance (lower = closer in meaning)."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(index_dir or INDEX_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def count_chunks(store) -> int:
    return len(store.get(include=[])["ids"])


def build_index(chunks, embeddings=None, index_dir=None):
    """Store the chunks. Only new chunks are embedded; chunks that left the crawl are removed."""
    store = open_index(embeddings, index_dir)
    wanted = {chunk_id(chunk): chunk for chunk in chunks}
    existing = set(store.get(include=[])["ids"])
    old_ids = sorted(existing - set(wanted))
    new_ids = [i for i in wanted if i not in existing]
    if old_ids:
        store.delete(ids=old_ids)
    if new_ids:
        store.add_documents([wanted[i] for i in new_ids], ids=new_ids)
    return store, {"added": len(new_ids), "removed": len(old_ids), "total": count_chunks(store)}


def load_index(embeddings=None, index_dir=None):
    """Open the saved index for searching. Stops with a clear message if it is empty."""
    store = open_index(embeddings, index_dir)
    if count_chunks(store) == 0:
        raise IndexNotBuiltError("The document index is empty. Build it first with:  python ingest.py")
    return store


def main(argv=None, embeddings=None) -> int:
    parser = argparse.ArgumentParser(description="Index the crawled pages for search_docs.")
    parser.add_argument("--source", type=Path, default=None, help="a crawl file (default: crawl.json, else the fixture)")
    args = parser.parse_args(argv)
    source = args.source or default_source()

    docs, data = load_pages(source)
    shown = source.relative_to(HERE) if source.is_relative_to(HERE) else source
    print(f"Read {shown.as_posix()}: {len(docs)} pages")
    print(f"  ({data.get('about', '')})")
    for doc in docs:
        print(f"  {doc.metadata['source']}  {len(doc.page_content):,} characters  \"{doc.metadata['title']}\"")

    chunks = split_pages(docs)
    print(f"\nSplit into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})")
    for doc in docs:
        mine = [c for c in chunks if c.metadata["source"] == doc.metadata["source"]]
        print(f"  {page_name(doc.metadata['source']):<12} {len(mine)} chunks")
    example = next(c for c in chunks if c.metadata["chunk"] == 2)
    print(f"  Example chunk {chunk_tag(example)}: {example.page_content[:90]!r}...")

    embeddings = embeddings or get_embeddings()
    dim = len(embeddings.embed_query("onion bag"))
    print("\nEmbedding new chunks and saving them to index/ ...")
    _, report = build_index(chunks, embeddings)
    print(f"{report['total']} chunks indexed, dim {dim} (new this run: {report['added']}, removed: {report['removed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
