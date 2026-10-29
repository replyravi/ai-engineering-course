"""
cafe_index.py - Step 14: put the loaded Documents into a search index, so the bot can look facts up.

    python cafe_index.py                                    build the index in index/ (no API key needed)
    python cafe_index.py --search "Is the kachori nut-free?"   the 3 closest chunks, with their citation tags

This is Episode 08's recipe, copied here so this folder runs on its own:
    - the same splitter: RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
      (ep08-chunk-lab/chunklab/strategies.py, the "recursive" strategy)
    - the same store: a Chroma collection on disk with cosine distance, one id per chunk,
      emptied and rebuilt on every run (ep08-chunk-lab/chunklab/index.py)
    - the same free local embeddings model from llm.get_embeddings()
New in this episode: the chunks come from three loaders, and their metadata becomes the citation tag.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")    # no usage pings from Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from common import DATA_DIR, INDEX_DIR  # noqa: E402
from loaders import load_folder, short_path  # noqa: E402

CHUNK_SIZE = 400              # characters, Episode 08's setting
CHUNK_OVERLAP = 60
COLLECTION = "cafe"
SPACE = {"hnsw": {"space": "cosine"}}      # cosine distance: 0 = same meaning, bigger = less related
DEFAULT_QUESTION = "Is the kachori nut-free?"


class IndexMissingError(RuntimeError):
    """Raised when someone searches before building the index."""


def citation(doc):
    """The tag a human can check: 'menu.csv row 7', 'policy.pdf page 1' or 'about.html'."""
    meta = doc.metadata
    if "row" in meta:                       # CSVLoader: row counts from 0, like the loader does
        return f"{meta['file']} row {meta['row']}"
    if "page_label" in meta:                # PyPDFLoader: the page number printed on the page
        return f"{meta['file']} page {meta['page_label']}"
    return meta["file"]


def split(docs):
    """Cut long Documents into chunks of at most 400 characters. Every chunk keeps its Document's metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              add_start_index=True)
    chunks = splitter.split_documents(docs)
    counts = {}
    for chunk in chunks:                    # number the chunks of each tag: about.html #1, #2, ...
        tag = citation(chunk)
        counts[tag] = counts.get(tag, 0) + 1
        chunk.metadata["chunk"] = counts[tag]
    return chunks


def chunk_id(chunk):
    """One id per chunk, for example 'policy.pdf page 1#2'."""
    return f"{citation(chunk)}#{chunk.metadata['chunk']}"


def open_index(embeddings=None, folder=INDEX_DIR):
    """Open (or create) the Chroma collection saved in index/."""
    from langchain_chroma import Chroma

    from llm import get_embeddings

    return Chroma(collection_name=COLLECTION, embedding_function=embeddings or get_embeddings(),
                  persist_directory=str(folder), collection_configuration=SPACE)


def build_index(chunks, embeddings=None, folder=INDEX_DIR):
    """Empty the collection, then embed and store every chunk."""
    store = open_index(embeddings, folder)
    store.reset_collection()                # start empty: no old chunks, no duplicates
    store.add_documents(chunks, ids=[chunk_id(chunk) for chunk in chunks])
    return store


def load_index(embeddings=None, folder=INDEX_DIR):
    """Open the saved index for searching. Stops with a clear message if it was never built."""
    if not (folder / "chroma.sqlite3").exists():
        raise IndexMissingError("The cafe index is not built yet. Build it with:  python cafe_index.py")
    return open_index(embeddings, folder)


def search(store, question, k=3):
    """The k closest chunks as (Document, distance), closest first."""
    return store.similarity_search_with_score(question, k=k)


def preview(text, width=90):
    """The first `width` characters of a chunk, on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def build():
    from llm import DEFAULT_LOCAL_EMBEDDINGS_MODEL, get_embeddings

    docs, report = load_folder(DATA_DIR)
    counts = ", ".join(f"{name} {row['documents']}" for name, row in report.rows.items())
    print(f"Loaded {len(docs)} Documents from {short_path(DATA_DIR)}/ ({counts})")
    chunks = split(docs)
    print(f"Split with RecursiveCharacterTextSplitter(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}) "
          f"-> {len(chunks)} chunks")
    for name in report.rows:
        mine = [chunk for chunk in chunks if chunk.metadata["file"] == name]
        longest = max(len(chunk.page_content) for chunk in mine)
        print(f"  {name:<12} {report.rows[name]['documents']:>2} Documents -> {len(mine):>2} chunks "
              f"(longest {longest} characters)")
    embeddings = get_embeddings()
    model = os.getenv("EMBEDDINGS_MODEL") or DEFAULT_LOCAL_EMBEDDINGS_MODEL
    print(f"Embedding model: {model} ({len(embeddings.embed_query('test'))} numbers per chunk)")
    build_index(chunks, embeddings)
    print(f"Saved {len(chunks)} chunks to {short_path(INDEX_DIR)}/ (Chroma collection '{COLLECTION}', cosine distance)")
    kachori = next(chunk for chunk in chunks if chunk_id(chunk) == "menu.csv row 7#1")
    print(f"Example: id '{chunk_id(kachori)}', metadata {kachori.metadata}")


def show_search(question, k):
    store = load_index()
    print(f"Question: {question}\n")
    for rank, (doc, distance) in enumerate(search(store, question, k), start=1):
        print(f"{rank}. distance {distance:.3f}  [{citation(doc)}]  {preview(doc.page_content)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build or search the cafe index.")
    parser.add_argument("--search", nargs="?", const=DEFAULT_QUESTION, metavar="QUESTION",
                        help=f'search instead of building (default question: "{DEFAULT_QUESTION}")')
    parser.add_argument("-k", type=int, default=3, help="how many chunks to show (default 3)")
    args = parser.parse_args(argv)
    try:
        if args.search:
            show_search(args.search, args.k)
        else:
            build()
    except IndexMissingError as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
