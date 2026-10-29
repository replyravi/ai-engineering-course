"""
chunklab/index.py - Steps 8-9: one Chroma collection per strategy, saved on disk, then reopened.

    python -m chunklab index            Step 8: split, embed and store all 4 strategies (rebuilds them)
    python -m chunklab index --reopen   Step 9: open the saved collections in a NEW process, with no embedding

The collections live in collections/ (git-ignored). Building them again replaces the old ones,
because in a lab we change the settings often. (Episode 07 showed how stable ids update an index
without rebuilding it.)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import os
import sys
import time

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")    # no usage pings from Chroma

from chunklab.strategies import (  # noqa: E402
    HERE, LABELS, STRATEGIES, hide_size_warnings, load_handbook, make_chunks, tag,
)

COLLECTIONS_DIR = HERE / "collections"
SPACE = {"hnsw": {"space": "cosine"}}      # cosine distance: 0 = same meaning, bigger = less related


class CollectionsMissingError(RuntimeError):
    """Raised when someone searches before building the collections."""


def open_collection(strategy, embeddings=None, folder=COLLECTIONS_DIR):
    """Open (or create) the Chroma collection of one strategy. Its name is the strategy name."""
    from langchain_chroma import Chroma

    from llm import get_embeddings

    return Chroma(
        collection_name=strategy,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(folder),
        collection_configuration=SPACE,
    )


def build_collection(strategy, chunks, embeddings, folder=COLLECTIONS_DIR):
    """Empty the collection, then embed and add every chunk. The id of a chunk is its tag, e.g. refunds.md#4."""
    store = open_collection(strategy, embeddings, folder)
    store.reset_collection()                       # start empty: no old chunks, no duplicates
    store.add_documents(chunks, ids=[tag(chunk) for chunk in chunks])
    return store


def saved_counts(folder=COLLECTIONS_DIR):
    """{strategy: number of chunks} read straight from disk with the chromadb client (no embedding model)."""
    import chromadb

    if not (folder / "chroma.sqlite3").exists():
        raise CollectionsMissingError("No collections yet. Build them first with:  python -m chunklab index")
    client = chromadb.PersistentClient(path=str(folder))
    names = {collection.name for collection in client.list_collections()}
    missing = [strategy for strategy in STRATEGIES if strategy not in names]
    if missing:
        raise CollectionsMissingError(f"Missing collections {missing}. Rebuild with:  python -m chunklab index")
    return {strategy: client.get_collection(strategy).count() for strategy in STRATEGIES}


def index_settings(strategy, folder=COLLECTIONS_DIR):
    """The HNSW settings Chroma saved for a collection (space, ef_search, max_neighbors, ...)."""
    import chromadb

    return chromadb.PersistentClient(path=str(folder)).get_collection(strategy).configuration["hnsw"]


def folder_size(folder):
    """Bytes used by every file inside `folder`."""
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def load_store(strategy, embeddings=None, folder=COLLECTIONS_DIR):
    """Open a saved collection for searching. Stops with a clear message if it was never built."""
    saved_counts(folder)
    return open_collection(strategy, embeddings, folder)


def build_all(folder=COLLECTIONS_DIR):
    from llm import DEFAULT_LOCAL_EMBEDDINGS_MODEL, get_embeddings

    embeddings = get_embeddings()
    model = os.getenv("EMBEDDINGS_MODEL") or DEFAULT_LOCAL_EMBEDDINGS_MODEL
    print(f"Embedding model: {model} ({len(embeddings.embed_query('test'))} numbers per chunk)")
    print(f"Rebuilding {len(STRATEGIES)} collections in {folder.name}/ (one per strategy) ...")
    hide_size_warnings()
    docs = load_handbook()
    total = 0
    for strategy in STRATEGIES:
        chunks = make_chunks(strategy, docs)
        start = time.perf_counter()
        build_collection(strategy, chunks, embeddings, folder)
        seconds = time.perf_counter() - start
        total += len(chunks)
        print(f"  {LABELS[strategy]:<20} {len(chunks):>3} chunks   embedded and stored in {seconds:.2f} s")
    print(f"Saved {total} chunks. {folder.name}/ is {folder_size(folder) / 1e6:.1f} MB on disk.")

    config = index_settings("headers", folder)
    print(f"Index type: HNSW (approximate nearest neighbours), distance: {config['space']}, "
          f"ef_search={config['ef_search']}, max_neighbors={config['max_neighbors']}")

    # One row of the 'headers' collection, exactly as Chroma stores it.
    headers = open_collection("headers", embeddings, folder)
    row = headers.get(ids=["refunds.md#3"], include=["embeddings", "documents", "metadatas"])
    vector = row["embeddings"][0]
    print("\nOne row of the 'headers' collection:")
    print(f"  id:        {row['ids'][0]}")
    print(f"  vector:    {len(vector)} numbers, starting {[round(float(x), 4) for x in vector[:4]]}")
    print(f"  text:      {row['documents'][0][:70].replace(chr(10), ' ')}...")
    metadata = row["metadatas"][0]
    order = ["source", "h1", "h2", "part", "chunk"]                 # Chroma returns the keys in any order
    ordered = {key: metadata[key] for key in order if key in metadata}
    print(f"  metadata:  {ordered}")


def reopen(folder=COLLECTIONS_DIR):
    print(f"Opening {folder.name}/ in this new process, with no embedding model loaded ...")
    start = time.perf_counter()
    counts = saved_counts(folder)
    seconds = time.perf_counter() - start
    for strategy, count in counts.items():
        print(f"  {LABELS[strategy]:<20} {count:>3} chunks")
    loaded = "sentence_transformers" in sys.modules
    print(f"Opened and counted in {seconds:.2f} s. Embedding model loaded: {'yes' if loaded else 'no'}. "
          "Texts embedded: 0 (the vectors were read from disk).")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chunklab index", description="Build or reopen the collections.")
    parser.add_argument("--reopen", action="store_true", help="only open the saved collections and count them")
    args = parser.parse_args(argv)
    try:
        if args.reopen:
            reopen()
        else:
            build_all()
    except CollectionsMissingError as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
