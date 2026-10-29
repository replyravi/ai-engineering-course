"""
chunklab/faiss_compare.py - Step 14: FAISS (a library) next to Chroma (an embedded database).

    python -m chunklab faiss                  same chunks and vectors in both: same top 3? times, size on disk
    python -m chunklab faiss --scale 100000   extra: exact search versus HNSW on 100,000 made-up vectors

Both stores get the 'headers' chunks and the SAME MiniLM vectors, embedded once, so we time the stores only.
FAISS here uses IndexFlatL2: EXACT search, the question is compared with every row.
Chroma uses HNSW: an APPROXIMATE index that hops through a graph of neighbours.

Safety: FAISS.load_local(..., allow_dangerous_deserialization=True) unpickles index.pkl, and a pickle
file can run code when it is loaded. Only load index files that you made yourself.

FAISS and PyTorch each bring their own copy of the OpenMP library (it runs work on several CPU cores).
Using both in one process can stop Python with "OMP: Error #15" (seen on macOS), so we allow two copies
with KMP_DUPLICATE_LIB_OK, the same setting scikit-learn makes when sentence-transformers imports it.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")    # see the OpenMP note above

from chunklab.index import folder_size  # noqa: E402
from chunklab.strategies import HERE, hide_size_warnings, load_questions, make_chunks, tag  # noqa: E402

FAISS_DIR = HERE / "faiss_index"
REPEATS = 20                         # each query runs this many times; we report the median
SCALE_SEED = 8                       # made-up vectors for --scale are the same on every run
SCALE_QUERIES = 100


def median_ms(function, repeats=REPEATS):
    """Run function() `repeats` times and return the median time in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def build_chroma(folder, ids, texts, metadatas, vectors):
    """A Chroma collection on disk, filled with ready-made vectors (the chromadb client, no LangChain)."""
    import chromadb

    client = chromadb.PersistentClient(path=str(folder))
    collection = client.create_collection("headers", configuration={"hnsw": {"space": "cosine"}},
                                          embedding_function=None)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
    return collection


def build_faiss(texts, metadatas, ids, vectors, embeddings):
    """A FAISS index in memory, filled with the same ready-made vectors."""
    from langchain_community.vectorstores import FAISS   # langchain-community is archived; FAISS still works

    return FAISS.from_embeddings(list(zip(texts, vectors)), embeddings, metadatas=metadatas, ids=ids)


def compare():
    import chromadb  # noqa: F401  (load both libraries BEFORE the timers: importing is not building)
    import faiss  # noqa: F401
    from langchain_community.vectorstores import FAISS

    from llm import get_embeddings

    hide_size_warnings()
    chunks = make_chunks("headers")
    ids = [tag(chunk) for chunk in chunks]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    questions = [question["question"] for question in load_questions()]

    embeddings = get_embeddings()
    start = time.perf_counter()
    vectors = embeddings.embed_documents(texts)
    query_vectors = [embeddings.embed_query(question) for question in questions]
    print(f"Same {len(chunks)} 'headers' chunks for both stores. Embedded once with MiniLM in "
          f"{time.perf_counter() - start:.2f} s (that cost is the same for every store).\n")

    # ignore_cleanup_errors: on Windows, Chroma can still hold its files open when the folder is deleted.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        start = time.perf_counter()
        collection = build_chroma(Path(folder), ids, texts, metadatas, vectors)
        chroma_build = time.perf_counter() - start
        chroma_size = folder_size(Path(folder))
        chroma_top = [collection.query(query_embeddings=[q], n_results=3, include=["distances"]) for q in query_vectors]
        chroma_ms = statistics.median(median_ms(lambda q=q: collection.query(query_embeddings=[q], n_results=3))
                                      for q in query_vectors)

    start = time.perf_counter()
    faiss_store = build_faiss(texts, metadatas, ids, vectors, embeddings)
    faiss_build = time.perf_counter() - start
    faiss_top = [faiss_store.similarity_search_with_score_by_vector(q, k=3) for q in query_vectors]
    faiss_ms = statistics.median(median_ms(lambda q=q: faiss_store.similarity_search_with_score_by_vector(q, k=3))
                                 for q in query_vectors)
    faiss_store.save_local(str(FAISS_DIR))
    faiss_size = folder_size(FAISS_DIR)

    print(f"{'':<34}{'build':>10}{'query (median)':>16}{'on disk':>12}")
    print(f"{'Chroma  (HNSW, cosine distance)':<34}{chroma_build * 1000:>8.1f}ms{chroma_ms:>14.3f}ms"
          f"{chroma_size / 1e3:>9.0f} KB")
    print(f"{'FAISS   (IndexFlatL2, exact)':<34}{faiss_build * 1000:>8.1f}ms{faiss_ms:>14.3f}ms"
          f"{faiss_size / 1e3:>9.0f} KB")
    print("(Chroma's folder holds a SQLite database plus its index files; FAISS saves index.faiss + index.pkl.)\n")

    same = 0
    for chroma_result, faiss_result in zip(chroma_top, faiss_top):
        same += chroma_result["ids"][0] == [doc.id for doc, _ in faiss_result]
    print(f"Same top 3, in the same order, for {same} of {len(questions)} questions.")

    print(f"\nExample: {questions[0]}")
    for rank, (doc, score) in enumerate(faiss_top[0]):
        chroma_distance = chroma_top[0]["distances"][0][rank]
        print(f"  {rank + 1}. Chroma {chroma_top[0]['ids'][0][rank]:<17} cosine distance {chroma_distance:.3f}   "
              f"FAISS {doc.id:<17} squared L2 {score:.3f}")
    print("For unit-length vectors like MiniLM's, squared L2 = 2 x cosine distance, so the order is the same.")

    loaded = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    print(f"\nSaved FAISS to {FAISS_DIR.name}/ and loaded it back: {loaded.index.ntotal} vectors. "
          "(allow_dangerous_deserialization=True: index.pkl is a pickle, so only load files you made yourself.)")


def scale(n):
    """Exact (flat) search versus HNSW on n made-up, clustered vectors of 384 numbers."""
    import faiss
    import numpy as np

    faiss.omp_set_num_threads(1)           # one CPU core, so we compare the methods, not your cores
    rng = np.random.default_rng(SCALE_SEED)
    dim = 384
    centres = rng.normal(size=(2000, dim)).astype("float32")             # 2,000 made-up "topics"
    data = centres[rng.integers(0, 2000, n)] + 0.25 * rng.normal(size=(n, dim)).astype("float32")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    queries = data[rng.integers(0, n, SCALE_QUERIES)] + 0.02 * rng.normal(size=(SCALE_QUERIES, dim)).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    print(f"{n:,} made-up vectors of {dim} numbers (seed {SCALE_SEED}), {SCALE_QUERIES} questions, 1 CPU core\n")

    start = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    flat.add(data)
    flat_build = time.perf_counter() - start
    start = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)       # 32 neighbours per node
    hnsw.add(data)
    hnsw_build = time.perf_counter() - start
    hnsw.hnsw.efSearch = 64                                                # how wide each search looks

    start = time.perf_counter()
    _, exact = flat.search(queries, 3)
    flat_ms = (time.perf_counter() - start) * 1000 / SCALE_QUERIES
    faiss.cvar.hnsw_stats.reset()
    start = time.perf_counter()
    _, approx = hnsw.search(queries, 3)
    hnsw_ms = (time.perf_counter() - start) * 1000 / SCALE_QUERIES
    hops = faiss.cvar.hnsw_stats.ndis / SCALE_QUERIES
    same = sum(set(a) == set(b) for a, b in zip(exact, approx))

    print(f"{'':<26}{'build':>9}{'query':>11}{'comparisons per query':>24}")
    print(f"{'exact (IndexFlatIP)':<26}{flat_build:>8.2f}s{flat_ms:>9.3f}ms{n:>24,}")
    print(f"{'HNSW (IndexHNSWFlat)':<26}{hnsw_build:>8.2f}s{hnsw_ms:>9.3f}ms{round(hops):>24,}")
    print(f"\nHNSW found the same top 3 as exact search for {same} of {SCALE_QUERIES} questions: "
          "faster, but approximate. (Real embeddings behave differently from made-up ones.)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chunklab faiss", description="FAISS next to Chroma.")
    parser.add_argument("--scale", type=int, nargs="?", const=100_000, default=None,
                        help="exact versus HNSW on this many made-up vectors (default 100000)")
    args = parser.parse_args(argv)
    if args.scale:
        scale(args.scale)
    else:
        compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
