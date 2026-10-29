"""
eval_retrieval.py - Step 14: measure retrieval with a golden question set (hit rate at k).

    python eval_retrieval.py                      # k=3, chunk_size=400, chunk_overlap=60
    python eval_retrieval.py --chunk-size 120     # try tiny chunks and watch the hit rate
    python eval_retrieval.py --k 1 2 3 4 5 6      # compare several k values in one run

A question is a HIT when one of the top-k chunks comes from an expected file AND
contains the expected answer phrase (both are written in golden.json).
No chat model and no key needed. The script builds its own throwaway index in memory,
so trying other settings never touches your real index/ folder.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma

from ingest import CHUNK_OVERLAP, CHUNK_SIZE, load_handbook, source_tag, split_documents
from llm import get_embeddings
from rag_chain import MAX_DISTANCE, estimate_tokens

GOLDEN_FILE = Path(__file__).resolve().parent / "golden.json"


def load_golden(path=GOLDEN_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_test_index(chunk_size, chunk_overlap, embeddings):
    """Split the handbook with these settings and index it in memory (nothing saved)."""
    chunks = split_documents(load_handbook(), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = Chroma(
        collection_name=f"eval_{chunk_size}_{chunk_overlap}",
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    store.add_documents(chunks, ids=[f"c{i}" for i in range(len(chunks))])
    return store, len(chunks)


def is_hit(docs, case):
    """True when a retrieved chunk is from an expected file and contains the answer phrase."""
    phrase = case["answer_contains"].lower()
    return any(
        doc.metadata["source"] in case["sources"] and phrase in doc.page_content.lower()
        for doc in docs
    )


def evaluate(store, golden, k):
    """Return (rows, hits, average_context_tokens) for the in-scope questions."""
    rows, hits, context_tokens = [], 0, 0
    for case in golden["in_scope"]:
        found = store.similarity_search_with_score(case["question"], k=k)
        docs = [doc for doc, _ in found]
        hit = is_hit(docs, case)
        hits += hit
        context_tokens += estimate_tokens("".join(doc.page_content for doc in docs))
        rows.append({"hit": hit, "question": case["question"],
                     "top": source_tag(docs[0]), "distance": found[0][1]})
    return rows, hits, round(context_tokens / len(golden["in_scope"]))


def main():
    parser = argparse.ArgumentParser(description="Hit rate at k on the golden questions.")
    parser.add_argument("--k", type=int, nargs="+", default=[3], help="one or more k values (default 3)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    golden = load_golden()
    total = len(golden["in_scope"])
    store, n_chunks = build_test_index(args.chunk_size, args.chunk_overlap, get_embeddings())
    print(f"Index: {n_chunks} chunks (chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap})\n")

    if len(args.k) > 1:  # a short table, one line per k
        print("  k   hit rate      context sent to the model")
        for k in args.k:
            _, hits, tokens = evaluate(store, golden, k)
            print(f"  {k}   {hits}/{total} = {hits / total:>4.0%}   about {tokens} tokens per question")
        return

    k = args.k[0]
    rows, hits, tokens = evaluate(store, golden, k)
    for row in rows:
        mark = "HIT " if row["hit"] else "MISS"
        print(f"{mark}  best {row['distance']:.3f} [{row['top']:<18}] {row['question']}")
    print(f"\nHit rate at k={k}: {hits}/{total} = {hits / total:.0%}   (about {tokens} context tokens per question)")

    print(f"\nOut-of-scope questions (gate: refuse without the model when best distance > {MAX_DISTANCE}):")
    for question in golden["out_of_scope"]:
        best = store.similarity_search_with_score(question, k=1)[0][1]
        verdict = "refused by the gate" if best > MAX_DISTANCE else "sent to the model, the prompt rule must refuse"
        print(f"  best {best:.3f}  {question}  -> {verdict}")


if __name__ == "__main__":
    main()
