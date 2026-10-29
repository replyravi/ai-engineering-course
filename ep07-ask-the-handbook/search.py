"""
search.py - Step 7: find the handbook chunks closest in MEANING to a question.

    python search.py "What is the refund window for a missed lunch?"
    python search.py "Is the kitchen nut-free?" -k 5

No chat model and no API key here: this is only the embedding model + Chroma.
The score is a cosine DISTANCE: lower means closer in meaning (0 = same meaning).

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse

from ingest import IndexNotBuiltError, load_index, source_tag, source_where


def search(question, k=3, store=None):
    """Return the k closest chunks as (Document, distance) pairs, closest first."""
    if store is None:
        store = load_index()
    return store.similarity_search_with_score(question, k=k)


def preview(text, width=110):
    """The first `width` characters of a chunk, on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def main():
    parser = argparse.ArgumentParser(description="Search the Dabbaloop handbook index.")
    parser.add_argument("question", nargs="?", default="What is the refund window for a missed lunch?")
    parser.add_argument("-k", type=int, default=3, help="how many chunks to return (default 3)")
    args = parser.parse_args()

    try:
        results = search(args.question, k=args.k)
    except IndexNotBuiltError as error:
        print(error)
        raise SystemExit(1)

    print(f"Question: {args.question}\n")
    for rank, (doc, distance) in enumerate(results, start=1):
        print(f"{rank}. distance {distance:.3f}  [{source_tag(doc)}]  {source_where(doc)}")
        print(f"   {preview(doc.page_content)}")


if __name__ == "__main__":
    main()
