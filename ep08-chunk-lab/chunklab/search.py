"""
chunklab/search.py - Step 13: search one collection, optionally only inside some files (a metadata filter).

    python -m chunklab search "What should I do if my food smells strange?"
    python -m chunklab search "What should I do if my food smells strange?" --file food_safety.md
    python -m chunklab search "When can I visit?" --file food_safety.md --file support.md -k 5
    python -m chunklab search "pause rules" --strategy recursive

The filter narrows the rows FIRST: only chunks whose metadata matches are compared with the question.
The score is a cosine DISTANCE: lower means closer in meaning.
No API key needed; build the collections first with:  python -m chunklab index

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse

from chunklab.index import CollectionsMissingError, load_store
from chunklab.strategies import STRATEGIES, breadcrumb, tag

DEFAULT_QUESTION = "What should I do if my food smells strange?"


def file_filter(files):
    """Chroma's filter for one file, or for several files with $in. None means no filter."""
    if not files:
        return None
    if len(files) == 1:
        return {"source": files[0]}
    return {"source": {"$in": files}}


def search(store, question, k=3, files=None):
    """The k closest chunks as (Document, distance), closest first, only from `files` if given."""
    return store.similarity_search_with_score(question, k=k, filter=file_filter(files))


def preview(text, width=100):
    """The first `width` characters of a chunk, on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chunklab search", description="Search one collection.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--file", action="append", help="only search this handbook file (repeat for more)")
    parser.add_argument("--strategy", default="headers", choices=list(STRATEGIES), help="which collection")
    parser.add_argument("-k", type=int, default=3, help="how many chunks to return (default 3)")
    args = parser.parse_args(argv)

    try:
        store = load_store(args.strategy)
    except CollectionsMissingError as error:
        print(error)
        return 1

    total = len(store.get(include=[])["ids"])
    allowed = len(store.get(where=file_filter(args.file), include=[])["ids"]) if args.file else total
    print(f"Question: {args.question}")
    print(f"Collection: {args.strategy} ({total} chunks)   filter: {file_filter(args.file)}   "
          f"chunks that pass the filter: {allowed}\n")
    if allowed == 0:
        names = sorted({meta["source"] for meta in store.get(include=["metadatas"])["metadatas"]})
        print(f"No chunk passed the filter. File names are exact, for example: {', '.join(names)}")
        return 1
    for rank, (doc, distance) in enumerate(search(store, args.question, args.k, args.file), start=1):
        where = breadcrumb(doc) or "no breadcrumb (this splitter keeps no headings)"
        print(f"{rank}. distance {distance:.3f}  [{tag(doc)}]  {where}")
        print(f"   {preview(doc.page_content)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
