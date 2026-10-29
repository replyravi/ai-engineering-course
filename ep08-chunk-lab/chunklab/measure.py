"""
chunklab/measure.py - Step 7: how many chunks, and how big, for each strategy?

    python -m chunklab measure

Every size is counted in MiniLM tokens, the units the embedder reads (it reads at most 254 of them).
The second table shows what overlap costs: the same text, stored more than once.
No API key and no internet needed.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse

from chunklab.strategies import (
    LABELS, STRATEGIES, count_tokens, hide_size_warnings, load_handbook, make_chunks, number_chunks,
    recursive_splitter, text_limit,
)

OVERLAP_SIZE = 150                  # a small chunk_size, so the cuts land inside paragraphs
OVERLAPS = [0, 20, 40, 60]          # characters


def quiet_long_text_warning():
    """Counting a whole file prints 'Token indices sequence length is longer than ...'. Step 4 explains it."""
    from transformers.utils import logging

    logging.set_verbosity_error()


def size_row(name, chunks, handbook_tokens, limit):
    """One table row: chunk count, average and largest size, how many are too long, stored tokens."""
    sizes = [count_tokens(chunk.page_content) for chunk in chunks]
    stored = sum(sizes)
    over = sum(size > limit for size in sizes)
    extra = (stored - handbook_tokens) / handbook_tokens
    return (f"{name:<22} {len(chunks):>6} {round(stored / len(chunks)):>9} {max(sizes):>9} {over:>10} "
            f"{stored:>15,} {extra:>+10.0%}")


def main(argv=None):
    argparse.ArgumentParser(prog="python -m chunklab measure", description=__doc__.split("\n")[1]).parse_args(argv)
    hide_size_warnings()
    quiet_long_text_warning()
    docs = load_handbook()
    limit = text_limit()
    handbook_tokens = sum(count_tokens(doc.page_content) for doc in docs)

    print(f"Chunk sizes in MiniLM tokens (MiniLM reads at most {limit} tokens of text per chunk)\n")
    print(f"{'strategy':<22} {'chunks':>6} {'average':>9} {'largest':>9} {'over ' + str(limit):>10} "
          f"{'stored tokens':>15} {'vs handbook':>10}")
    for strategy in STRATEGIES:
        print(size_row(LABELS[strategy], make_chunks(strategy, docs), handbook_tokens, limit))
    print(size_row("(no splitting)", docs, handbook_tokens, limit))
    print(f"\nThe handbook itself: {len(docs)} files, {handbook_tokens:,} MiniLM tokens. "
          "'vs handbook' = extra tokens stored because of overlap and cut words.")

    print(f"\nWhat overlap costs (recursive splitter, chunk_size={OVERLAP_SIZE} characters):\n")
    print(f"{'chunk_overlap':<22} {'chunks':>6} {'average':>9} {'largest':>9} {'over ' + str(limit):>10} "
          f"{'stored tokens':>15} {'vs handbook':>10}")
    for overlap in OVERLAPS:
        chunks = number_chunks(recursive_splitter(OVERLAP_SIZE, overlap).split_documents(docs))
        print(size_row(f"{overlap} characters", chunks, handbook_tokens, limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
