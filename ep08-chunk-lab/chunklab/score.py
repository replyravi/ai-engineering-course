"""
chunklab/score.py - Steps 10-12: stop guessing, measure. Which strategy finds the answers?

    python -m chunklab score --questions   Step 10: the ten questions, and a check that each answer is in its section
    python -m chunklab score               Step 11: the hit@3 scoreboard, from the collections of Step 8
    python -m chunklab score --fake        Step 12: the same scoring code with FAKE embeddings (no model, offline)
    python -m chunklab score --sweep       extra: the recursive splitter at five chunk sizes

A question is a HIT at k=3 when one of the 3 closest chunks comes from a right file AND contains
the answer phrase (both written in data/questions.json). No chat model and no API key needed.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import os
import sys

from rich.console import Console
from rich.text import Text

from chunklab.strategies import (
    LABELS, STRATEGIES, count_tokens, hide_size_warnings, load_handbook, load_questions, make_chunks,
    number_chunks, recursive_splitter, split_by_headers, tag,
)

K = 3
HOOK_QUESTION = 0           # Episode 07's miss: the refund question ...
HOOK_RULE = "Report a missed lunch or dinner within 6 hours"   # ... and the start of the main rule paragraph
SWEEP_SIZES = [100, 200, 400, 800, 1600]                         # characters; overlap = 15 percent

console = Console(highlight=False, soft_wrap=True)
HIT, MISS, BEST = "green", "magenta", "bold green"


# ---------------------------------------------------------------------------
# The scoring maths (tests/test_scoring.py checks these)
# ---------------------------------------------------------------------------

def holds_answer(doc, question):
    """True when the chunk comes from a right file and contains the answer phrase."""
    return (doc.metadata["source"] in question["sources"]
            and question["answer_contains"].lower() in doc.page_content.lower())


def rank_of_answer(docs, question):
    """1 for the closest chunk, 2 for the next, ...: the first chunk that holds the answer (None if none)."""
    for rank, doc in enumerate(docs, start=1):
        if holds_answer(doc, question):
            return rank
    return None


def rank_all(store, question_text):
    """Every chunk of the collection, closest first."""
    count = len(store.get(include=[])["ids"])
    return [doc for doc, _ in store.similarity_search_with_score(question_text, k=count)]


def score_store(store, questions, k=K, count=count_tokens):
    """One row per question: the answer's rank, hit or not, and the context tokens of the top k.

    count: how to measure the context (MiniLM tokens by default; tests may pass len).
    """
    rows = []
    for question in questions:
        ranked = rank_all(store, question["question"])
        rank = rank_of_answer(ranked, question)
        rows.append({
            "rank": rank,
            "hit": rank is not None and rank <= k,
            "first": rank == 1,
            "top": tag(ranked[0]),
            "context_tokens": sum(count(doc.page_content) for doc in ranked[:k]),
            "ranked": ranked,
        })
    return rows


def summarise(rows):
    """hit@k count, found-first count and the average context size."""
    return {
        "hits": sum(row["hit"] for row in rows),
        "first": sum(row["first"] for row in rows),
        "context": round(sum(row["context_tokens"] for row in rows) / len(rows)),
    }


def pick_winner(summaries):
    """Most hits at k; on a tie, most found first; then the least context sent to the model."""
    return max(summaries, key=lambda name: (summaries[name]["hits"], summaries[name]["first"],
                                            -summaries[name]["context"]))


def rule_rank(rows):
    """Where the main refund rule paragraph lands for the refund question (Episode 07's miss)."""
    for rank, doc in enumerate(rows[HOOK_QUESTION]["ranked"], start=1):
        if doc.metadata["source"] == "refunds.md" and HOOK_RULE in doc.page_content:
            return rank
    return None


# ---------------------------------------------------------------------------
# Building the collections to score
# ---------------------------------------------------------------------------

def in_memory_store(strategy, chunks, embeddings):
    """A throwaway Chroma collection in memory (nothing saved), for --fake and --sweep."""
    from langchain_chroma import Chroma

    store = Chroma(collection_name=f"try_{strategy}", embedding_function=embeddings,
                   collection_configuration={"hnsw": {"space": "cosine"}})
    store.reset_collection()
    store.add_documents(chunks, ids=[tag(chunk) for chunk in chunks])
    return store


def saved_stores():
    """The four collections from Step 8, opened with the real embedding model (for the questions)."""
    from chunklab.index import load_store, saved_counts
    from llm import get_embeddings

    saved_counts()                      # stops with a clear message if Step 8 was not run yet
    embeddings = get_embeddings()
    return {strategy: load_store(strategy, embeddings) for strategy in STRATEGIES}


def fake_stores():
    """The four strategies indexed with FAKE embeddings: made-up numbers, no model, no internet."""
    os.environ["HF_HUB_OFFLINE"] = "1"         # the MiniLM tokenizer must come from your disk, never the internet
    from testing.fakes import fake_embeddings

    docs = load_handbook()
    return {strategy: in_memory_store(strategy, make_chunks(strategy, docs), fake_embeddings())
            for strategy in STRATEGIES}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def say(*parts):
    line = Text()
    for part in parts:
        if isinstance(part, tuple):
            line.append(*part)
        else:
            line.append(part)
    console.print(line)


def cell(text, width, style=""):
    return (f"{text:>{width}}", style)


def print_scoreboard(results, questions, fake=False):
    """results = {strategy: rows}. Prints one line per question and the totals."""
    names = list(results)
    widths = {name: max(10, len(LABELS[name]) + 2) for name in names}
    say(f"Rank of the first chunk that holds the answer (MISS = not in the top {K}):")
    say(f"{'#':<3}{'question':<46}", *[cell(LABELS[name], widths[name], "bold") for name in names])
    for i, question in enumerate(questions):
        text = question["question"] if len(question["question"]) <= 44 else question["question"][:41] + "..."
        parts = [f"{i + 1:<3}{text:<46}"]
        for name in names:
            row = results[name][i]
            label = f"{row['rank']}" if row["hit"] else f"{row['rank'] or '-'} MISS"   # '-' = no chunk has it
            parts.append(cell(label, widths[name], HIT if row["hit"] else MISS))
        say(*parts)

    summaries = {name: summarise(rows) for name, rows in results.items()}
    winner = pick_winner(summaries)
    total = len(questions)
    say("")
    say(f"{'':<3}{f'hit@{K}':<46}", *[cell(f"{summaries[n]['hits']}/{total}", widths[n], BEST if n == winner else "")
                                      for n in names])
    say(f"{'':<3}{'found first (hit@1)':<46}", *[cell(f"{summaries[n]['first']}/{total}", widths[n]) for n in names])
    say(f"{'':<3}{f'context per question (top {K}, MiniLM tokens)':<46}",
        *[cell(summaries[n]["context"], widths[n]) for n in names])
    say(f"{'':<3}{'rank of the main refund rule (Episode 07)':<46}",
        *[cell(rule_rank(results[n]) or "-", widths[n]) for n in names])
    say("")
    if fake:
        say(("No winner: ", "bold yellow"), "fake vectors carry no meaning. Use the real model (Step 11) to compare.")
        return summaries, None
    say(("Winner: ", "bold"), (LABELS[winner], BEST),
        f" ({summaries[winner]['hits']}/{total} at k={K}, {summaries[winner]['first']}/{total} found first). "
        f"Tie rule: most hits at k={K}, then most found first, then the least context.")
    return summaries, winner


# ---------------------------------------------------------------------------
# Step 10 - check the ground truth before trusting any score
# ---------------------------------------------------------------------------

def check_questions(questions):
    """Every answer phrase must really be in the section we wrote down."""
    sections = {(s.metadata["source"], s.metadata.get("h2")): s.page_content for s in split_by_headers(load_handbook())}
    print(f"{len(questions)} questions in data/questions.json. "
          "Each one names the section that answers it and a phrase a chunk must contain:\n")
    good = 0
    for i, question in enumerate(questions, start=1):
        home = question["sources"][0]
        text = sections.get((home, question["section"]), "")
        found = question["answer_contains"].lower() in text.lower()
        good += found
        also = f"   (also accepted: {', '.join(question['sources'][1:])})" if len(question["sources"]) > 1 else ""
        print(f"{i:>2}  {question['question']}")
        print(f"    answer in {home} > {question['section']}{also}")
        print(f"    phrase \"{question['answer_contains']}\": {'found in that section' if found else 'NOT FOUND - fix the file'}")
    print(f"\n{good} of {len(questions)} answer phrases were found in their sections.")
    return good == len(questions)


# ---------------------------------------------------------------------------
# Extra - one strategy at several sizes (the precision versus context curve)
# ---------------------------------------------------------------------------

def sweep(questions):
    from llm import get_embeddings

    embeddings = get_embeddings()
    docs = load_handbook()
    print(f"Recursive splitter at {len(SWEEP_SIZES)} sizes (overlap = 15 percent), in-memory collections "
          "(your saved collections are not touched):\n")
    print(f"{'chunk_size':>10} {'chunks':>7} {f'hit@{K}':>7} {'found first':>12} {'context per question':>21}")
    for size in SWEEP_SIZES:
        chunks = number_chunks(recursive_splitter(size, round(size * 0.15)).split_documents(docs))
        summary = summarise(score_store(in_memory_store(f"sweep{size}", chunks, embeddings), questions))
        mark = "   <- the size we use" if size == 400 else ""
        print(f"{size:>10} {len(chunks):>7} {summary['hits']:>4}/{len(questions)} {summary['first']:>9}/{len(questions)} "
              f"{summary['context']:>14} tokens{mark}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chunklab score", description="The hit@3 scoreboard.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--questions", action="store_true", help="list the questions and check the answers exist")
    group.add_argument("--fake", action="store_true", help="fake embeddings: tests the scoring code offline")
    group.add_argument("--sweep", action="store_true", help="recursive splitter at several chunk sizes")
    args = parser.parse_args(argv)

    hide_size_warnings()
    questions = load_questions()
    if args.questions:
        return 0 if check_questions(questions) else 1
    if args.sweep:
        sweep(questions)
        return 0

    if args.fake:
        say(("FAKE embeddings", "bold yellow"), ": every text gets made-up numbers, so these scores are pure luck.")
        say("This run only proves the scoring code works: no embedding model, no internet, no API key.\n")
        stores = fake_stores()
    else:
        from chunklab.index import CollectionsMissingError

        try:
            stores = saved_stores()
        except CollectionsMissingError as error:
            print(error)
            return 1
        counts = ", ".join(f"{name} {len(store.get(include=[])['ids'])}" for name, store in stores.items())
        say(f"Scoring the 4 saved collections ({counts} chunks) with {len(questions)} questions, k={K}.\n")

    results = {strategy: score_store(store, questions) for strategy, store in stores.items()}
    print_scoreboard(results, questions, fake=args.fake)
    if args.fake:
        say(f"\nEmbedding model loaded: {'yes' if 'sentence_transformers' in sys.modules else 'no'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
