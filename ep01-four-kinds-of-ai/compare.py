"""
compare.py - one scoreboard for the four kinds of AI (step 15).

All four approaches are scored on the SAME 10 test rows that nobody trained on.

    python compare.py                                        the scoreboard
    python compare.py "Great job, the dosa was cold again."  four answers for one sentence

The Gemini row comes from the file d_genai.py saved (outputs/gemini_test_labels.csv),
so this script runs offline and does not spend your free quota a second time.
"""

import csv
import sys
import time
from collections import Counter

from a_rules import rules_label
from b_supervised import train_model
from c_clusters import embed, find_groups
from common import load_and_split
from d_genai import RESULTS_FILE

NEEDS_LIVE_RUN = "needs live run"


# ---------------------------------------------------------------------------
# Score each approach on the test rows
# ---------------------------------------------------------------------------

def timed(fn, rows: int):
    """Run fn() and return (its result, milliseconds per row).

    Why this exists: the scoreboard compares speed as well as accuracy.
    One timer means rules, the trained model, and k-means are measured
    the same way.

    Called by: score_rules, score_supervised, and score_clusters.
    """
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000 / rows


def score_rules(test) -> tuple:
    """Score approach A on the test rows. Return (how many right, ms per row).

    Why this exists: the scoreboard must call the same rules_label as
    a_rules.py, not a second copy of the keywords.

    What it does: rules_label from a_rules.py on each test sentence.
    No training. The timer covers those calls.

    Called by: build_rows.
    """
    guesses, ms = timed(lambda: [rules_label(t) for t in test["text"]], len(test))
    return sum(g == t for g, t in zip(guesses, test["label"])), ms


def score_supervised(train, test) -> tuple:
    """Score approach B. Train first, then time only the predictions.

    Why this exists: training is a one-time cost. The scoreboard's ms/row
    is inference, the same idea as predict.py.

    What it does: train_model from b_supervised.py on the 30 training rows,
    then model.predict on the 10 test rows inside timed().

    Called by: build_rows.
    """
    model = train_model(train["clean"], train["label"])                  # training is not timed
    guesses, ms = timed(lambda: model.predict(test["clean"]), len(test))  # inference is
    return int((guesses == test["label"]).sum()), ms


def fit_named_groups(train, embeddings=None):
    """Cluster the training sentences, then name each group by its most common label.

    Why this exists: k-means returns group numbers, not "complaint".
    The scoreboard needs a label so it can count right and wrong.
    The human labels are used only after grouping, to name the groups.
    They are not shown to k-means while it groups.

    What it does: embed() and find_groups() from c_clusters.py, then
    Counter.most_common on the labels inside each group.

    Called by: score_clusters and four_answers.
    These groups are fit on the 30 training rows, so the group numbers
    do not match `python c_clusters.py`, which groups all 40.
    """
    vectors = embed(train["text"], embeddings)
    kmeans = find_groups(vectors)
    names = {}
    for group in set(kmeans.labels_):
        labels_in_group = train["label"][kmeans.labels_ == group]
        names[group] = Counter(labels_in_group).most_common(1)[0][0]
    return kmeans, names


def local_embeddings(embeddings=None):
    """Return an embeddings model, loading the local one only if none was passed.

    Why this exists: get_embeddings() from llm.py takes a few seconds.
    The scoreboard should load it once and not include that wait in ms/row.
    Tests pass fake_embeddings() and this function returns that object
    without downloading MiniLM.

    Called by: score_clusters and four_answers.
    """
    if embeddings is None:
        from llm import get_embeddings
        embeddings = get_embeddings()
    return embeddings


def score_clusters(train, test, embeddings=None) -> tuple:
    """Score approach C on the test rows. Return (how many right, ms per row).

    Why this exists: the scoreboard needs a number for clustering, and
    cluster ids alone cannot be compared to praise/complaint/suggestion.

    What it does: local_embeddings(), fit_named_groups() on train, then
    time kmeans.predict(embed(test)). Map each group id through the names
    from fit_named_groups and count matches against the human labels.

    Called by: build_rows.
    """
    embeddings = local_embeddings(embeddings)
    kmeans, names = fit_named_groups(train, embeddings)
    groups, ms = timed(lambda: kmeans.predict(embed(test["text"], embeddings)), len(test))
    guesses = [names[g] for g in groups]
    return sum(g == t for g, t in zip(guesses, test["label"])), ms


def saved_gemini_score(test, path=RESULTS_FILE):
    """Read the CSV that d_genai.py saved. Return (right, ms per row), or None.

    Why this exists: `python compare.py` must work offline and must not
    spend the Gemini quota a second time. It does not call classify or get_llm.

    What it does: open outputs/gemini_test_labels.csv. If the ids are not
    the current test ids, return None (the split changed; run d_genai.py again).
    If the file is missing, return None and the scoreboard prints "needs live run".

    Called by: build_rows, and tests/test_pipeline.py / tests/test_genai.py.
    """
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [int(r["id"]) for r in rows] != list(test["id"]):
        return None   # saved for a different split: run d_genai.py again
    right = sum(r["gemini"] == r["label"] for r in rows)
    return right, sum(float(r["ms"]) for r in rows) / len(rows)


# ---------------------------------------------------------------------------
# The scoreboard table
# ---------------------------------------------------------------------------

COLUMNS = [("Approach", 20), ("AI map ring", 17), ("Accuracy", 15), ("Labels", 7),
           ("Training", 9), ("Internet", 10), ("ms/row", 15), ("Idea from", 11)]


def format_ms(ms) -> str:
    """Turn milliseconds into a short cell, or "needs live run" when ms is None.

    Why this exists: rules take a fraction of a millisecond and Gemini takes
    hundreds. One formatter keeps the column readable.

    Called by: format_scoreboard.
    """
    if ms is None:
        return NEEDS_LIVE_RUN
    return f"{ms:.3f}" if ms < 1 else f"{ms:.1f}" if ms < 100 else f"{ms:.0f}"


def format_scoreboard(rows: list, total: int) -> str:
    """Build the scoreboard text. Does not run any model.

    Why this exists: printing is separate from scoring, so tests can check
    the table shape with fake numbers.

    What it does: one header, a rule line, then one row per approach.
    right is None when Gemini has not been run yet.

    Called by: show_scoreboard and tests/test_pipeline.py.
    """
    lines = ["".join(f"{name:<{width}}" for name, width in COLUMNS).rstrip()]
    lines.append("-" * len(lines[0]))
    for r in rows:
        accuracy = NEEDS_LIVE_RUN if r["right"] is None else f"{r['right']}/{total} ({r['right'] / total:.0%})"
        cells = [r["approach"], r["ring"], accuracy, r["labels"], r["training"], r["internet"],
                 format_ms(r["ms"]), r["era"]]
        lines.append("".join(f"{cell:<{width}}" for cell, (_, width) in zip(cells, COLUMNS)).rstrip())
    return "\n".join(lines)


def build_rows(train, test, embeddings=None, gemini_file=RESULTS_FILE) -> list:
    """Score A, B, and C, and read D from disk. Return four row dicts.

    Why this exists: show_scoreboard and the tests both need the same four
    rows. The tests pass fake embeddings and a missing Gemini file.

    What it does: score_rules, score_supervised, score_clusters, then
    saved_gemini_score. D's right and ms stay None when that file is missing.

    Called by: show_scoreboard and tests/test_pipeline.py.
    """
    a_right, a_ms = score_rules(test)
    b_right, b_ms = score_supervised(train, test)
    c_right, c_ms = score_clusters(train, test, embeddings)
    gemini = saved_gemini_score(test, gemini_file)
    d_right, d_ms = gemini if gemini else (None, None)
    return [
        {"approach": "A Keyword rules", "ring": "AI (rules)", "right": a_right, "ms": a_ms,
         "labels": "no", "training": "no", "internet": "no", "era": "1950s-80s"},
        {"approach": "B TF-IDF + LogReg", "ring": "Machine learning", "right": b_right, "ms": b_ms,
         "labels": "yes", "training": "yes", "internet": "no", "era": "1990s"},
        {"approach": "C MiniLM + k-means", "ring": "Deep learning", "right": c_right, "ms": c_ms,
         "labels": "no*", "training": "yes", "internet": "1st run", "era": "2010s"},
        {"approach": "D Gemini zero-shot", "ring": "GenAI", "right": d_right, "ms": d_ms,
         "labels": "no", "training": "no", "internet": "yes", "era": "2020s"},
    ]


def show_scoreboard(embeddings=None, gemini_file=RESULTS_FILE) -> str:
    """Load the split, score the four approaches, and print the table.

    Why this exists: `python compare.py` with no sentence takes this path.
    It does not call four_answers and it does not call Gemini.

    What it does: load_and_split() from common.py, build_rows, format_scoreboard.

    Called by: main, when the command has no extra words.
    """
    train, test = load_and_split()
    rows = build_rows(train, test, embeddings, gemini_file)
    table = format_scoreboard(rows, len(test))
    print(f"Four kinds of AI, scored on the same {len(test)} test rows\n")
    print(table)
    print("\n* C never sees labels while grouping. We use the training labels only to NAME each group.")
    print("  Training = this program learns from our rows. Gemini was pre-trained by Google; we train nothing.")
    if rows[3]["right"] is None:
        print("  D: add your free Gemini key and run d_genai.py once, then run compare.py again.")
    return table


# ---------------------------------------------------------------------------
# Four answers for one sentence (the cold open)
# ---------------------------------------------------------------------------

def four_answers(sentence: str, embeddings=None, llm=None) -> dict:
    """Run all four approaches on one sentence. Return a dict of four answers.

    Why this exists: this is the cold-open demo.
    `python compare.py "Great job, the dosa was cold again."`
    One sentence, four brains, so a new person can see they disagree.

    What it does, in order:
      1. rules_label from a_rules.py. No training.
      2. load_and_split and clean_text from common.py, then train_model
         from b_supervised.py, then model.predict on the cleaned sentence.
      3. local_embeddings, fit_named_groups, embed, then kmeans.predict.
      4. classify from d_genai.py, which calls get_llm unless a model was passed.
         A missing key is caught and stored as the setup message.

    Called by: main, when you pass a sentence. tests/test_genai.py calls it
    with fake embeddings and a fake llm.
    """
    from common import clean_text
    from d_genai import classify
    from llm import MissingAPIKeyError

    train, _ = load_and_split()
    embeddings = local_embeddings(embeddings)
    answers = {"A rules": rules_label(sentence)}
    answers["B trained model"] = train_model(train["clean"], train["label"]).predict([clean_text(sentence)])[0]
    kmeans, names = fit_named_groups(train, embeddings)
    group = kmeans.predict(embed([sentence], embeddings))[0]
    answers["C clusters"] = f"group {group + 1} (mostly {names[group]})"
    try:
        answer, _ = classify([sentence], llm)[0]
        answers["D Gemini"] = f"{answer.label} ({answer.reason})"
    except MissingAPIKeyError:
        answers["D Gemini"] = "needs your free Gemini key (run check_setup.py)"
    return answers


def main(argv=None) -> int:
    """Print the scoreboard, or four answers when words follow the script name.

    Why this exists: one script, two demos. No words -> show_scoreboard.
    Words -> four_answers. That split is why the scoreboard stays offline
    even though four_answers can call Gemini.

    Called by: the bottom of this file when you run `python compare.py`.
    """
    words = sys.argv[1:] if argv is None else argv
    if words:
        sentence = " ".join(words)
        print(f"\"{sentence}\"\n")
        for approach, answer in four_answers(sentence).items():
            print(f"  {approach:<16} {answer}")
    else:
        show_scoreboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
