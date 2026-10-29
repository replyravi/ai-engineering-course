"""
a_rules.py - Approach A: hand-written rules (rule-based, or "symbolic", AI).

A person writes the rules. The computer only follows them.
No examples, no training: rules + data -> answers.

Run:  python a_rules.py
"""

import re
import time

from common import load_feedback

# The rulebook is read from top to bottom. The FIRST label with a matching word wins.
RULEBOOK = {
    "suggestion": ["please", "could", "should", "would", "maybe", "suggest", "add"],
    "praise": ["good", "great", "love", "loved", "tasty", "delicious", "best", "amazing", "awesome",
               "excellent", "thanks", "thank", "friendly", "fresh", "perfect", "nice", "lovely"],
    "complaint": ["bad", "cold", "late", "slow", "dirty", "rude", "stale", "oily", "worst",
                  "expensive", "waited", "queue", "broken", "hair"],
}


def rules_label(text: str) -> str:
    """Label one comment using RULEBOOK. No training, no model.

    Why this exists: approach A shows AI that a person writes by hand.
    compare.py and the tests reuse this same function, so the rulebook
    lives in one place.

    What it does: lower-case the text, take whole words only (so "cold"
    does not match inside "scolded"), then walk RULEBOOK from top to bottom.
    The first keyword that is in the sentence wins. If nothing matches,
    return "unknown".

    Called by: main (below, on all 40 rows), compare.score_rules (the 10
    test rows), compare.four_answers (one sentence you type), and
    tests/test_rules.py.
    """
    words = set(re.findall(r"[a-z']+", text.lower()))   # the words in the sentence
    for label, keywords in RULEBOOK.items():
        for keyword in keywords:
            if keyword in words:
                return label
    return "unknown"   # no rule matched


def accuracy(predicted, truth) -> float:
    """Return the fraction of guesses that match the answer key. 1.0 means all right.

    Why this exists: a small helper for "how many did we get right?".
    main() counts matches itself and does not call this. The tests do,
    so the formula stays checked.

    Called by: tests/test_rules.py only.
    """
    right = sum(p == t for p, t in zip(predicted, truth))
    return right / len(truth)


def main() -> None:
    """Run approach A on all 40 rows and print accuracy, misses, and speed.

    Why this exists: this is the script entry for `python a_rules.py`.
    Importing rules_label from another file does not run main, because of
    the `if __name__ == "__main__"` check at the bottom.

    What it does: load_feedback() from common.py, then rules_label on every
    row, then compare those guesses to the human label column.

    Called by: the bottom of this file when you run the script. Nothing else.
    """
    df = load_feedback()

    start = time.perf_counter()
    df["rules"] = df["text"].map(rules_label)
    ms_per_row = (time.perf_counter() - start) * 1000 / len(df)

    right = int((df["rules"] == df["label"]).sum())
    print(f"Rules accuracy on all {len(df)} rows: {right / len(df):.0%} ({right} of {len(df)})")
    print(f"No rule matched at all: {(df['rules'] == 'unknown').sum()} rows")
    print(f"Time: {ms_per_row:.4f} ms per row")

    misses = df[df["rules"] != df["label"]]
    tricky = df[df["note"] != ""]
    tricky_right = int((tricky["rules"] == tricky["label"]).sum())
    print(f"Tricky rows (sarcasm or negation): {tricky_right} of {len(tricky)} right")

    print("\nTwo misses worth a look (a word fooled the rules):")
    fooled = misses[misses["rules"] != "unknown"]
    for note in ["sarcasm", "negation"]:
        row = fooled[fooled["note"] == note].iloc[0]
        print(f"  [{note}] \"{row['text']}\"")
        print(f"      rules said {row['rules']}, the answer key says {row['label']}")


if __name__ == "__main__":
    main()
