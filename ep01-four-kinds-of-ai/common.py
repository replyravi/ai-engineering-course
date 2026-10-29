"""
common.py - load, clean and split the Byte Bites feedback (shared by every approach).

Step 6:  load the CSV and look at it          (lifecycle stage: collect)
Step 8:  clean the text and split train/test  (lifecycle stage: clean)

Run:  python common.py
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "feedback.csv"
LABELS = ["praise", "complaint", "suggestion"]

SEED = 42        # a fixed seed: the "random" split is the same on every run and every laptop
TEST_ROWS = 10   # rows we hide from training, to test on later


# ---------------------------------------------------------------------------
# Step 6 - load
# ---------------------------------------------------------------------------

def load_feedback(path=DATA_FILE) -> pd.DataFrame:
    """Read data/feedback.csv into a table (a pandas DataFrame).

    Why this exists: every approach must start from the same 40 comments.
    One function means nobody opens the file a different way.

    What it does: reads id, text, label, and note. keep_default_na=False
    keeps an empty note as "" so later checks like note != "" still work.

    Called by: load_and_split (below), a_rules.main, c_clusters.main,
    e_reply_and_probe.main, and tests/test_pipeline.py.
    """
    df = pd.read_csv(path, keep_default_na=False)   # keep empty notes as "" (not NaN)
    return df


def show_dataset(df: pd.DataFrame) -> None:
    """Print how many rows and labels are in the table, plus the first 3 rows.

    Why this exists: step 6 is "look at the data" before any model runs.
    It does not change the table.

    Called by: the bottom of this file, only when you run `python common.py`.
    Other scripts do not call it.
    """
    print(f"{len(df)} rows, {df['label'].nunique()} labels")
    for label, count in df["label"].value_counts().items():
        print(f"  {label:<11} {count}")
    tricky = df[df["note"] != ""]
    notes = ", ".join(f"{count} {name}" for name, count in tricky["note"].value_counts().items())
    print(f"Rows marked tricky: {len(tricky)} ({notes})")
    print()
    print(df.head(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Step 8 - clean and split
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Turn one comment into a simpler string that programs can compare.

    Why this exists: "DON'T" and "do not" are the same idea. If we leave
    them different, the trained model treats them as different words and
    misses the word "not", which flips praise into a complaint.

    What it does, in order:
      1. lower case
      2. can't / won't / n't become "not" (the NOT is kept on purpose)
      3. drop punctuation
      4. squeeze extra spaces

    Called by: load_and_split (every row), predict.predict (one new sentence),
    compare.four_answers (one new sentence), show_split (a printed example),
    and tests/test_pipeline.py.
    """
    text = text.lower()
    text = text.replace("can't", "can not").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)        # didn't -> did not, wasn't -> was not
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # keep only letters, digits and spaces
    return re.sub(r"\s+", " ", text).strip()


def split_feedback(df: pd.DataFrame, test_rows: int = TEST_ROWS, seed: int = SEED):
    """Hide 10 rows for testing and keep 30 for training.

    Why this exists: a model that is scored on the same rows it trained on
    looks better than it is. The test rows stay unseen.

    What it does: stratify keeps praise, complaint, and suggestion in both
    parts. seed 42 makes the same 10 test ids on every laptop. Rows are
    sorted by id so printed output does not jump around.

    Called by: load_and_split, and tests/test_pipeline.py (also with seed 7,
    to prove a different seed really changes the split).
    """
    train, test = train_test_split(df, test_size=test_rows, random_state=seed, stratify=df["label"])
    return train.sort_values("id"), test.sort_values("id")


def load_and_split():
    """Load the CSV, add a clean column, and return (train, test).

    Why this exists: steps 9 to 15 all need the same cleaned 30/10 split.
    They call this instead of repeating the three steps.

    What it does: load_feedback(), then clean_text on every row into df["clean"],
    then split_feedback(). The original text stays in df["text"].

    Called by: b_supervised.main, d_genai.main, predict.load_or_train,
    compare.show_scoreboard, compare.four_answers, and tests/test_pipeline.py.
    """
    df = load_feedback()
    df["clean"] = df["text"].map(clean_text)
    return split_feedback(df)


def show_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print the train/test sizes, the test ids, and one cleaning example.

    Why this exists: you should see that the hidden rows are the same every
    time you run the script. It does not train anything.

    Called by: the bottom of this file, only when you run `python common.py`.
    """
    print(f"train {len(train)}, test {len(test)}")
    print("test ids:", ", ".join(str(i) for i in test["id"]))
    print("test labels:", ", ".join(f"{label} {count}" for label, count in test["label"].value_counts().items()))
    print()
    example = "Please DON'T stop the Friday pav bhaji!!"
    print("Cleaning example:")
    print("  before:", example)
    print("  after: ", clean_text(example))


if __name__ == "__main__":
    show_dataset(load_feedback())
    print("\n" + "-" * 60 + "\n")
    show_split(*load_and_split())
