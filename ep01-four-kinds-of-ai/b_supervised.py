"""
b_supervised.py - Approach B: supervised machine learning (classic ML).

We give the computer examples WITH answers (labels). It writes its own rules as numbers.
Data + answers -> learned rules (the "weights").

  TF-IDF:              turns each sentence into numbers, one per word (how important that word is here)
  Logistic regression: learns one weight per word for each label, then picks the label with the best score

Training happens once. Inference (predicting) reuses the frozen weights on new sentences.

Run:  python b_supervised.py
"""

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from common import LABELS, load_and_split

# Feature engineering: filler words carry no opinion, so we drop them.
# We do NOT use scikit-learn's built-in English list: it would also delete
# "not", "please" and "would", the very words that matter for this task.
FILLER_WORDS = ["the", "a", "an", "is", "are", "was", "were", "and", "to", "of", "in", "on", "at",
                "for", "my", "it", "this", "i", "me", "so", "just", "by", "with", "as", "we", "you", "be"]


def train_model(texts, labels) -> Pipeline:
    """Learn a classifier from cleaned sentences and their labels.

    Why this exists: approach B is the one place that trains. predict.py and
    compare.py must train the same way, so they import this function instead
    of copying the Pipeline.

    What it does:
      1. TfidfVectorizer turns each sentence into numbers. ngram_range=(1, 2)
         keeps single words and pairs, so "not fresh" is its own feature.
         FILLER_WORDS are dropped. "not" and "please" are kept on purpose.
      2. LogisticRegression learns one weight per feature for each label.
      3. model.fit is the training step. It returns the fitted Pipeline.
         Later code calls model.predict on new sentences. That is inference,
         and it does not learn again.

    Called by: main (below), predict.load_or_train, compare.score_supervised,
    compare.four_answers, and tests/test_pipeline.py.
    """
    model = Pipeline([
        # single words AND word pairs, so "not fresh" is a feature of its own
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words=FILLER_WORDS)),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    model.fit(texts, labels)
    return model


def top_words(model: Pipeline, n: int = 5) -> dict:
    """Return the n strongest words the model learned for each label.

    Why this exists: the weights are the rules the model wrote for itself.
    Printing them shows a person what the training actually learned
    (for example "cold" pointing at complaint).

    What it does: reads the TF-IDF word list and the logistic-regression
    coefficients, then keeps the n largest weights per label.

    Called by: main (below) and tests/test_pipeline.py.
    predict.py and compare.py do not call this. They only need predictions.
    """
    words = model.named_steps["tfidf"].get_feature_names_out()
    logreg = model.named_steps["logreg"]
    result = {}
    for label, weights in zip(logreg.classes_, logreg.coef_):
        biggest = weights.argsort()[::-1][:n]
        result[label] = [words[i] for i in biggest]
    return result


def main() -> None:
    """Train on 30 rows, score the 10 hidden rows, and print the mistakes.

    Why this exists: this is the script entry for `python b_supervised.py`.
    Other files import train_model and do not run this main.

    What it does: load_and_split() from common.py, train_model on train["clean"]
    and train["label"], then model.predict on test["clean"]. The confusion
    matrix shows which labels were swapped. top_words shows the learned words.

    Called by: the bottom of this file when you run the script. Nothing else.
    """
    train, test = load_and_split()

    start = time.perf_counter()
    model = train_model(train["clean"], train["label"])          # training (once)
    train_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    predicted = model.predict(test["clean"])                      # inference (every new sentence)
    ms_per_row = (time.perf_counter() - start) * 1000 / len(test)

    right = int((predicted == test["label"]).sum())
    print(f"Trained on {len(train)} labelled rows in {train_ms:.0f} ms")
    print(f"Test accuracy on {len(test)} unseen rows: {right / len(test):.0%} ({right} of {len(test)})")
    print(f"Prediction time: {ms_per_row:.3f} ms per row")

    matrix = confusion_matrix(test["label"], predicted, labels=LABELS)
    print("\nConfusion matrix (rows = answer key, columns = model's guess):")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())

    print("\nTop five words the model taught itself, per label:")
    for label, words in top_words(model).items():
        print(f"  {label:<11} {', '.join(words)}")

    print("\nTest rows it got wrong:")
    for text, truth, guess in zip(test["text"], test["label"], predicted):
        if truth != guess:
            print(f"  \"{text}\"  key: {truth}, model: {guess}")


if __name__ == "__main__":
    main()
