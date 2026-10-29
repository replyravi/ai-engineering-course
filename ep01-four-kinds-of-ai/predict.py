"""
predict.py - deploy and monitor, in miniature (step 14).

    python predict.py "the sambar was cold"

Deploy:  use the trained model from a file. The first run trains it (approach B) and saves it.
Monitor: every prediction is added to logs/predictions.csv, so you can check later
         what real users typed and how sure the model was.
"""

import csv
import sys
from datetime import datetime

import joblib

from b_supervised import train_model
from common import HERE, clean_text, load_and_split

MODEL_FILE = HERE / "outputs" / "feedback_model.joblib"
LOG_FILE = HERE / "logs" / "predictions.csv"


def load_or_train(model_file=MODEL_FILE):
    """Return (model, trained_now). Train only when no saved file exists yet.

    Why this exists: deploy means "use the model from a file", not "train
    again on every sentence". The first call creates outputs/feedback_model.joblib.
    Later calls load that file.

    What it does: if the file exists, joblib.load it and return trained_now=False.
    If it does not, call load_and_split() and train_model() from b_supervised.py,
    save with joblib.dump, and return trained_now=True.
    Only load a joblib file you created. Loading one can run code.

    Called by: main (below). Tests call main, which calls this, with a temp path.
    """
    if model_file.exists():
        # joblib files can run code when loaded: only load model files YOU made.
        return joblib.load(model_file), False
    train, _ = load_and_split()
    model = train_model(train["clean"], train["label"])
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    return model, True


def predict(text: str, model) -> tuple:
    """Return (label, confidence) for one new sentence. Confidence is 0 to 1.

    Why this exists: the saved model expects the same cleaning as training.
    This function cleans first, then asks for probabilities, so a caller
    cannot forget clean_text.

    What it does: clean_text() from common.py, then model.predict_proba.
    That returns one chance per label. The biggest chance is the label.
    That chance is the confidence.

    Called by: main (below). This is not compare.four_answers. That path
    calls model.predict itself after train_model.
    """
    chances = model.predict_proba([clean_text(text)])[0]   # one chance per label
    best = chances.argmax()
    return model.classes_[best], float(chances[best])


def log_prediction(text: str, label: str, confidence: float, log_file=LOG_FILE) -> int:
    """Append one prediction to logs/predictions.csv. Return how many rows exist now.

    Why this exists: monitor means "keep what real sentences were typed and
    how sure the model was", so you can look back later.

    What it does: creates logs/ if needed, writes the header on the first
    call, then appends time, text, label, and confidence.

    Called by: main (below).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "text", "label", "confidence"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), text, label, f"{confidence:.2f}"])
    with open(log_file, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1   # minus the header row


def main(argv=None, model_file=MODEL_FILE, log_file=LOG_FILE) -> int:
    """Label one sentence from the command line, then log it.

    Why this exists: this is the tiny deployed app.
    `python predict.py "the sambar was cold"`

    What it does: load_or_train(), predict(), log_prediction().
    Returns 2 when you forget the sentence, 0 when a label was printed.
    Tests pass argv, model_file, and log_file so they do not touch the real outputs/.

    Called by: the bottom of this file, and tests/test_pipeline.py.
    """
    words = sys.argv[1:] if argv is None else argv
    if not words:
        print('Usage: python predict.py "your feedback sentence"')
        return 2
    text = " ".join(words)

    model, trained_now = load_or_train(model_file)
    if trained_now:
        print(f"No saved model yet: trained one and saved it to outputs/{model_file.name}")
    else:
        print(f"Loaded the saved model from outputs/{model_file.name}")

    label, confidence = predict(text, model)
    print(f"\"{text}\" -> {label} (confidence {confidence:.0%})")

    count = log_prediction(text, label, confidence, log_file)
    print(f"Logged to logs/{log_file.name} ({count} prediction{'' if count == 1 else 's'} so far)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
