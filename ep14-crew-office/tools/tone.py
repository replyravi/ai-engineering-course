"""
tools/tone.py - Step 3: a REAL tone tool, tested with known sentences before any agent uses it.

Older tutorials gave the agent a "sentiment tool" whose code always answered "positive".
The agent asked again and again, got the same word, and finally gave up: a tool that never
changes its answer teaches the agent nothing. Our tool returns information that can change
the Reviewer's next decision:

  1. sentiment scores (negative, neutral, positive) for the whole email AND for each sentence,
     from a real local model: cardiffnlp/twitter-roberta-base-sentiment-latest (CC BY 4.0,
     by Cardiff NLP), in the ONNX copy that Hugging Face's Xenova account publishes, so no
     PyTorch is needed. Downloaded once (about 500 MB), then it works offline.
  2. pushy phrases ("act now", "last chance" ...) and exclamation marks, from a word list
  3. the length in words
  -> a verdict: "ok", or "check" with the reasons (for example, which sentence reads cold)

Why this model? We tested a famous two-label model first (SST-2, trained on film reviews).
It called "The order is ready for pickup." NEGATIVE, so it would have blocked every polite
business email. A model with a NEUTRAL label fits emails better. Test the scale first!

The limits we chose ("ok" means: no sentence 0.60 or more negative, the whole email under
0.50 negative, no pushy phrase, at most 1 exclamation mark, at most 150 words) are OUR rules.
Change them if your emails need different ones.

    python -m tools.tone                         (the known test sentences)
    python -m tools.tone "Your text here"        (any text)
    python -m tools.tone --words "Your text"     (the small word-list fallback, no model)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import json
import re
import sys
from pathlib import Path

import numpy as np
from crewai.tools import BaseTool, ToolFailure
from pydantic import BaseModel, ConfigDict, Field

from notebook import CaseFile

# The ONNX copy of cardiffnlp/twitter-roberta-base-sentiment-latest (same weights, CC BY 4.0).
MODEL_ID = "Xenova/twitter-roberta-base-sentiment-latest"
REVISION = "f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445"      # the exact version we tested (2026-10-26)
MODEL_FILES = ["config.json", "tokenizer.json", "onnx/model.onnx"]

PUSHY_PHRASES = ["act now", "last chance", "limited time", "hurry", "don't miss", "do not miss",
                 "urgent", "expires", "only today", "guaranteed", "risk-free", "you will regret",
                 "once in a lifetime", "buy now", "100%"]
COLD_SENTENCE = 0.60      # a sentence this negative (or more) "reads cold"
COLD_EMAIL = 0.50         # the whole email this negative (or more) reads negative
MAX_WORDS = 150
MAX_EXCLAMATIONS = 1

# The small fallback: a few words people use in warm and in cold emails.
WARM_WORDS = {"congratulations", "thank", "thanks", "glad", "happy", "love", "great", "welcome",
              "helpful", "useful", "easy", "simple", "warm", "pleased", "enjoy", "wonderful"}
COLD_WORDS = {"problem", "problems", "losing", "lost", "miss", "missed", "wrong", "regret", "bad",
              "fail", "failed", "costing", "angry", "complaint", "late", "worse", "mess", "ruin", "ruins"}

KNOWN_SENTENCES = [
    "Congratulations on opening your fourth outlet. We would love to help your team handle the festival rush.",
    "Act now! This offer expires tonight and you will regret missing it.",
    "Your staff keep losing pre-orders and that is costing you customers.",
    "The order is ready for pickup.",
]


# ---------------------------------------------------------------------------
# 1. Two ways to score sentiment: the real model, and a tiny word list
# ---------------------------------------------------------------------------

class LocalSentimentModel:
    """Twitter-RoBERTa sentiment, run with onnxruntime. Same scores as the transformers pipeline."""

    name = "twitter-roberta-sentiment (local ONNX)"

    def __init__(self, download: bool = True):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        # local_files_only=True: use the copy on this computer, never the internet (the tests do this).
        paths = {f: Path(hf_hub_download(MODEL_ID, f, revision=REVISION, local_files_only=not download))
                 for f in MODEL_FILES}
        self.labels = json.loads(paths["config.json"].read_text(encoding="utf-8"))["id2label"]
        self.tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        self.tokenizer.enable_truncation(max_length=512)        # the model reads at most 512 tokens
        self.session = onnxruntime.InferenceSession(str(paths["onnx/model.onnx"]),
                                                    providers=["CPUExecutionProvider"])

    def scores(self, text: str) -> dict[str, float]:
        tokens = self.tokenizer.encode(text)
        logits = self.session.run(None, {
            "input_ids": np.array([tokens.ids], dtype=np.int64),
            "attention_mask": np.array([tokens.attention_mask], dtype=np.int64),
        })[0][0]
        chances = np.exp(logits - logits.max())
        chances /= chances.sum()                                  # softmax: three chances that add up to 1
        return {self.labels[str(i)]: round(float(c), 4) for i, c in enumerate(chances)}


class WordListSentiment:
    """A tiny fallback: count warm and cold words. Crude, but tested, and honest about 'neutral'."""

    name = "word list (fallback)"

    def scores(self, text: str) -> dict[str, float]:
        words = re.findall(r"[a-z']+", text.lower())
        warm = sum(word in WARM_WORDS for word in words)
        cold = sum(word in COLD_WORDS for word in words)
        total = warm + cold + 1                                   # the +1 is a little "neutral" weight
        return {"negative": round(cold / total, 4), "neutral": round(1 / total, 4),
                "positive": round(warm / total, 4)}


def model_is_downloaded() -> bool:
    """True when the sentiment model is already on this computer."""
    try:
        LocalSentimentModel(download=False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. The tone report: the information the Reviewer acts on
# ---------------------------------------------------------------------------

def sentences(text: str) -> list[str]:
    """Split an email into sentences and lines ("Dear Nandini," is a line of its own)."""
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]


def label_of(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def tone_report(text: str, sentiment) -> dict:
    overall = sentiment.scores(text)
    cold = []
    for number, sentence in enumerate(sentences(text), start=1):
        negative = sentiment.scores(sentence)["negative"]
        if negative >= COLD_SENTENCE:
            cold.append({"sentence": number, "negative": negative, "text": sentence})
    lowered = text.lower()
    pushy = [phrase for phrase in PUSHY_PHRASES if phrase in lowered]
    exclamations = text.count("!")
    words = len(text.split())

    reasons = []
    if overall["negative"] >= COLD_EMAIL:
        reasons.append(f"the whole email reads negative ({overall['negative']})")
    reasons += [f"sentence {c['sentence']} reads cold ({c['negative']}): {c['text']!r}" for c in cold]
    if pushy:
        reasons.append(f"pushy phrases: {', '.join(pushy)}")
    if exclamations > MAX_EXCLAMATIONS:
        reasons.append(f"{exclamations} exclamation marks, keep it to {MAX_EXCLAMATIONS} or none")
    if words > MAX_WORDS:
        reasons.append(f"{words} words, keep it to {MAX_WORDS} or fewer")
    return {
        "label": label_of(overall), "scores": overall, "engine": sentiment.name,
        "cold_sentences": cold, "pushy_phrases": pushy, "exclamation_marks": exclamations,
        "words": words, "verdict": "ok" if not reasons else "check", "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 3. The CrewAI tool
# ---------------------------------------------------------------------------

class ToneInput(BaseModel):
    text: str = Field(description="The full email body to check, as plain text")


class CheckToneTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "check_tone"
    description: str = (
        "Check the tone of an email body with a local sentiment model and a pushy-phrase list. "
        "Returns JSON: the label and scores of the whole email, the sentences that read cold, pushy "
        "phrases, exclamation marks, the word count, and a verdict: 'ok', or 'check' with the "
        "reasons. Call it once per draft."
    )
    args_schema: type[BaseModel] = ToneInput
    sentiment: object = None          # None = load the local model the first time it is needed
    case: CaseFile | None = None

    def _run(self, text: str):
        if self.sentiment is None:
            try:
                self.sentiment = LocalSentimentModel()
            except Exception as error:   # not downloaded and no internet, for example
                return ToolFailure(message=f"Could not load the sentiment model ({type(error).__name__}). "
                                           "Run this once with internet:  python -m tools.tone")
        if self.case is not None:
            self.case.note_call(self.name)
        return json.dumps(tone_report(text, self.sentiment))


# ---------------------------------------------------------------------------
# Try it by hand, before any agent uses it
# ---------------------------------------------------------------------------

def main(args: list[str]) -> int:
    use_words = "--words" in args
    texts = [a for a in args if a != "--words"] or KNOWN_SENTENCES
    if use_words:
        sentiment = WordListSentiment()
    else:
        print(f"Loading {MODEL_ID} (ONNX). The first run downloads about 500 MB once ...")
        sentiment = LocalSentimentModel()
    print(f"Engine: {sentiment.name}\n")
    print(f"{'label':<9}{'negative':>9}{'neutral':>9}{'positive':>9}  verdict")
    for text in texts:
        report = tone_report(text, sentiment)
        s = report["scores"]
        print(f"{report['label']:<9}{s['negative']:>9.4f}{s['neutral']:>9.4f}{s['positive']:>9.4f}  "
              f"{report['verdict']}")
        print(f"  {text}")
        for reason in report["reasons"]:
            print(f"    - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
