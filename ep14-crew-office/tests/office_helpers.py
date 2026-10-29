"""
office_helpers.py - small helpers shared by the Episode 14 tests (no internet, no key).

  ensure_prices()    make data/prices.csv and data/news.csv if they are missing (same seed, same files)
  series(...)        a tiny price series with weekday dates, for hand-checked maths
  FixedSentiment     a pretend sentiment model that always gives the scores we choose
  CountingLLM        a scripted model that also reports pretend token usage (to test CrewAI's counting)
"""

import os

# CrewAI keeps the task outputs of the latest run in a small file in your user data folder,
# inside a folder named after the current folder. Give the tests' runs their own name.
os.environ.setdefault("CREWAI_STORAGE_DIR", "ep14-crew-office-tests")

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

import make_prices  # noqa: E402
from replay import ScriptedLLM  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
KESAR = Path("leads/kesar-crumb-bakehouse.md")


def ensure_prices() -> None:
    if not (PROJECT / "data" / "prices.csv").is_file() or not (PROJECT / "data" / "news.csv").is_file():
        prices, news = make_prices.make_prices()
        make_prices.write_csv(prices, PROJECT / "data" / "prices.csv")
        make_prices.write_csv(news, PROJECT / "data" / "news.csv")


def series(*closes: float, start: str = "2026-01-05") -> pd.Series:
    return pd.Series(closes, index=pd.bdate_range(start, periods=len(closes)), dtype=float)


class FixedSentiment:
    name = "fixed (test)"

    def __init__(self, negative=0.0, neutral=1.0, positive=0.0, per_sentence=None):
        self.overall = {"negative": negative, "neutral": neutral, "positive": positive}
        self.per_sentence = per_sentence or {}

    def scores(self, text):
        if text in self.per_sentence:
            return {"negative": self.per_sentence[text], "neutral": 0.0, "positive": 0.0}
        return self.overall


class CountingLLM(ScriptedLLM):
    """Every call reports 100 prompt + 10 completion tokens (pretend numbers, for counting tests)."""

    def call(self, messages, *args, **kwargs):
        reply = super().call(messages, *args, **kwargs)
        self._track_token_usage_internal({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
        return reply
