"""
shelf_helpers.py - small fakes shared by the Episode 13 tests (no internet, no model download).

  handmade_page()   the hand-made test shelf (fixtures/handmade_shelf.html): 3 invented books
  handmade_fetch    a fake fetch(url): returns that page instead of downloading anything
  word_embed        a fake embeddings model: texts that share words get similar vectors
  FakeResponse      a pretend requests.Response for the polite-fetcher tests
"""

from pathlib import Path

import numpy as np
import requests

from scraper import category_url

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
TRAVEL_URL = category_url("travel")

SLOW_TRAIN = "The Slow Train to Kanyakumari: Three Weeks on Indian Railways"
SALT_AND_SNOW = "Salt and Snow: Walking the High Passes of Ladakh"
COFFEE_STOPS = "Coffee Stops of the Old Silk Road"


def handmade_page() -> str:
    return (FIXTURES / "handmade_shelf.html").read_text(encoding="utf-8")


def handmade_fetch(url: str) -> str:
    """Pretend to download `url`: always returns the hand-made shelf page."""
    return handmade_page()


def word_embed(texts: list[str]) -> list[np.ndarray]:
    """A fake embeddings model: one number per vocabulary word, so shared words = similar vectors.

    Good enough to test the TOOL (sorting, scores, JSON). Real meaning needs the real model.
    """
    vocabulary = ["train", "slow", "railways", "snow", "walking", "passes", "coffee", "road", "silk"]
    vectors = []
    for text in texts:
        words = text.lower().replace(":", " ").split()
        vector = np.array([words.count(word) for word in vocabulary], dtype=float) + 0.01
        vectors.append(vector)
    return vectors


class FakeResponse(requests.Response):
    """A real requests.Response object filled by hand (no internet)."""

    def __init__(self, status_code: int = 200, text: str = "", content_type: str = "text/html"):
        super().__init__()
        self.status_code = status_code
        self._content = text.encode("utf-8")
        self.headers["Content-Type"] = content_type
        # requests.get() guesses the encoding from the header, like this:
        self.encoding = requests.utils.get_encoding_from_headers(self.headers)


class FakeGet:
    """A fake requests.get: answers robots.txt and pages from a dict, and remembers every call."""

    def __init__(self, pages: dict[str, FakeResponse]):
        self.pages = pages
        self.calls: list[str] = []

    def __call__(self, url, headers=None, timeout=None):
        self.calls.append(url)
        self.last_headers = headers
        return self.pages.get(url, FakeResponse(404, "not found"))
