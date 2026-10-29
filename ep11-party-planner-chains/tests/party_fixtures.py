"""
tests/party_fixtures.py - shared test helpers for this episode.

Test files import what they need, for example:
    from party_fixtures import MENU, no_keys, small_prices  # noqa: F401

- no_keys:       no provider settings and no keys (even if your real .env has one)
- small_prices:  a tiny prices.csv in a temporary folder, with a note line
- MENU, ITEMS:   a scripted menu and a scripted typed grocery list for the fake model
"""

import pytest

from costing import Ingredient, Ingredients

MENU = "Lemon rice - tangy\nCurd rice - cooling\nVada - crisp\nKesari - sweet"

ITEMS = Ingredients(items=[
    Ingredient(item="rice", qty_per_person=100, unit="g"),
    Ingredient(item="curd", qty_per_person=50, unit="g"),
    Ingredient(item="coconut", qty_per_person=0.25, unit="piece"),
])

SMALL_PRICES = """# sample prices for a test
item,unit,price_rupees
rice,kg,60
curd,kg,70
coconut,piece,35
milk,litre,58
"""


@pytest.fixture
def no_keys(monkeypatch):
    for name in ["LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def small_prices(tmp_path, monkeypatch):
    """Write a 4-item price list and make costing.load_prices() read it."""
    import costing

    path = tmp_path / "prices.csv"
    path.write_text(SMALL_PRICES, encoding="utf-8")
    monkeypatch.setattr(costing, "PRICES_FILE", path)
    return path
