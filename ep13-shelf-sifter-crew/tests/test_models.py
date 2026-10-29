"""
test_models.py - the Picks shape (Step 10) and the scorecard check of scout.py (Step 12).
"""

import pytest
from pydantic import ValidationError

from models import Pick, Picks
from scout import check_picks
from shelf_helpers import SLOW_TRAIN, TRAVEL_URL, handmade_page
from scraper import parse_books


def a_pick(**changes) -> dict:
    pick = {"title": SLOW_TRAIN, "price": 12.5, "rating": 4,
            "url": "https://books.toscrape.com/catalogue/the-slow-train-to-kanyakumari_901/index.html",
            "reason": "Four stars for the lowest price."}
    return {**pick, **changes}


def test_picks_accepts_a_good_answer():
    picks = Picks(category="travel", interest="trains", picks=[a_pick()])

    assert picks.picks[0] == Pick(**a_pick())


@pytest.mark.parametrize("bad", [
    {"picks": [a_pick(rating=7)]},           # stars go from 1 to 5
    {"picks": []},                           # at least one pick
    {"picks": [a_pick()] * 4},               # at most three picks
    {"picks": [a_pick(price="cheap")]},      # a price is a number
])
def test_picks_refuses_answers_that_break_the_shape(bad):
    with pytest.raises(ValidationError):
        Picks(category="travel", interest="trains", **bad)


def test_scorecard_accepts_picks_copied_exactly():
    books = parse_books(handmade_page(), TRAVEL_URL)

    assert check_picks(Picks(category="travel", interest="trains", picks=[a_pick()]), books) == []


def test_scorecard_catches_a_changed_price_and_an_invented_book():
    books = parse_books(handmade_page(), TRAVEL_URL)
    picks = Picks(category="travel", interest="trains",
                  picks=[a_pick(price=10.0), a_pick(title="A Book Nobody Scraped")])

    problems = check_picks(picks, books)

    assert problems == [
        f"{SLOW_TRAIN!r}: price is 10.0, the shelf says 12.5",
        "'A Book Nobody Scraped': this title is not on the scraped shelf",
    ]
