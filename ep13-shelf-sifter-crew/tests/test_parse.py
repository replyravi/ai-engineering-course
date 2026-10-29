"""
test_parse.py - Step 5: test the HTML cleaner on a saved page, before any agent uses it.

The page is fixtures/handmade_shelf.html: written by hand with the same tags and classes as a
real shelf page of books.toscrape.com, and three invented books. No internet needed.
"""

from scraper import parse_books
from shelf_helpers import COFFEE_STOPS, SALT_AND_SNOW, SLOW_TRAIN, TRAVEL_URL, handmade_page


def test_finds_every_book_on_the_page():
    books = parse_books(handmade_page(), TRAVEL_URL)

    assert [book["title"] for book in books] == [SLOW_TRAIN, SALT_AND_SNOW, COFFEE_STOPS]


def test_cleans_each_book_into_a_row():
    first = parse_books(handmade_page(), TRAVEL_URL)[0]

    assert first == {
        "title": SLOW_TRAIN,
        "price": 12.5,          # "£12.50" became a number
        "rating": 4,            # class="star-rating Four" became 4
        "stock": "In stock",    # spaces and line breaks removed
        "url": "https://books.toscrape.com/catalogue/the-slow-train-to-kanyakumari_901/index.html",
    }


def test_uses_the_full_title_not_the_shortened_link_text():
    # The page shows "Salt and Snow: Walking the ..." as link text; the title attribute has it all.
    second = parse_books(handmade_page(), TRAVEL_URL)[1]

    assert second["title"] == SALT_AND_SNOW
    assert "..." not in second["title"]
