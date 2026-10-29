"""
scraper.py - Steps 3, 4 and 6: fetch a shelf page politely, then clean its HTML into rows.

Two jobs, kept apart on purpose:
  fetch_politely(url)    -> the page's HTML (robots.txt checked, at least 2 seconds between requests)
  parse_books(html, url) -> a list of clean rows: title, price, rating, stock, url

parse_books never touches the internet, so we can test it with a saved page (Step 5).

    python scraper.py      (parses the page you saved in Step 3 and prints three books)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from robots_check import USER_AGENT, is_allowed, load_robots

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"

SITE = "https://books.toscrape.com/"
ALLOWED_HOSTS = {"books.toscrape.com"}   # our tools only ever visit the practice site
POLITE_DELAY_SECONDS = 2.0                # at least this long between two requests

# A few shelves of the practice bookshop (the number is part of the site's address).
CATEGORIES = {
    "travel": "travel_2",
    "mystery": "mystery_3",
    "philosophy": "philosophy_7",
    "music": "music_14",
    "science": "science_22",
    "poetry": "poetry_23",
    "humor": "humor_30",
    "history": "history_32",
    "food-and-drink": "food-and-drink_33",
}

# The site shows stars as a CSS class: <p class="star-rating Four">
RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def category_url(category: str) -> str:
    """The address of the first page of a shelf, e.g. category_url("travel")."""
    return f"{SITE}catalogue/category/books/{CATEGORIES[category]}/index.html"


# ---------------------------------------------------------------------------
# 1. Clean the HTML into rows (Step 4)
# ---------------------------------------------------------------------------

def parse_books(html: str, page_url: str) -> list[dict]:
    """Turn one shelf page into a list of book rows. Works offline."""
    soup = BeautifulSoup(html, "html.parser")
    books = []
    for card in soup.select("article.product_pod"):          # one card per book
        link = card.select_one("h3 a")
        stars = card.select_one("p.star-rating")["class"]     # ["star-rating", "Four"]
        price_text = card.select_one("p.price_color").get_text(strip=True)   # "£45.17"
        books.append({
            "title": link["title"].strip(),                   # the link text is cut short with "...", the title attribute is not
            "price": float(re.sub(r"[^0-9.]", "", price_text)),   # keep only digits and the dot
            "rating": next(RATING_WORDS[c] for c in stars if c in RATING_WORDS),
            "stock": card.select_one("p.availability").get_text(strip=True),   # "In stock"
            "url": urljoin(page_url, link["href"]),           # "../../../x/index.html" -> a full address
        })
    return books


# ---------------------------------------------------------------------------
# 2. Fetch politely (Steps 3 and 6)
# ---------------------------------------------------------------------------

class PoliteFetcher:
    """Downloads pages like a polite visitor.

    1. Only the practice site.        2. Only pages robots.txt allows.
    3. A pause between requests.      4. A clear User-Agent and a timeout.

    get, sleep, clock and log can be swapped for fakes in tests, so tests never use the internet.
    """

    def __init__(self, delay=POLITE_DELAY_SECONDS, get=requests.get, sleep=time.sleep,
                 clock=time.monotonic, log=print):
        self.delay = delay
        self.get = get
        self.sleep = sleep
        self.clock = clock
        self.log = log
        self.robots = {}             # one robots.txt per site, downloaded once
        self.last_request_at = None  # when we sent our previous request

    def wait_your_turn(self) -> None:
        """Sleep until `delay` seconds have passed since our previous request."""
        if self.last_request_at is not None:
            waited = self.clock() - self.last_request_at
            if waited < self.delay:
                self.log(f"  waiting {self.delay - waited:.1f} s (polite delay)")
                self.sleep(self.delay - waited)
        self.last_request_at = self.clock()

    def check_robots(self, url: str) -> None:
        """Raise PermissionError if robots.txt does not allow this page."""
        host = urlsplit(url).netloc
        if host not in self.robots:
            self.wait_your_turn()                   # downloading robots.txt is a request too
            self.robots[host], status = load_robots(url, get=self.get)
            self.log(f"  robots.txt: HTTP {status}")
            crawl_delay = self.robots[host].crawl_delay(USER_AGENT)
            if crawl_delay and float(crawl_delay) > self.delay:
                self.delay = float(crawl_delay)     # the site asks for a longer pause: we obey
        if not is_allowed(self.robots[host], url):
            raise PermissionError(f"robots.txt does not allow {url}")
        self.log("  robots.txt: allowed")

    def fetch(self, url: str) -> str:
        """Return the HTML of `url`, politely."""
        if urlsplit(url).netloc not in ALLOWED_HOSTS:
            raise ValueError(f"this scraper only visits {', '.join(sorted(ALLOWED_HOSTS))}, not {url!r}")
        self.check_robots(url)
        self.wait_your_turn()
        response = self.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()                 # 404 or 500 -> an error, not an empty page
        # The server does not say which text encoding it uses ("text/html" only), so requests
        # would guess ISO-8859-1 and every "£" would come out as "Â£". The page itself says UTF-8.
        if "charset" not in response.headers.get("Content-Type", "").lower():
            response.encoding = "utf-8"
        self.log(f"  GET {url} -> HTTP {response.status_code} ({len(response.content):,} bytes)")
        return response.text


_shared_fetcher = PoliteFetcher()   # one fetcher per program, so the delay counts across all calls


def fetch_politely(url: str) -> str:
    """Download one page of the practice site, politely (see PoliteFetcher)."""
    return _shared_fetcher.fetch(url)


if __name__ == "__main__":
    saved = FIXTURES / "travel.html"
    if not saved.exists():
        raise SystemExit("No saved page yet. Run Step 3 first:  python save_fixture.py")
    books = parse_books(saved.read_text(encoding="utf-8"), category_url("travel"))
    print(f"Parsed {len(books)} books from fixtures/{saved.name}. The first three:\n")
    for book in books[:3]:
        print(book)
