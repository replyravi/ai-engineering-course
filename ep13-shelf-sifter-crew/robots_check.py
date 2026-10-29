"""
robots_check.py - Step 2: ask the website for permission BEFORE we scrape it.

robots.txt is a small text file at the root of a website (https://site/robots.txt).
It tells robots (scrapers, search engines) which paths they may visit.
Python reads it for us: urllib.robotparser is part of the standard library.

    python robots_check.py                  (checks the Travel shelf of books.toscrape.com)
    python robots_check.py <page-url>       (checks any page you want to scrape)

robots.txt is only half of the permission. Also read the site's terms. books.toscrape.com
is a practice bookshop made for learning web scraping (see the README, Step 2).

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

HERE = Path(__file__).resolve().parent

# Say who we are. A polite scraper sends a clear User-Agent with a way to find out more.
# Sharing your own copy? Put your project name and link here.
USER_AGENT = "shelf-sifter-crew/1.0 (learning project; https://github.com/replyravi/ai-engineering-course)"

# books.toscrape.com is a practice bookshop made for learning web scraping.
TRAVEL_SHELF = "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"

# A robots.txt written by hand for this course, to see what "Disallow" rules do.
SAMPLE_ROBOTS = HERE / "fixtures" / "robots_sample.txt"

# What the HTTP status of robots.txt means for us (the same rules Python's RobotFileParser uses).
STATUS_MEANING = {
    200: "the site has rules; we follow them",
    401: "the site says no robots allowed, so we may fetch nothing",
    403: "the site says no robots allowed, so we may fetch nothing",
    404: "the site has no robots.txt, so it sets no rules",
}


def robots_url(page_url: str) -> str:
    """The robots.txt address for any page: always at the root of the site."""
    parts = urlsplit(page_url)
    return f"{parts.scheme}://{parts.netloc}/robots.txt"


def parse_robots(text: str) -> RobotFileParser:
    """Turn the text of a robots.txt file into a parser we can ask questions."""
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return parser


def load_robots(page_url: str, get=requests.get, timeout: int = 10):
    """Download robots.txt for the site of page_url. Returns (parser, HTTP status code).

    We download it with requests (so the site sees our User-Agent) and follow the same
    rules as Python's own RobotFileParser.read():
      200        -> read the rules
      401 / 403  -> the site says "no robots": nothing is allowed
      other 4xx  -> there is no robots.txt: no rules, so everything is allowed
      5xx        -> the site has a problem: do not scrape now (be careful, not clever)
    `get` can be swapped for a fake in tests, so tests never use the internet.
    """
    response = get(robots_url(page_url), headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if response.status_code == 200:
        return parse_robots(response.text), 200       # the site has rules: read them

    parser = RobotFileParser()
    if response.status_code in (401, 403) or response.status_code >= 500:
        parser.disallow_all = True                     # "no robots" or the site has trouble: do not scrape
    else:
        parser.allow_all = True                        # 404: no robots.txt, so no rules
    return parser, response.status_code


def is_allowed(parser: RobotFileParser, page_url: str, user_agent: str = USER_AGENT) -> bool:
    """May our User-Agent fetch this page?"""
    return parser.can_fetch(user_agent, page_url)


def main(page_url: str = TRAVEL_SHELF) -> int:
    print(f"Robots file:  {robots_url(page_url)}")
    try:
        parser, status = load_robots(page_url)
    except requests.RequestException as error:
        print(f"Could not download robots.txt: {error}")
        print("No answer means no permission. Check your internet connection and try again.")
        return 1
    print(f"HTTP status:  {status} ({STATUS_MEANING.get(status, 'see load_robots() for what this means')})")
    print(f"User-Agent:   {USER_AGENT}")
    print(f"Allowed: {is_allowed(parser, page_url)}   {page_url}")

    # The practice site has no rules, so here is what rules look like (a file we wrote ourselves).
    sample = parse_robots(SAMPLE_ROBOTS.read_text(encoding="utf-8"))
    print(f"\nPractice rules from fixtures/{SAMPLE_ROBOTS.name} (written by hand, not the real site):")
    for path in ["/catalogue/category/books/travel_2/index.html", "/basket/", "/accounts/login/"]:
        print(f"  Allowed: {str(is_allowed(sample, 'https://books.toscrape.com' + path)):<5}  {path}")
    print(f"  Wait between requests (Crawl-delay): {sample.crawl_delay(USER_AGENT)} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
