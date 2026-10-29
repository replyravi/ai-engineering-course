"""
save_fixture.py - Step 3: download ONE shelf page politely and save it on your computer.

Why? While you build the parser you will run it many times. With a saved copy (a "fixture")
you parse the same page again and again, offline, without sending the site a single extra request.

    python save_fixture.py              (the Travel shelf -> fixtures/travel.html)
    python save_fixture.py mystery      (any shelf from scraper.CATEGORIES)

The saved page stays on your computer (.gitignore keeps it out of git): it is the site's page, not ours.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import sys

import requests

from scraper import CATEGORIES, FIXTURES, category_url, fetch_politely, parse_books


def main(category: str = "travel") -> int:
    if category not in CATEGORIES:
        print(f"Unknown shelf {category!r}. Pick one of: {', '.join(CATEGORIES)}")
        return 1
    url = category_url(category)
    print(f"Fetching {url}")
    try:
        html = fetch_politely(url)
    except (requests.RequestException, PermissionError) as error:
        print(f"Could not download the page: {error}")
        return 1

    target = FIXTURES / f"{category}.html"
    target.write_text(html, encoding="utf-8")
    print(f"Saved fixtures/{target.name} ({len(html):,} characters, {len(parse_books(html, url))} books on the page)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
