"""
fixtures/make_fixture.py - how fixtures/crawl.json was made (you do not need to run it).

It reads the three rule-book pages in site/ and writes them in the same shape that
crawl.py saves after a real Apify crawl: {"about", "start_url", "max_pages", "pages": [...]},
each page with url, title and text. The urls are where the pages live once published.

It is an offline copy of OUR OWN pages, made on this computer. It is NOT a recorded Apify run:
Apify's text extraction may break lines a little differently.

    python fixtures/make_fixture.py

Author: Ravi Shanker Singh (TechWithRavi)
"""

import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
EPISODE = HERE.parent
sys.path.insert(0, str(EPISODE))      # so "import crawl" works when this file is run directly

from crawl import DEFAULT_START_URL, MAX_PAGES  # noqa: E402

SITE = EPISODE / "site"
PAGES = [("index.html", ""), ("fees.html", "fees.html"), ("units.html", "units.html")]   # (file, url path)


def page_text(html: str) -> tuple[str, str]:
    """The title and the readable text of one page: every heading, paragraph and list item on its own line."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    lines = [tag.get_text(" ", strip=True) for tag in main.find_all(["h1", "h2", "p", "li"])]
    return soup.title.get_text(strip=True), "\n".join(line for line in lines if line)


def main():
    pages = []
    for file_name, url_path in PAGES:
        title, text = page_text((SITE / file_name).read_text(encoding="utf-8"))
        pages.append({"url": DEFAULT_START_URL + url_path, "title": title, "text": text})
    data = {
        "about": ("Offline copy of the 3 rule-book pages in site/, made by fixtures/make_fixture.py "
                  "in the same shape crawl.py saves. Not a recorded Apify run."),
        "start_url": DEFAULT_START_URL,
        "max_pages": MAX_PAGES,
        "pages": pages,
    }
    target = HERE / "crawl.json"
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote fixtures/{target.name}: {len(pages)} pages, "
          f"{sum(len(p['text']) for p in pages):,} characters of text")


if __name__ == "__main__":
    main()
