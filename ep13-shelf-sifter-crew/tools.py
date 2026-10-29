"""
tools.py - Steps 6 and 9: the two custom tools of our crew.

The model never sees your Python code. It only sees four things about a tool:
  1. its NAME           scrape_shelf
  2. its DESCRIPTION    one or two sentences: what it does and when to use it
  3. its ARGUMENTS      names, types and a description of each (the args_schema)
  4. what it RETURNS    here: JSON text
So write those four like instructions for a new colleague.

    python tools.py scrape              (Step 6: the Travel shelf, live and polite)
    python tools.py scrape mystery
    python tools.py similar "a long journey by train"   (Step 9: on your saved page, offline)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import json
import sys
from typing import Callable

from crewai.tools import BaseTool, ToolFailure
from pydantic import BaseModel, Field

from models import Book
from scraper import CATEGORIES, FIXTURES, category_url, fetch_politely, parse_books
from similarity import load_embedder, top_matches


class Shelf:
    """The books the Scout REALLY scraped. Both tools share one Shelf, like a notice board.

    scrape_shelf pins the books on it, find_similar_books searches them, and after the run
    scout.py reads it to see who actually scored (not just what the agents said).
    """

    def __init__(self):
        self.books: list[dict] = []
        self.source_url: str | None = None
        self.successful_calls: list[str] = []   # one entry per tool call that worked

    def fill(self, books: list[dict], url: str) -> None:
        self.books, self.source_url = books, url


# ---------------------------------------------------------------------------
# Tool 1: scrape_shelf (Step 6)
# ---------------------------------------------------------------------------

class ScrapeShelfInput(BaseModel):
    """The arguments of scrape_shelf. The model reads these names, types and descriptions."""

    url: str = Field(
        description="The full address of ONE shelf page of https://books.toscrape.com, for example "
                    "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
    )


class ScrapeShelfTool(BaseTool):
    name: str = "scrape_shelf"
    description: str = (
        "Read one shelf (category page) of the practice bookshop books.toscrape.com and return its books "
        "as a JSON list. Each book has title, price (in pounds), rating (1 to 5 stars), stock and url. "
        "It checks robots.txt and waits 2 seconds between requests, so call it once per shelf."
    )
    args_schema: type[BaseModel] = ScrapeShelfInput
    shelf: Shelf                                    # where the scraped books are pinned
    fetch: Callable[[str], str] = fetch_politely    # tests pass a fake fetch that reads a saved page

    def _run(self, url: str):
        try:
            html = self.fetch(url)
        except Exception as error:   # wrong site, robots.txt says no, no internet, HTTP 404 ...
            # ToolFailure tells CrewAI "this call FAILED". A plain error string would look like a success.
            return ToolFailure(message=f"Could not read {url}: {error}")

        books = [Book(**row).model_dump() for row in parse_books(html, url)]   # Book checks every row
        if not books:
            return ToolFailure(message=f"No books found on {url}. Is it a shelf (category) page?")
        self.shelf.fill(books, url)
        self.shelf.successful_calls.append(self.name)
        return json.dumps(books, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: find_similar_books (Step 9)
# ---------------------------------------------------------------------------

class FindSimilarInput(BaseModel):
    query: str = Field(description="What the reader likes, in plain words, for example 'a long journey by train'")


class FindSimilarBooksTool(BaseTool):
    name: str = "find_similar_books"
    description: str = (
        "Find the 3 books on the shelf the Scout scraped whose titles are closest in meaning to the query. "
        "Returns a JSON list with title, score (higher means closer, 1.0 is the maximum), price, rating and url. "
        "Works only after scrape_shelf has run."
    )
    args_schema: type[BaseModel] = FindSimilarInput
    shelf: Shelf
    embed: Callable | None = None   # None = load the local MiniLM model the first time it is needed

    def _run(self, query: str):
        if not self.shelf.books:
            return ToolFailure(message="The shelf is empty: scrape_shelf has not returned any books yet.")
        if self.embed is None:
            self.embed = load_embedder()
        matches = top_matches(query, self.shelf.books, self.embed)
        self.shelf.successful_calls.append(self.name)
        return json.dumps(matches, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Try each tool on its own, BEFORE any agent uses it.
# ---------------------------------------------------------------------------

def show_what_the_model_sees(tool: BaseTool) -> None:
    print(f"Tool name:    {tool.name}")
    print(f"Description:  {tool.description}")
    print(f"Arguments:    {json.dumps(tool.args_schema.model_json_schema()['properties'])}\n")


def main(args: list[str]) -> int:
    command = args[0] if args else "scrape"
    shelf = Shelf()

    if command == "scrape":
        category = args[1] if len(args) > 1 else "travel"
        if category not in CATEGORIES:
            print(f"Unknown shelf {category!r}. Pick one of: {', '.join(CATEGORIES)}")
            return 1
        tool = ScrapeShelfTool(shelf=shelf)
        show_what_the_model_sees(tool)
        result = tool.run(url=category_url(category))
        if isinstance(result, ToolFailure):
            print(f"FAILED: {result.message}")
            return 1
        books = json.loads(result)
        print(f"\ntool.run(url=...) returned {len(result):,} characters of JSON: a list of {len(books)} books.")
        print("First book:", json.dumps(books[0], ensure_ascii=False))
        return 0

    if command == "similar":
        query = " ".join(args[1:]) or "a long journey by train"
        saved = FIXTURES / "travel.html"
        if not saved.exists():
            raise SystemExit("No saved page yet. Run Step 3 first:  python save_fixture.py")
        shelf.fill(parse_books(saved.read_text(encoding="utf-8"), category_url("travel")), "fixtures/travel.html")
        tool = FindSimilarBooksTool(shelf=shelf)
        show_what_the_model_sees(tool)
        result = tool.run(query=query)
        print(f"tool.run(query={query!r}) returned:")
        print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))
        return 0

    print("Use:  python tools.py scrape [category]   or   python tools.py similar \"your words\"")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
