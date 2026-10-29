"""
test_tools.py - test each TOOL on its own, before an agent uses it (Steps 6, 8 and 9).

Every tool gets a fake fetch (the hand-made page) or a fake embeddings model, so nothing is
downloaded. One test uses the real local MiniLM model, only if it is already on this computer.
"""

import json

import pytest
from crewai.tools import ToolFailure
from pydantic import ValidationError

from broken_tool import broken_scrape_shelf, fixed_scrape_shelf
from shelf_helpers import COFFEE_STOPS, SALT_AND_SNOW, SLOW_TRAIN, TRAVEL_URL, handmade_fetch, word_embed
from similarity import load_embedder, model_is_downloaded
from tools import FindSimilarBooksTool, ScrapeShelfTool, Shelf
from wrap_langchain_tool import make_agent, price_per_star


# ---------------------------------------------------------------------------
# scrape_shelf
# ---------------------------------------------------------------------------

def test_scrape_tool_returns_json_and_pins_the_books_on_the_shelf():
    shelf = Shelf()
    tool = ScrapeShelfTool(shelf=shelf, fetch=handmade_fetch)

    books = json.loads(tool.run(url=TRAVEL_URL))

    assert [book["title"] for book in books] == [SLOW_TRAIN, SALT_AND_SNOW, COFFEE_STOPS]
    assert shelf.books == books
    assert shelf.successful_calls == ["scrape_shelf"]


def test_the_model_sees_a_name_a_description_and_one_text_argument():
    tool = ScrapeShelfTool(shelf=Shelf())
    arguments = tool.args_schema.model_json_schema()["properties"]

    assert tool.name == "scrape_shelf"
    assert "books.toscrape.com" in tool.description
    assert list(arguments) == ["url"]
    assert arguments["url"]["type"] == "string"
    assert "travel_2" in arguments["url"]["description"]    # an example of a good value


def test_a_failed_download_is_reported_as_a_failure_not_as_an_answer():
    def no_internet(url):
        raise ConnectionError("no internet")

    shelf = Shelf()
    result = ScrapeShelfTool(shelf=shelf, fetch=no_internet).run(url=TRAVEL_URL)

    assert isinstance(result, ToolFailure)
    assert "no internet" in result.message
    assert shelf.books == [] and shelf.successful_calls == []


def test_a_page_without_books_is_a_failure():
    result = ScrapeShelfTool(shelf=Shelf(), fetch=lambda url: "<html><p>Hello</p></html>").run(url=TRAVEL_URL)

    assert isinstance(result, ToolFailure)
    assert "No books found" in result.message


# ---------------------------------------------------------------------------
# find_similar_books
# ---------------------------------------------------------------------------

def shelf_with_handmade_books() -> Shelf:
    shelf = Shelf()
    ScrapeShelfTool(shelf=shelf, fetch=handmade_fetch).run(url=TRAVEL_URL)
    return shelf


def test_similarity_tool_returns_three_matches_best_first():
    tool = FindSimilarBooksTool(shelf=shelf_with_handmade_books(), embed=word_embed)

    matches = json.loads(tool.run(query="a slow train"))

    assert [match["title"] for match in matches][0] == SLOW_TRAIN
    assert len(matches) == 3
    scores = [match["score"] for match in matches]
    assert scores == sorted(scores, reverse=True)
    assert set(matches[0]) == {"title", "score", "price", "rating", "url"}


def test_similarity_tool_needs_a_scraped_shelf_first():
    result = FindSimilarBooksTool(shelf=Shelf(), embed=word_embed).run(query="trains")

    assert isinstance(result, ToolFailure)
    assert "scrape_shelf" in result.message


@pytest.mark.skipif(not model_is_downloaded(), reason="Local model not downloaded yet: run python similarity.py once")
def test_real_local_model_matches_meaning_not_just_words():
    tool = FindSimilarBooksTool(shelf=shelf_with_handmade_books(), embed=load_embedder())

    matches = json.loads(tool.run(query="a long journey by train"))
    snow = json.loads(tool.run(query="mountain walks in the snow"))

    assert matches[0]["title"] == SLOW_TRAIN and matches[0]["score"] == pytest.approx(0.563, abs=0.002)
    assert snow[0]["title"] == SALT_AND_SNOW      # "mountain walks" ~ "Walking the High Passes"


# ---------------------------------------------------------------------------
# Step 8: the broken tool and the fixed tool
# ---------------------------------------------------------------------------

def test_broken_tool_asks_the_model_for_a_number():
    assert broken_scrape_shelf.args_schema.model_json_schema()["properties"]["url"]["type"] == "integer"
    assert broken_scrape_shelf.description == "Scrape."
    with pytest.raises(ValidationError, match="valid integer"):
        broken_scrape_shelf.args_schema(url=TRAVEL_URL)


def test_fixed_tool_asks_for_text_and_says_what_to_pass():
    assert fixed_scrape_shelf.args_schema.model_json_schema()["properties"]["url"]["type"] == "string"
    assert "full page address" in fixed_scrape_shelf.description
    assert fixed_scrape_shelf.result_as_answer is True


# ---------------------------------------------------------------------------
# A LangChain tool inside CrewAI
# ---------------------------------------------------------------------------

def test_a_langchain_tool_must_be_wrapped_first():
    from crewai.tools.base_tool import Tool

    with pytest.raises(ValidationError, match="BaseTool"):
        make_agent([price_per_star])                 # the raw LangChain tool is rejected

    wrapped = Tool.from_langchain(price_per_star)
    make_agent([wrapped])                            # accepted
    assert wrapped.name == "price_per_star"
    assert wrapped.run(price=26.08, rating=5) == 5.22
