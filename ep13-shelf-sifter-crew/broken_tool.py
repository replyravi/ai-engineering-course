"""
broken_tool.py - Step 8: break the scrape tool on purpose, read the red error, fix it.

The BROKEN tool has two classic mistakes:
  1. a wrong type hint:   url: int   (a web address is text, so it must be url: str)
  2. a lazy description:  "Scrape."  (the model cannot tell what to pass, or in what form)
The @tool decorator turns the type hints and the docstring into what the model sees,
so the two mistakes go straight into the model's instructions.

    python broken_tool.py             the Scout with the BROKEN tool (live, needs your key)
    python broken_tool.py --fixed     the same Scout with the FIXED tool (live)
    python broken_tool.py --replay    both runs offline: a SCRIPTED model, no key, no internet

In --replay the "model" is scripted (testing/fakes.py), so you see the same error every time.
It plays the call any model would make here: the web address, sent as text.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import argparse
import json

from crewai import Agent, Crew, Process, Task
from crewai.tools import ToolExecutionFailedError, tool

from llm import MissingAPIKeyError, get_crewai_llm
from scraper import FIXTURES, category_url, fetch_politely, parse_books

TRAVEL_URL = category_url("travel")
FETCH = {"page": fetch_politely}   # --replay swaps in a function that reads a saved page


def scrape(url: str) -> str:
    """The real work, shared by both tools: fetch the page politely and clean it into JSON."""
    return json.dumps(parse_books(FETCH["page"](url), url), ensure_ascii=False)


# ---------------------------------------------------------------------------
# BROKEN: wrong type hint + lazy description
# ---------------------------------------------------------------------------
@tool("scrape_shelf")
def broken_scrape_shelf(url: int) -> str:
    """Scrape."""
    return scrape(str(url))


# ---------------------------------------------------------------------------
# FIXED: the right type hint + a description that says what to pass
# ---------------------------------------------------------------------------
@tool("scrape_shelf", result_as_answer=True)
def fixed_scrape_shelf(url: str) -> str:
    """Read one shelf (category page) of the practice bookshop books.toscrape.com and return its
    books as a JSON list (title, price, rating, stock, url). Pass the full page address as url, for
    example https://books.toscrape.com/catalogue/category/books/travel_2/index.html"""
    return scrape(url)


def show_what_the_model_sees(scrape_tool, label: str) -> None:
    print(f"Tool {scrape_tool.name!r} ({label}) - what the model sees:")
    # The description is the docstring. Python 3.13 removes its indent, 3.11 and 3.12 keep it,
    # so we print every line with the same indent: the output looks the same on every version.
    first, *more = scrape_tool.description.splitlines()
    print(f"  description: {first}")
    for line in more:
        print(f"    {line.strip()}")
    print(f"  arguments:   {json.dumps(scrape_tool.args_schema.model_json_schema()['properties'])}\n")


def scout_crew(scrape_tool, llm, verbose: bool = True) -> Crew:
    """A one-agent crew, like Step 7, with the tool we want to test."""
    scout = Agent(
        role="Shelf Sifter",
        goal="Read the travel shelf of the practice bookshop and report its books exactly as scraped",
        backstory="You are careful and polite. You only report what the scrape_shelf tool returned.",
        tools=[scrape_tool], llm=llm, max_iter=3, verbose=verbose,
    )
    task = Task(
        description=f"Call the scrape_shelf tool once, with this url: {TRAVEL_URL}",
        expected_output="The JSON list of books that scrape_shelf returned.",
        agent=scout,
    )
    return Crew(agents=[scout], tasks=[task], process=Process.sequential,
                tool_failure_policy="raise", verbose=verbose)


def run_once(scrape_tool, label: str, llm) -> int:
    show_what_the_model_sees(scrape_tool, label)
    try:
        result = scout_crew(scrape_tool, llm).kickoff()
    except ToolExecutionFailedError as error:
        print("\nThe crew STOPPED because a tool call failed. The key lines of the error:")
        for line in str(error).splitlines():
            if line.strip().startswith(("Tool '", "url", "Input should", "Expected arguments")):
                print(f"  {line.strip()}")
        if label == "BROKEN":
            print("\nHow to read it: the tool asks for a NUMBER (url: int), but a web address is text.")
            print("Fix the type hint (url: str) and the description, then run:  python broken_tool.py --fixed")
        else:
            print("\nThe fixed tool itself failed: read the error (no internet? a typo in the url?).")
        return 2
    try:
        books = json.loads(result.raw)
    except json.JSONDecodeError:
        print("\nThe crew finished, but the answer is NOT the tool's JSON: the model answered on its own.")
        print(f"It starts: {result.raw[:120]!r}")
        return 3
    print(f"\nClean run: one tool call, no errors. The Scout's answer is a JSON list of {len(books)} books.")
    return 0


def replay() -> int:
    """Both runs offline. The model is SCRIPTED, the tools and CrewAI are real."""
    from testing.fakes import FakeCrewAILLM, react_tool_use

    handmade = (FIXTURES / "handmade_shelf.html").read_text(encoding="utf-8")
    FETCH["page"] = lambda url: handmade     # no internet: read the hand-made test page
    call = react_tool_use("scrape_shelf", {"url": TRAVEL_URL}, thought="I will read the shelf")

    print("REPLAY 1 of 2: the BROKEN tool (scripted model, no internet)\n")
    first = run_once(broken_scrape_shelf, "BROKEN", FakeCrewAILLM(responses=[call]))
    print("\n" + "=" * 70)
    print("REPLAY 2 of 2: the FIXED tool, same scripted call (the page is fixtures/handmade_shelf.html)\n")
    second = run_once(fixed_scrape_shelf, "FIXED", FakeCrewAILLM(responses=[call]))
    return 0 if (first, second) == (2, 0) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Step 8: break the scrape tool on purpose, then fix it.")
    parser.add_argument("--fixed", action="store_true", help="use the fixed tool")
    parser.add_argument("--replay", action="store_true", help="both runs offline with a scripted model")
    args = parser.parse_args(argv)
    if args.replay:
        return replay()

    scrape_tool, label = (fixed_scrape_shelf, "FIXED") if args.fixed else (broken_scrape_shelf, "BROKEN")
    try:
        llm = get_crewai_llm()
    except MissingAPIKeyError as error:
        show_what_the_model_sees(scrape_tool, label)
        print(error)
        print("No key yet? See the same error offline:  python broken_tool.py --replay")
        return 1
    return run_once(scrape_tool, label, llm)


if __name__ == "__main__":
    raise SystemExit(main())
