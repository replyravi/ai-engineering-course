"""
scout.py - Steps 7, 10, 11 and 12: run the Shelf Sifter crew from the terminal.

    python scout.py --stop-after scout          (Step 7: one agent, one tool call)
    python scout.py --stop-after analyst        (Step 10: + the Analyst's typed picks)
    python scout.py                             (Steps 11 and 12: the full crew, the guide, usage metrics)
    python scout.py --category poetry --interest "short poems about the sea"
    python scout.py --quiet                     (no step-by-step log, only the results)

After the run we check the scorecard, like after a cricket match: which tools REALLY ran,
did any tool fail, and do the picks match what was scraped? A confident answer is not proof.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import argparse
import json
from pathlib import Path

from crewai.tools import ToolExecutionFailedError

from crew import GUIDE_FILE, STAGES, build_crew
from llm import MissingAPIKeyError, get_model_name, get_provider
from scraper import CATEGORIES, category_url
from tools import Shelf

DEFAULT_INTEREST = "a long journey by train"


def differences(pick, books_by_title: dict) -> list[str]:
    """How one pick differs from the book the tool REALLY scraped (an empty list = exact match)."""
    real = books_by_title.get(pick.title)
    if real is None:
        return ["this title is not on the scraped shelf"]
    return [f"{field} is {getattr(pick, field)!r}, the shelf says {real[field]!r}"
            for field in ("price", "rating", "url") if getattr(pick, field) != real[field]]


def check_picks(picks, books: list[dict]) -> list[str]:
    """Compare every pick with the scraped shelf. Returns the problems found (empty = all good)."""
    by_title = {book["title"]: book for book in books}
    return [f"{pick.title!r}: {problem}" for pick in picks.picks for problem in differences(pick, by_title)]


def print_scorecard(result, shelf: Shelf, picks, guide_text: str | None) -> int:
    """Print who actually scored. Returns the number of problems (0 = all good)."""
    problems = 0
    print("\nScorecard (who actually scored)")
    for name in ("scrape_shelf", "find_similar_books"):
        count = shelf.successful_calls.count(name)
        extra = f", {len(shelf.books)} books from {shelf.source_url}" if name == "scrape_shelf" and count else ""
        print(f"  {name + ' ':.<26} {count} successful call(s){extra}")
    if not shelf.books:
        print("  PROBLEM: no tool returned any books, so every answer came from the model's imagination.")
        problems += 1
    print(f"  {'tool failures ':.<26} {len(result.tool_failures)}")
    problems += len(result.tool_failures)

    if picks is not None:
        by_title = {book["title"]: book for book in shelf.books}
        exact = sum(not differences(pick, by_title) for pick in picks.picks)
        print(f"  {'picks match the shelf ':.<26} {exact} of {len(picks.picks)}")
        for mismatch in check_picks(picks, shelf.books):
            print(f"  PROBLEM: {mismatch}")
        problems += len(picks.picks) - exact
    if guide_text is not None and picks is not None:
        linked = sum(pick.url in guide_text for pick in picks.picks)
        print(f"  {'pick links in the guide ':.<26} {linked} of {len(picks.picks)}")
        problems += len(picks.picks) - linked

    usage = result.token_usage     # CrewAI adds up every model call of the run
    print("\nUsage metrics (result.token_usage)")
    print(f"  requests {usage.successful_requests}   prompt tokens {usage.prompt_tokens}   "
          f"completion tokens {usage.completion_tokens}   reasoning tokens {usage.reasoning_tokens}   "
          f"total {usage.total_tokens}")
    return problems


def run(category: str = "travel", interest: str = DEFAULT_INTEREST, stop_after: str = "writer",
        llm=None, fetch=None, embed=None, verbose: bool = True) -> int:
    """Build the crew, run it once, print the results and the scorecard. Returns an exit code."""
    inputs = {"category": category, "interest": interest, "shelf_url": category_url(category)}
    team = ", ".join(STAGES[: STAGES.index(stop_after) + 1])
    print(f"Shelf Sifter crew: shelf={category!r}  interest={interest!r}  agents: {team}")

    shelf = Shelf()
    try:
        crew = build_crew(llm=llm, shelf=shelf, stop_after=stop_after, fetch=fetch, embed=embed, verbose=verbose)
    except MissingAPIKeyError as error:
        print(error)          # friendly help instead of a long traceback
        return 1
    model = getattr(llm, "model", None) or f"{get_provider()}/{get_model_name()}"
    print(f"Model: {model}   Process: {crew.process.value}   max_rpm: {crew.max_rpm}\n")

    try:
        result = crew.kickoff(inputs=inputs)
    except ToolExecutionFailedError as error:
        print("\nThe crew STOPPED because a tool call failed (tool_failure_policy='raise'):")
        print(f"  {error.record.summary()}")
        return 2
    except Exception as error:  # no internet, a wrong model name, the free-tier limit ...
        print(f"\nThe crew stopped with an error: {type(error).__name__}: {error}")
        print("(429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and run again)")
        return 1

    picks, guide_text, problems = None, None, 0
    print("\n" + "=" * 70)
    if stop_after == "scout":
        scout_answer = result.tasks_output[0].raw
        if scout_answer == json.dumps(shelf.books, ensure_ascii=False):
            print(f"The Scout's answer is the tool's JSON, untouched: a list of {len(shelf.books)} books. The first one:")
            print(json.dumps(shelf.books[0], ensure_ascii=False))
        else:
            print(f"PROBLEM: the Scout's answer is NOT the tool's JSON. It starts: {scout_answer[:120]!r}")
            problems += 1
    if stop_after in ("analyst", "writer"):
        picks = result.tasks_output[1].pydantic     # a real Picks object, not text
        if picks is None:
            print(f"PROBLEM: the Analyst's answer did not fit Picks. It starts: {result.tasks_output[1].raw[:120]!r}")
            return 3
        print(f"The Analyst's typed answer: {type(picks).__name__} with {len(picks.picks)} pick(s)")
        for number, pick in enumerate(picks.picks, start=1):
            print(f"  {number}. {pick.title}  £{pick.price:.2f}  rating {pick.rating}/5  - {pick.reason}")
    if stop_after == "writer":
        guide = Path(GUIDE_FILE.format(category=category))
        guide_text = guide.read_text(encoding="utf-8")
        print(f"\nThe Writer saved {guide.as_posix()} ({len(guide_text):,} characters):\n")
        print(guide_text)

    problems += print_scorecard(result, shelf, picks, guide_text)
    if problems:
        print(f"\n{problems} problem(s) found: do not trust this result. Read the log above.")
        return 3
    print("\nAll checks passed.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Shelf Sifter crew on one shelf of books.toscrape.com.")
    parser.add_argument("--category", default="travel", choices=sorted(CATEGORIES), help="which shelf (default: travel)")
    parser.add_argument("--interest", default=DEFAULT_INTEREST, help="what the reader likes, in plain words")
    parser.add_argument("--stop-after", default="writer", choices=STAGES, help="run only the first agents")
    parser.add_argument("--quiet", action="store_true", help="hide CrewAI's step-by-step log")
    args = parser.parse_args(argv)
    return run(args.category, args.interest, args.stop_after, verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
