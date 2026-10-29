"""
desk_crew.py - Step 12: Crew B, an EDUCATION-ONLY research desk on invented prices.

Three agents, one after the other (Process.sequential):

  Market Analyst      price_indicators + read_data_file (+ calculator)  -> what moved, what it means
  Strategy Explainer  paper_backtest (+ calculator)                     -> what the paper test showed
  Risk Reviewer       (+ calculator)                                    -> a typed DeskReport, with risks

Models explain, code computes: every number comes from a tool and is written into the Ledger.
Then OUR code writes reports/<ticker>.md: it adds a table of the tool numbers, stamps the
disclaimer on top and at the end (claims_check.py), and checks that every number in the
agents' text came from a tool. Today's date is passed in, so no model has to guess the year.

No trading, no broker, no live prices, no buy, sell or hold. DEMO1 to DEMO3 are invented.

    python desk_crew.py --ticker DEMO1           (Step 12, needs your key)
    python desk_crew.py --ticker DEMO1 --replay  (a SCRIPTED model, no key, no internet)
    python desk_crew.py --replay --plant        (the scripted Analyst "guesses" one number)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import argparse
import time
from datetime import date
from pathlib import Path

from crewai import Agent, Crew, Process, Task

from approve import shown
from claims_check import MODEL_END, MODEL_START, check_report, print_result, stamp_disclaimer
from llm import MissingAPIKeyError, get_crewai_llm, get_model_name, get_provider
from make_prices import COMPANIES
from models import DeskReport
from notebook import Ledger
from tools.backtest import PaperBacktestTool, backtest_for
from tools.calculator import CalculatorTool
from tools.files import data_file_tool
from tools.indicators import PRICES_FILE, PriceIndicatorsTool, indicators_for

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
NEWS_FILE = HERE / "data" / "news.csv"
ROLES = ["Market Analyst", "Strategy Explainer", "Risk Reviewer"]
NUMBER_RULES = ("Rules: copy every number exactly as a tool gives it, written as digits. If you need a "
                "number no tool gave, get it from the calculator tool. Never say whether anyone should "
                "buy, sell or hold, and never give a price target or a position size.")


# ---------------------------------------------------------------------------
# 1. The crew
# ---------------------------------------------------------------------------

def make_agents(llm_for, ledger: Ledger, verbose: bool = True) -> dict:
    """llm_for(role) gives each agent its own model (Episode 13: never share one LLM object)."""
    analyst = Agent(
        role="Market Analyst",
        goal="Explain in plain words what moved {ticker}, using only numbers the tools return",
        backstory="You read the scoreboard; you never guess the score. This is a lesson on invented "
                  "data for beginners, not a tip.",
        tools=[PriceIndicatorsTool(ledger=ledger), data_file_tool(ledger), CalculatorTool(ledger=ledger)],
        llm=llm_for("Market Analyst"), max_iter=5, allow_delegation=False, verbose=verbose,
    )
    strategy = Agent(
        role="Strategy Explainer",
        goal="Explain what a paper backtest of one simple rule showed on {ticker}, and what it cannot show",
        backstory="You teach how trading rules are TESTED. You never suggest trading.",
        tools=[PaperBacktestTool(ledger=ledger), CalculatorTool(ledger=ledger)],
        llm=llm_for("Strategy Explainer"), max_iter=4, allow_delegation=False, verbose=verbose,
    )
    risk = Agent(
        role="Risk Reviewer",
        goal="Turn the desk's notes on {ticker} into a careful research note with its risks and limits",
        backstory="You protect learners from overconfidence. You remove anything that sounds like advice.",
        tools=[CalculatorTool(ledger=ledger)],
        llm=llm_for("Risk Reviewer"), max_iter=3, allow_delegation=False, verbose=verbose,
    )
    return {"Market Analyst": analyst, "Strategy Explainer": strategy, "Risk Reviewer": risk}


def make_tasks(agents: dict) -> list[Task]:
    analyst_task = Task(
        description="Explain what moved {ticker} ({company}, an invented company) for a beginner. "
                    "Today is {today}.\n"
                    "1. Call price_indicators once with ticker {ticker}.\n"
                    "2. Call read_data_file once with path data/news.csv and find the invented headlines "
                    "for {ticker} on the days of its biggest moves.\n"
                    "3. Write two short paragraphs: what moved (with dates), and what each number means.\n"
                    + NUMBER_RULES,
        expected_output="Two short paragraphs; every number copied from a tool.",
        agent=agents["Market Analyst"],
    )
    strategy_task = Task(
        description="Explain the paper backtest of {ticker}. Today is {today}.\n"
                    "1. Call paper_backtest once with ticker {ticker}.\n"
                    "2. In one paragraph: the rule in plain words, both paper returns, their difference, "
                    "and why one invented year of prices proves nothing.\n" + NUMBER_RULES,
        expected_output="One paragraph; every number copied from the tool.",
        agent=agents["Strategy Explainer"],
    )
    risk_task = Task(
        description="Write the research note on {ticker} as a DeskReport, from the answers above. "
                    "Fill what_moved, what_the_numbers_mean and paper_backtest from them and keep every "
                    "number exactly as written there. Add 2 to 5 risks and limits a learner should know. "
                    "Remove any sentence that sounds like advice. Today is {today}; the data ends on the "
                    "'to' date the tools gave.\n" + NUMBER_RULES,
        expected_output="A DeskReport: ticker, what_moved, what_the_numbers_mean, paper_backtest, risks.",
        agent=agents["Risk Reviewer"],
        context=[analyst_task, strategy_task],
        output_pydantic=DeskReport,
    )
    return [analyst_task, strategy_task, risk_task]


def manager_agent(llm) -> Agent:
    """The manager of the hierarchical variant (Step 13): it delegates, it has no tools of its own."""
    return Agent(
        role="Desk Manager",
        goal="Get the research note on {ticker} done by the right desk member, with every number from a tool",
        backstory="You hand each task to the best coworker and check their work. You never write numbers "
                  "yourself and never give advice.",
        llm=llm, allow_delegation=True, max_iter=6, verbose=False,
    )


def build_desk(llm=None, ledger: Ledger | None = None, process: str = "sequential", manager_llm=None,
               llm_for=None, verbose: bool = True) -> Crew:
    """The desk. process="hierarchical" adds a manager (Step 13). Pass llm=FakeCrewAILLM(...) in tests."""
    ledger = ledger if ledger is not None else Ledger()
    if llm_for is None:
        llm_for = (lambda role: llm) if llm is not None else (lambda role: get_crewai_llm())
    agents = make_agents(llm_for, ledger, verbose=verbose)
    settings = dict(agents=list(agents.values()), tasks=make_tasks(agents), memory=False,
                    max_rpm=10, tool_failure_policy="raise", verbose=verbose)
    if process == "hierarchical":
        return Crew(process=Process.hierarchical, manager_agent=manager_agent(manager_llm), **settings)
    return Crew(process=Process.sequential, **settings)


# ---------------------------------------------------------------------------
# 2. The report: written by the agents, framed and checked by our code
# ---------------------------------------------------------------------------

def render_report(report: DeskReport, ticker: str, today: str, written_by: str) -> str:
    i = indicators_for(ticker)
    b = backtest_for(ticker)
    company = COMPANIES[ticker][0]
    risks = "\n".join(f"- {risk}" for risk in report.risks)
    rows = [
        ("Period return", f"{i['period_return_pct']}%"), ("Last 20 days", f"{i['return_last_20_days_pct']}%"),
        ("Last close", i["last_close"]), ("20-day average", i["average_20_days"]),
        ("50-day average", i["average_50_days"]), ("Volatility (per year)", f"{i['volatility_pct_per_year']}%"),
        ("Max drawdown", f"{i['max_drawdown_pct']}%"),
        ("Crossover rule (paper)", f"{b['rule_return_pct']}%"),
        ("Buy-and-hold (paper)", f"{b['buy_and_hold_return_pct']}%"),
    ]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    text = f"""# {ticker} research note (invented company: {company})

| | |
|---|---|
| Data | {i['from']} to {i['to']}, {i['trading_days']} trading days, invented sample data |
| Today | {today} (passed in by our code) |
| Written by | {written_by} |

{MODEL_START}
## What moved

{report.what_moved}

## What the numbers mean

{report.what_the_numbers_mean}

## The paper backtest

{report.paper_backtest}

## Risks and limits

{risks}
{MODEL_END}

## The tool numbers (copied by code, not by a model)

| Number | Value |
|---|---|
{table}
"""
    return stamp_disclaimer(text)


def run_desk(ticker: str, today: str, process: str = "sequential", llm=None, manager_llm=None,
             llm_for=None, verbose: bool = True, written_by: str = "", report_name: str | None = None) -> dict:
    """Run the desk once and write the report. Returns what happened (for Steps 12 and 13)."""
    ledger = Ledger()
    ledger.dates.add(today)                       # today's date is a fact we passed in
    crew = build_desk(llm=llm, ledger=ledger, process=process, manager_llm=manager_llm,
                      llm_for=llm_for, verbose=verbose)
    started = time.perf_counter()
    result = crew.kickoff(inputs={"ticker": ticker, "company": COMPANIES[ticker][0], "today": today})
    seconds = time.perf_counter() - started
    report = result.tasks_output[-1].pydantic
    if report is None:
        raise ValueError(f"the Risk Reviewer's answer did not fit DeskReport: {result.raw[:120]!r}")
    markdown = render_report(report, ticker, today, written_by)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{report_name or ticker}.md"
    path.write_text(markdown, encoding="utf-8")
    ledger.save(path.with_suffix(".numbers.json"))
    return {"result": result, "path": path, "seconds": seconds, "ledger": ledger,
            "check": check_report(markdown, ledger.to_dict())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Crew B: an education-only research note on invented prices.")
    parser.add_argument("--ticker", default="DEMO1", choices=sorted(COMPANIES))
    parser.add_argument("--today", default=None, help="today's date as YYYY-MM-DD (default: your computer's date)")
    parser.add_argument("--replay", action="store_true", help="a scripted model: no key, no internet")
    parser.add_argument("--plant", action="store_true", help="with --replay: the Analyst guesses one number")
    parser.add_argument("--quiet", action="store_true", help="hide CrewAI's step-by-step log")
    args = parser.parse_args(argv)
    today = args.today or date.today().isoformat()
    if not PRICES_FILE.is_file():
        print("data/prices.csv not found. Make it first (Step 9):  python make_prices.py")
        return 1

    llm, written_by = None, ""
    if args.replay:
        from replay import ScriptedLLM, desk_sequential_script

        llm = ScriptedLLM(responses=desk_sequential_script(args.ticker, PRICES_FILE, NEWS_FILE, args.plant))
        written_by = "scripted replay (no AI model), sequential crew"
        print("REPLAY: the model is SCRIPTED (replay.py). Its texts say [scripted]; the tools, the numbers,\n"
              "the disclaimer and the claims check are real.\n")
    else:
        try:
            get_crewai_llm()                      # fail early, with the free-key help, if there is no key
        except MissingAPIKeyError as error:
            print(error)
            print("No key yet? See the same run offline, with a scripted model: add --replay")
            return 1
        written_by = f"{get_provider()}/{get_model_name()}, sequential crew"
    print(f"Research desk: {args.ticker} ({COMPANIES[args.ticker][0]})   today: {today}   process: sequential")
    print("Invented sample data, not real companies or prices. Education only.\n")

    try:
        run = run_desk(args.ticker, today, llm=llm, verbose=not args.quiet, written_by=written_by)
    except Exception as error:  # no internet, the free-tier limit, a failed tool, an answer that did not fit
        print(f"\nThe desk stopped with an error: {type(error).__name__}: {error}")
        print("(429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and run again)")
        return 1
    usage = run["result"].token_usage
    print("\n" + "=" * 70)
    print(f"Saved {shown(run['path'])} and {shown(run['path'].with_suffix('.numbers.json'))}")
    print(f"Tool calls: {', '.join(run['ledger'].calls)}")
    print(f"Usage: requests {usage.successful_requests}   total tokens {usage.total_tokens}   "
          f"seconds {run['seconds']:.1f}\n")
    code = print_result(run["check"])
    print("\nRead it:  reports/" + run["path"].name + "   Check it again any time:  python claims_check.py "
          + "reports/" + run["path"].name)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
