"""
compare_process.py - Step 13: sequential versus hierarchical, MEASURED, not guessed.

  sequential    an assembly line: task 1, then task 2, then task 3 (what we teach and use)
  hierarchical  a supervisor: a manager agent hands out each task ("Delegate work to coworker"),
                reads the answer, and decides what happens next

A manager adds control, and it adds model calls, tokens and seconds. So we run the same desk
twice and count. Two lessons from older tutorials are built in:
  - the manager model is chosen on purpose, at least as strong as the workers
    (Gemini: gemini-3.8-flash; set MANAGER_MODEL in .env to change it), with no temperature
    setting (Gemini 3.6 Flash and newer ignore it),
  - only the manager may delegate (workers have allow_delegation=False), and max_iter and
    max_rpm cap every agent, so no task can bounce between agents forever.

    python compare_process.py --ticker DEMO1           (two live runs, needs your key)
    python compare_process.py --ticker DEMO1 --replay  (scripted models: shows the mechanism, not real costs)

Status (2026-10-26): CrewAI's hierarchical process with a real Gemini manager has NOT been
tested in this course yet. The replay runs CrewAI's real hierarchical code with scripted models.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import argparse
import os
from collections import Counter
from datetime import date

from crewai import LLM
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

from desk_crew import NEWS_FILE, run_desk
from llm import DEFAULT_OLLAMA_URL, MissingAPIKeyError, check_api_key, get_crewai_llm, get_model_name, get_provider
from make_prices import COMPANIES
from tools.indicators import PRICES_FILE

MANAGER_MODELS = {"gemini": "gemini-3.8-flash"}     # other providers: the same model as the workers


def manager_model_name() -> str:
    provider = get_provider()
    model = os.getenv("MANAGER_MODEL", "").strip() or MANAGER_MODELS.get(provider) or get_model_name(provider)
    return f"{provider}/{model}"


def make_manager_llm() -> LLM:
    """An explicit manager model (never CrewAI's default, never a weaker one than the workers)."""
    provider = get_provider()
    check_api_key(provider)
    if provider == "ollama":
        return LLM(model=manager_model_name(), base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL))
    return LLM(model=manager_model_name())


def measure(process: str, ticker: str, today: str, replay: bool, verbose: bool) -> dict:
    """Run the desk once with `process` and count what it cost."""
    scripted = []
    if replay:
        from replay import ScriptedLLM, desk_hierarchical_scripts, desk_sequential_script

        if process == "sequential":
            shared = ScriptedLLM(responses=desk_sequential_script(ticker, PRICES_FILE, NEWS_FILE))
            scripted, llm_for, manager = [shared], (lambda role: shared), None
        else:
            scripts = {role: ScriptedLLM(responses=r)
                       for role, r in desk_hierarchical_scripts(ticker, PRICES_FILE, NEWS_FILE).items()}
            scripted, llm_for, manager = list(scripts.values()), (lambda role: scripts[role]), scripts["manager"]
        manager_name = "scripted" if process == "hierarchical" else "-"
    else:
        llm_for = lambda role: get_crewai_llm()      # one model object per agent  # noqa: E731
        manager = make_manager_llm() if process == "hierarchical" else None
        manager_name = manager_model_name() if process == "hierarchical" else "-"

    tool_calls: Counter = Counter()
    with crewai_event_bus.scoped_handlers():          # listen to CrewAI's own events during this run

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def count(source, event):
            tool_calls[event.tool_name] += 1

        run = run_desk(ticker, today, process=process, llm_for=llm_for, manager_llm=manager,
                       verbose=verbose, written_by=f"{process} crew" + (" (scripted replay)" if replay else ""),
                       report_name=f"{ticker}_{process}")
        crewai_event_bus.flush()                      # wait until every event has been handled

    usage = run["result"].token_usage
    check = run["check"]
    good = not check["unsupported_numbers"] and not check["advice_phrases"] and check["disclaimer_present"]
    return {
        "process": process,
        "manager": manager_name,
        "llm_calls": sum(len(s.received) for s in scripted) if replay else usage.successful_requests,
        "tool_calls": sum(tool_calls.values()),
        "delegations": sum(n for name, n in tool_calls.items() if "coworker" in name.lower()),
        "tokens": "n/a (scripted)" if replay else usage.total_tokens,
        "seconds": round(run["seconds"], 1),
        "quality": (f"yes: {len(check['unsupported_numbers'])} unsupported, disclaimer" if good
                    else f"NO: {len(check['unsupported_numbers'])} unsupported, "
                         f"{len(check['advice_phrases'])} advice"),
        "report": run["path"].name,
    }


def pingpong_replay(max_iter: int = 3) -> int:
    """Delegation ping-pong, capped: a scripted manager keeps handing the same task back."""
    from crewai import Agent, Crew, Process, Task

    from replay import ScriptedLLM
    from testing.fakes import react_final_answer, react_tool_use

    def bounce(round_number: int) -> str:
        return react_tool_use("Delegate work to coworker",
                              {"task": "Write the risk section for DEMO1",
                               "context": f"[scripted] Not good enough, redo it (round {round_number}).",
                               "coworker": "Risk Reviewer"}, thought="[scripted] Send it back again")

    final = react_final_answer("[scripted] Here is the risk section I have.")
    manager_llm = ScriptedLLM(responses=[bounce(n) for n in range(1, 11)] + [final])
    worker_llm = ScriptedLLM(responses=[react_final_answer("[scripted] Here is the risk section again.")] * 10)
    worker = Agent(role="Risk Reviewer", goal="Write risks", backstory="Careful.", llm=worker_llm,
                   allow_delegation=False, verbose=False)
    manager = Agent(role="Desk Manager", goal="Get good work", backstory="Never satisfied.", llm=manager_llm,
                    allow_delegation=True, max_iter=max_iter, verbose=False)
    task = Task(description="Write the risk section for DEMO1.", expected_output="A short risk section")
    with crewai_event_bus.scoped_handlers():
        delegations: Counter = Counter()

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def count(source, event):
            delegations[event.tool_name] += 1

        result = Crew(agents=[worker], tasks=[task], process=Process.hierarchical, manager_agent=manager).kickoff()
        crewai_event_bus.flush()
    print(f"Delegation ping-pong (scripted manager that is never satisfied), manager max_iter={max_iter}:")
    print(f"  delegations made:        {sum(delegations.values())}")
    print(f"  manager model calls:     {len(manager_llm.received)}")
    print(f"  worker model calls:      {len(worker_llm.received)}")
    forced = "MUST give your absolute best final answer" in str(manager_llm.received[-1])
    print(f"  last manager call:       {'CrewAI forced a final answer (the cap was reached)' if forced else 'normal'}")
    print(f"  final answer starts:     {result.raw.splitlines()[0]!r}")
    print("  The cap stopped the bill, not the problem: the stubborn manager still asked for a tool, and")
    print("  CrewAI used that text as the answer. Read the final answer; a cap is not a quality check.")
    print("  Without a cap, CrewAI's default max_iter is 25 laps, and every lap costs model calls.")
    return 0


def print_table(rows: list[dict]) -> None:
    headers = [("process", "Process"), ("manager", "Manager model"), ("llm_calls", "LLM calls"),
               ("tool_calls", "Tool calls"), ("delegations", "Delegations"), ("tokens", "Tokens"),
               ("seconds", "Seconds"), ("quality", "Same report quality?")]
    widths = {k: max(len(title), *(len(str(r[k])) for r in rows)) for k, title in headers}
    print("  ".join(title.ljust(widths[k]) for k, title in headers).rstrip())
    print("  ".join("-" * widths[k] for k, _ in headers))
    for row in rows:
        print("  ".join(str(row[k]).ljust(widths[k]) for k, _ in headers).rstrip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the research desk sequentially and hierarchically, and compare.")
    parser.add_argument("--ticker", default="DEMO1", choices=sorted(COMPANIES))
    parser.add_argument("--today", default=None, help="today's date as YYYY-MM-DD (default: your computer's date)")
    parser.add_argument("--replay", action="store_true", help="scripted models: the mechanism, not real costs")
    parser.add_argument("--verbose", action="store_true", help="show CrewAI's step-by-step log")
    parser.add_argument("--pingpong", action="store_true", help="scripted: a manager that keeps delegating, capped")
    args = parser.parse_args(argv)
    if args.pingpong:
        return pingpong_replay()
    today = args.today or date.today().isoformat()
    if not PRICES_FILE.is_file():
        print("data/prices.csv not found. Make it first (Step 9):  python make_prices.py")
        return 1
    if not args.replay:
        try:
            make_manager_llm()
        except MissingAPIKeyError as error:
            print(error)
            print("No key yet? See the mechanism offline:  python compare_process.py --replay")
            return 1

    print(f"Research desk on {args.ticker}, run twice   today: {today}   "
          f"{'SCRIPTED models (replay.py)' if args.replay else 'live models'}\n")
    rows = []
    for process in ("sequential", "hierarchical"):
        print(f"Running the {process} desk ...")
        try:
            rows.append(measure(process, args.ticker, today, args.replay, args.verbose))
        except Exception as error:  # the free-tier limit, a delegation loop that hit max_iter, ...
            print(f"  the {process} run stopped: {type(error).__name__}: {error}")
            return 1
    print()
    print_table(rows)
    print(f"\nReports: reports/{rows[0]['report']} and reports/{rows[1]['report']}")
    if args.replay:
        print("Scripted runs: the counts show HOW a manager adds calls (one delegation and one answer per")
        print("task), not what Gemini would really do. Tokens and seconds only mean something in a live run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
