"""
replay.py - the SCRIPTED model behind every --replay run (no key, no internet).

With --replay, our programs run the real CrewAI, the real tools, the real files and every
real check, but the "model" is FakeCrewAILLM from testing/fakes.py: it answers with lines WE
wrote here, in order. That lets you see a whole run before you have a key, and it gives
the tests a run that is the same every time.

Rules we keep, so nobody mistakes a replay for AI output:
  - every text a scripted agent writes is marked [scripted]
  - the scripted agents only quote what the tools and files really return (we compute those
    values with the same functions first), like a well-behaved model would
  - --plant adds ONE made-up number on purpose, so you can watch the checks catch it

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import json
from pathlib import Path

from testing.fakes import FakeCrewAILLM, react_final_answer, react_tool_use


class ScriptedLLM(FakeCrewAILLM):
    """A FakeCrewAILLM that kickoff_for_each cannot clone.

    CrewAI copies every agent's model for each run of kickoff_for_each. A copied script would
    start again at reply 1 for every lead, so every copy of this one IS the same object.
    """

    def __copy__(self):
        return self


# ---------------------------------------------------------------------------
# Crew A: the outreach office
# ---------------------------------------------------------------------------

def numbered_line(path: Path, prefix: str) -> tuple[int, str]:
    """(line number, text after '- Label: ') of the first line that starts with '- <prefix>'."""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith(f"- {prefix}"):
            return number, line[2:].strip()
    raise ValueError(f"{path} has no line starting with '- {prefix}'")


def fact(path: Path, prefix: str, fact_id: str) -> dict:
    number, text = numbered_line(path, prefix)
    return {"id": fact_id, "text": text, "source": f"{path.as_posix()}:{number}"}


# What each scripted Researcher picks, and the email each scripted Writer writes.
LEAD_SCRIPTS = {
    "kesar-crumb-bakehouse": {
        "facts": ["Recent news", "How pre-orders", "Problem"],
        "points": ["What it does", "Set-up", "Free trial"],
        "facts_used": ["F1", "F2", "P1", "P2", "P3"],
        "subject": "[scripted] Pick-up messages for Kesar Crumb's festival boxes",
        "body": ("Dear Nandini,\n\n"
                 "Congratulations on opening the Viman Nagar outlet in August 2026. When festival orders "
                 "come in by phone and go into a paper register, keeping every pick-up time clear is a big job.\n\n"
                 "Pickup Parrot sends each customer a short message when their order is ready, and keeps one "
                 "shared list of pre-orders. Set-up takes about 15 minutes, and the free trial runs for 30 days "
                 "with no card needed.\n\n"
                 "Would it help to try it at one outlet before Diwali?\n\n"
                 "Warm regards,\nNeha Rao\nPartnerships, Pickup Parrot"),
        "plant": "Bakeries that use Pickup Parrot see 40% fewer missed pick-ups. ",
    },
    "kayal-learning-circle": {
        "facts": ["Recent news", "How reminders", "Problem"],
        "points": ["Messages", "Set-up", "Free trial"],
        "facts_used": ["F1", "F2", "P1", "P2", "P3"],
        "subject": "[scripted] Class-change messages for Kayal Learning Circle",
        "body": ("Dear Arun,\n\n"
                 "Congratulations on starting the weekend online batches in June 2026. Sending each class "
                 "change to parents one by one from your own phone takes a lot of time.\n\n"
                 "Pickup Parrot can send every parent a short message when a class moves, by SMS or email, "
                 "in English or Malayalam. Set-up takes about 15 minutes, and the free trial runs for 30 days "
                 "with no card needed.\n\n"
                 "Would a trial with one batch be useful?\n\n"
                 "Best wishes,\nNeha Rao\nPartnerships, Pickup Parrot"),
        "plant": "Centres like yours save 6 hours a week. ",
    },
    "dhoop-chhaon-textiles": {
        "facts": ["Recent news", "How sample requests", "Problem"],
        "points": ["What it does", "Set-up", "Free trial"],
        "facts_used": ["F1", "F2", "P1", "P2", "P3"],
        "subject": "[scripted] Sample updates for Dhoop Chhaon's buyers",
        "body": ("Dear Ms Sheikh,\n\n"
                 "Congratulations on shipping a first order to a buyer in Portugal in September 2026. When "
                 "buyers email for fabric samples, a short note that a sample is on its way can answer the "
                 "question before they ask it.\n\n"
                 "Pickup Parrot sends that note as soon as a sample is ready, and keeps one shared list of "
                 "every request. Set-up takes about 15 minutes, and the free trial runs for 30 days with no "
                 "card needed.\n\n"
                 "Would you like to try it for your next sample requests?\n\n"
                 "Kind regards,\nNeha Rao\nPartnerships, Pickup Parrot"),
        "plant": "Exporters who use it win 25% more repeat orders. ",
    },
}


def scripted_brief(lead_file: Path) -> dict:
    from outreach_crew import PRODUCT_FILE, field, lead_inputs

    script = LEAD_SCRIPTS[lead_file.stem]
    return {
        "company": lead_inputs(lead_file, "")["company"],
        "contact_name": field(lead_file, "Contact").split(",")[0].strip(),
        "facts": [fact(lead_file, p, f"F{i}") for i, p in enumerate(script["facts"], start=1)],
        "product_points": [fact(PRODUCT_FILE, p, f"P{i}") for i, p in enumerate(script["points"], start=1)],
    }


def scripted_draft(lead_file: Path, plant: bool = False) -> dict:
    script = LEAD_SCRIPTS[lead_file.stem]
    body = script["body"]
    if plant:      # one made-up number, placed before the question
        question = body.rfind("\n\nWould")
        body = body[:question] + "\n\n" + script["plant"].strip() + body[question:]
    return {"subject": script["subject"], "body": body, "facts_used": script["facts_used"]}


def scripted_verdict(draft: dict, brief: dict, sentiment) -> dict:
    """What a careful Reviewer reports: exactly what the two tools REALLY say about this draft."""
    from models import LeadBrief
    from tools.brief_check import check_against_brief
    from tools.tone import tone_report

    tone = tone_report(draft["body"], sentiment)
    found = check_against_brief(f"{draft['subject']}\n{draft['body']}", LeadBrief(**brief))
    fixes = list(tone["reasons"])
    fixes += [f"remove {n}: it is not in the brief" for n in found["numbers_outside_brief"]]
    fixes += [f"remove {link}: our code adds the recipient" for link in found["links_or_addresses"]]
    ok = tone["verdict"] == "ok" and found["ok"]
    return {
        "status": "pass" if ok else "fix",
        "tone": f"[scripted] {tone['label']} {tone['scores'][tone['label']]}, {tone['verdict']}",
        "numbers_outside_brief": found["numbers_outside_brief"],
        "fixes": [f"[scripted] {f}" for f in fixes],
    }


def outreach_script(lead_file: Path, stop_after: str, sentiment, plant: bool = False) -> list[str]:
    """The scripted replies of one lead: Researcher 3, Writer 1, Reviewer 3."""
    brief = scripted_brief(lead_file)
    replies = [
        react_tool_use("read_lead_file", {"path": lead_file.as_posix()}),
        react_tool_use("read_lead_file", {"path": "product.md"}),
        react_final_answer(json.dumps(brief)),
    ]
    if stop_after == "researcher":
        return replies
    draft = scripted_draft(lead_file, plant)
    replies.append(react_final_answer(json.dumps(draft)))
    if stop_after == "writer":
        return replies
    replies += [
        react_tool_use("check_tone", {"text": draft["body"]}),
        react_tool_use("check_against_brief", {"text": f"{draft['subject']}\n{draft['body']}"}),
        react_final_answer(json.dumps(scripted_verdict(draft, brief, sentiment))),
    ]
    return replies


# ---------------------------------------------------------------------------
# Crew B: the research desk (education only)
# ---------------------------------------------------------------------------

def moved(value: float) -> str:
    return f"up {abs(value)}%" if value >= 0 else f"down {abs(value)}%"


def headline_on(day: str, ticker: str, news_file: Path) -> str | None:
    for line in news_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{day},{ticker},"):
            return line.split(",", 2)[2].removeprefix("Invented headline: ")
    return None


def desk_texts(ticker: str, prices_file: Path, news_file: Path, plant: bool = False) -> dict:
    """The scripted answers of the desk, built from what the tools REALLY return."""
    from tools.backtest import backtest_for
    from tools.indicators import indicators_for

    i = indicators_for(ticker, prices_file)
    b = backtest_for(ticker, prices_file)
    down, up = i["biggest_down_day"], i["biggest_up_day"]
    down_news = headline_on(down["date"], ticker, news_file)
    up_news = headline_on(up["date"], ticker, news_file)
    volatility = 18.5 if plant else i["volatility_pct_per_year"]      # --plant: a guessed number

    what_moved = (
        f"[scripted] {ticker} closed at {i['last_close']} on {i['to']}, {moved(i['period_return_pct'])} "
        f"from {i['first_close']} on {i['from']}. Its biggest one-day fall was {down['change_pct']}% on "
        f"{down['date']}" + (f", the day of an invented headline: '{down_news}'." if down_news else ".") +
        f" Its biggest one-day rise was {up['change_pct']}% on {up['date']}" +
        (f", the day of '{up_news}'." if up_news else ".") +
        f" Over the last 20 days it moved {moved(i['return_last_20_days_pct'])}."
    )
    numbers_mean = (
        f"[scripted] The 20-day average close is {i['average_20_days']} and the 50-day average is "
        f"{i['average_50_days']}: when the short average is above the long one, recent prices have been "
        f"higher than older ones. Volatility of {volatility}% a year measures how much the price jumps "
        f"around from day to day; bigger means bumpier. The max drawdown of {i['max_drawdown_pct']}% is the "
        f"worst fall from a high ({i['max_drawdown_from_high_on']}) to a later low ({i['max_drawdown_low_on']})."
    )
    backtest = (
        f"[scripted] On paper, from {b['test_from']} to {b['test_to']} ({b['test_days']} days, no trading "
        f"costs), the 20/50-day crossover rule returned {b['rule_return_pct']}% and buy-and-hold returned "
        f"{b['buy_and_hold_return_pct']}%, a difference of {b['difference_pct_points']} percentage points. "
        f"The rule was in the stock on {b['days_in_the_stock']} of {b['test_days']} days and switched in or "
        f"out {b['switches_in_or_out']} times. One invented price series says nothing about any real stock."
    )
    report = {
        "ticker": ticker,
        "what_moved": what_moved,
        "what_the_numbers_mean": numbers_mean,
        "paper_backtest": backtest,
        "risks": [
            "[scripted] The prices and headlines are invented sample data: no real company moved like this.",
            "[scripted] A rule that did better or worse on one invented year can do the opposite on another.",
            "[scripted] The paper test leaves out trading costs and taxes, which would lower a real result.",
            f"[scripted] Even this series fell {abs(i['max_drawdown_pct'])}% from a high at one point.",
        ],
    }
    return {"analyst": what_moved + " " + numbers_mean, "strategy": backtest, "report": json.dumps(report)}


def desk_sequential_script(ticker: str, prices_file: Path, news_file: Path, plant: bool = False) -> list[str]:
    """Analyst 3 replies, Strategy Explainer 2, Risk Reviewer 1."""
    t = desk_texts(ticker, prices_file, news_file, plant)
    return [
        react_tool_use("price_indicators", {"ticker": ticker}),
        react_tool_use("read_data_file", {"path": "data/news.csv"}),
        react_final_answer(t["analyst"]),
        react_tool_use("paper_backtest", {"ticker": ticker}),
        react_final_answer(t["strategy"]),
        react_final_answer(t["report"]),
    ]


def desk_hierarchical_scripts(ticker: str, prices_file: Path, news_file: Path) -> dict[str, list[str]]:
    """One script per agent: the manager delegates each task, the workers do them."""
    t = desk_texts(ticker, prices_file, news_file)

    def delegate(coworker: str, task: str, context: str) -> str:
        return react_tool_use("Delegate work to coworker", {"task": task, "context": context, "coworker": coworker},
                              thought="I will hand this task to the right coworker")

    return {
        "manager": [
            delegate("Market Analyst", f"Explain what moved {ticker} using price_indicators and read_data_file.",
                     "[scripted] Quote every number exactly as the tools give it."),
            react_final_answer(t["analyst"]),
            delegate("Strategy Explainer", f"Explain the paper backtest of {ticker}.",
                     "[scripted] Use paper_backtest once. Education only."),
            react_final_answer(t["strategy"]),
            delegate("Risk Reviewer", f"Write the DeskReport for {ticker}.",
                     "[scripted] " + t["analyst"] + " " + t["strategy"]),
            react_final_answer(t["report"]),
        ],
        "Market Analyst": desk_sequential_script(ticker, prices_file, news_file)[:3],
        "Strategy Explainer": desk_sequential_script(ticker, prices_file, news_file)[3:5],
        "Risk Reviewer": [react_final_answer(t["report"])],
    }

