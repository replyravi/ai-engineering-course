"""
outreach_crew.py - Steps 4, 5, 6 and 8: Crew A writes email DRAFTS. It cannot send anything.

Three agents, one after the other (Process.sequential):

  Researcher  reads the lead file and product.md (read_lead_file)    -> a typed LeadBrief
  Writer      writes from the brief only (context=[research_task])   -> a typed EmailDraft
  Reviewer    checks tone and claims (check_tone, check_against_brief) -> a typed Verdict

Then OUR code checks the agents (the scorecard), adds the recipient from the lead file and
the opt-out line, and saves the draft in drafts/pending/. A human decides in approve.py (Step 7).
No tool here can send an email, and no code in this project can either (tests/test_approve.py).

    python outreach_crew.py --stop-after researcher       Step 4: the brief
    python outreach_crew.py --stop-after writer           Step 5: + the draft
    python outreach_crew.py                               Step 6: + the verdict, saves a pending draft
    python outreach_crew.py --all                         Step 8: every lead (kickoff_for_each)
    python outreach_crew.py --lead kayal-learning-circle  another lead
    add --replay to any of them: a SCRIPTED model (no key, no internet); --plant adds a made-up number

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import argparse
import json
import re
import warnings
from datetime import date, datetime
from pathlib import Path

from crewai import Agent, Crew, Process, Task

from approve import OPT_OUT, shown
from claims_check import as_value, find_numbers, unsupported_numbers
from inputs_check import check_inputs, placeholders_in
from llm import MissingAPIKeyError, get_crewai_llm, get_model_name, get_provider
from models import EmailDraft, LeadBrief, Verdict
from notebook import CaseFile
from tools.brief_check import CheckAgainstBriefTool, check_against_brief
from tools.files import lead_file_tool
from tools.tone import CheckToneTool, LocalSentimentModel, tone_report

# CrewAI warns that callbacks on methods "will prevent checkpointing". We never use checkpointing.
warnings.filterwarnings("ignore", message=".*callbacks cannot be serialized.*")

HERE = Path(__file__).resolve().parent
LEADS = sorted(Path("leads").glob("*.md"))          # relative paths: they appear in the agents' text
PRODUCT_FILE = Path("product.md")
PENDING = HERE / "drafts" / "pending"
STAGES = ["researcher", "writer", "reviewer"]


# ---------------------------------------------------------------------------
# 1. Inputs: built by code from the files (the model never types an address)
# ---------------------------------------------------------------------------

def field(path: Path, label: str) -> str:
    """The text after '- Label:' in a lead or product file."""
    match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise ValueError(f"{path} has no line '- {label}: ...'")
    return match.group(1).strip()


def lead_inputs(lead_file: Path, today: str) -> dict:
    """The kickoff inputs for one lead: exactly the placeholders our tasks use."""
    company = lead_file.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    return {
        "lead_file": lead_file.as_posix(),
        "company": company,
        "sender": field(PRODUCT_FILE, "Sender of our outreach emails"),
        "today": today,
    }


def recipient(lead_file: Path) -> tuple[str, str]:
    """(name, email address) from the lead file. Copied by code, never written by a model."""
    return field(lead_file, "Contact").split(",")[0].strip(), field(lead_file, "Email")


# ---------------------------------------------------------------------------
# 2. The crew
# ---------------------------------------------------------------------------

def make_agents(llm, case: CaseFile, sentiment=None, verbose: bool = True) -> dict:
    def model():
        # One crewai.LLM object per agent (Episode 13: a shared object is counted once per agent).
        return llm if llm is not None else get_crewai_llm()

    researcher = Agent(
        role="Lead Researcher",
        goal="Collect three true facts about {company} from its lead file, each with the line it came from",
        backstory="You only write what the files say. You never guess, and you never add facts "
                  "from your own knowledge. Every fact carries its exact source line.",
        tools=[lead_file_tool(case)],
        llm=model(), max_iter=4, allow_delegation=False, verbose=verbose,
    )
    writer = Agent(
        role="Outreach Writer",
        goal="Write a short, warm first email to {company} that uses only facts from the brief",
        backstory="You write like a polite neighbour, not a salesperson: no pressure, no hype, "
                  "no promise the brief does not support.",
        tools=[],                       # the Writer only writes
        llm=model(), max_iter=2, allow_delegation=False, verbose=verbose,
    )
    reviewer = Agent(
        role="Email Reviewer",
        goal="Check the tone of the draft for {company} and that every claim in it comes from the brief",
        backstory="You are the last check before a human reads the draft. You trust what the "
                  "tools report, not your impression, and you never rewrite the email yourself.",
        tools=[CheckToneTool(sentiment=sentiment, case=case), CheckAgainstBriefTool(case=case)],
        llm=model(), max_iter=4, allow_delegation=False, verbose=verbose,
    )
    return {"researcher": researcher, "writer": writer, "reviewer": reviewer}


def make_tasks(agents: dict, case: CaseFile) -> dict:
    research_task = Task(
        description="Read the lead file {lead_file} with read_lead_file, then read product.md.\n"
                    "Write a brief about {company} for a first email about Pickup Parrot:\n"
                    "- exactly 3 facts about {company}: a recent event, how they work today, and a "
                    "problem they mentioned;\n"
                    "- 1 to 3 points from product.md that fit this lead.\n"
                    "Each item has an id (F1, F2, F3 for facts; P1, P2, P3 for product points), the "
                    "fact in plain words from ONE line, and its source as '<file>:<line number>' exactly "
                    "as the tool shows it. Never add anything the files do not say. Today is {today}.",
        expected_output="A LeadBrief: company, contact_name, 3 facts and 1 to 3 product points, "
                        "each with id, text and source.",
        agent=agents["researcher"],
        output_pydantic=LeadBrief,      # CrewAI turns the answer into a LeadBrief object
        callback=case.save_brief,       # our code keeps the brief for check_against_brief
    )
    write_task = Task(
        description="Write a first email from Pickup Parrot to the contact person of {company}, "
                    "using ONLY the brief.\n"
                    "1. 120 words or fewer, plain text, no exclamation marks.\n"
                    "2. Use one or two lead facts and at least one product point, and list the ids "
                    "you used in facts_used.\n"
                    "3. Every number you write must appear in the brief.\n"
                    "4. No pressure: no deadlines, no 'act now'. End with one simple question.\n"
                    "5. Sign off as: {sender}\n"
                    "6. Do not write an email address, a link or an unsubscribe line: our code adds "
                    "the recipient and the opt-out line.\n"
                    "Today is {today}; do not invent dates.",
        expected_output="An EmailDraft: subject, body and facts_used.",
        agent=agents["writer"],
        context=[research_task],        # the brief is pasted into this task's prompt: the hand-off
        output_pydantic=EmailDraft,
    )
    review_task = Task(
        description="Review the email draft for {company} before a human sees it.\n"
                    "1. Call check_tone once with the draft body.\n"
                    "2. Call check_against_brief once with the subject and the body together.\n"
                    "3. status is 'pass' only if check_tone's verdict is 'ok' AND check_against_brief "
                    "says ok is true; otherwise 'fix'.\n"
                    "4. tone: the label and verdict from check_tone. numbers_outside_brief: copied from "
                    "check_against_brief. fixes: one short line per problem the tools found (empty "
                    "when status is pass). Do not rewrite the email.",
        expected_output="A Verdict: status, tone, numbers_outside_brief and fixes.",
        agent=agents["reviewer"],
        context=[research_task, write_task],
        output_pydantic=Verdict,
    )
    return {"researcher": research_task, "writer": write_task, "reviewer": review_task}


def build_crew(llm=None, case: CaseFile | None = None, stop_after: str = "reviewer",
               sentiment=None, verbose: bool = True) -> Crew:
    """The outreach crew. stop_after="researcher" or "writer" runs only the first agents."""
    case = case if case is not None else CaseFile()
    agents = make_agents(llm, case, sentiment=sentiment, verbose=verbose)
    tasks = make_tasks(agents, case)
    keep = STAGES[: STAGES.index(stop_after) + 1]
    return Crew(
        agents=[agents[name] for name in keep],
        tasks=[tasks[name] for name in keep],
        process=Process.sequential,
        memory=False,                      # each lead is its own case: no memory shared between leads (Step 8)
        before_kickoff_callbacks=[case.start],   # our notebook opens a new page for every lead
        max_rpm=10,                        # at most 10 model requests a minute (protects your free quota)
        tool_failure_policy="raise",       # a failed tool call stops the crew instead of letting it guess
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# 3. Our scorecard: check the agents' work against what the tools REALLY saw
# ---------------------------------------------------------------------------

def check_brief(brief: LeadBrief, case: CaseFile) -> list[str]:
    """Every fact must point at a line the file tool really read, and its numbers must be on that line."""
    problems = []
    for fact in brief.facts + brief.product_points:
        line = case.line_at(fact.source)
        if line is None:
            problems.append(f"{fact.id}: source {fact.source!r} is not a line the file tool read")
            continue
        on_the_line = [as_value(n)[0] for n in find_numbers(line)]
        extra = unsupported_numbers(fact.text, on_the_line)     # a number the line does not have
        if extra:
            problems.append(f"{fact.id}: {extra} not on line {fact.source}")
    return problems


def check_draft(draft: EmailDraft, brief: LeadBrief) -> list[str]:
    problems = []
    found = check_against_brief(f"{draft.subject}\n{draft.body}", brief)
    if found["numbers_outside_brief"]:
        problems.append(f"numbers not in the brief: {found['numbers_outside_brief']}")
    if found["links_or_addresses"]:
        problems.append(f"links or addresses in the draft: {found['links_or_addresses']}")
    ids = {fact.id for fact in brief.facts + brief.product_points}
    unknown = [i for i in draft.facts_used if i not in ids]
    if unknown:
        problems.append(f"facts_used names ids that are not in the brief: {unknown}")
    return problems


def print_usage(usage, label: str = "Usage") -> None:
    print(f"{label}: requests {usage.successful_requests}   prompt tokens {usage.prompt_tokens}   "
          f"completion tokens {usage.completion_tokens}   reasoning tokens {usage.reasoning_tokens}   "
          f"total {usage.total_tokens}")


def report_lead(result, lead_file: Path, case: CaseFile, stop_after: str, model_name: str,
                sentiment=None, row: dict | None = None) -> int:
    """Print the typed answers and the scorecard of one lead; save the pending draft. Returns problems.

    row (optional) is filled with this lead's summary line for the end of a multi-lead run."""
    lead = lead_file.stem
    row = row if row is not None else {}
    row.update(lead=lead, traced="-", outside="-", reviewer="-", draft="-")
    brief = result.tasks_output[0].pydantic
    problems = 0
    if brief is None:
        print(f"PROBLEM: the Researcher's answer did not fit LeadBrief. It starts: {result.tasks_output[0].raw[:120]!r}")
        return 1
    print(f"The Researcher's typed answer: LeadBrief with {len(brief.facts)} facts and "
          f"{len(brief.product_points)} product point(s)")
    print(brief.model_dump_json(indent=2))
    brief_problems = check_brief(brief, case)
    traced = len(brief.facts) + len(brief.product_points) - len(brief_problems)
    print(f"\nScorecard ({lead})")
    print(f"  {'read_lead_file ':.<30} {case.runs[lead]['tool_calls'].count('read_lead_file')} call(s)")
    print(f"  {'items traced to a line ':.<30} {traced} of {len(brief.facts) + len(brief.product_points)}")
    row["traced"] = f"{traced} of {len(brief.facts) + len(brief.product_points)}"
    for problem in brief_problems:
        print(f"  PROBLEM: {problem}")
    problems += len(brief_problems)
    if stop_after == "researcher":
        return problems

    draft = result.tasks_output[1].pydantic
    if draft is None:
        print(f"PROBLEM: the Writer's answer did not fit EmailDraft. It starts: {result.tasks_output[1].raw[:120]!r}")
        return problems + 1
    outside = check_against_brief(f"{draft.subject}\n{draft.body}", brief)["numbers_outside_brief"]
    draft_problems = check_draft(draft, brief)
    print(f"  {'numbers outside the brief ':.<30} {len(outside)} {outside or ''}".rstrip())
    row["outside"] = str(len(outside))
    for problem in draft_problems:
        print(f"  PROBLEM: {problem}")
    problems += len(draft_problems)
    print(f"\nThe Writer's typed answer: EmailDraft (facts used: {', '.join(draft.facts_used)})")
    print(f"Subject: {draft.subject}\n\n{draft.body}\n")
    if stop_after == "writer":
        return problems

    verdict = result.tasks_output[2].pydantic
    if verdict is None:
        print(f"PROBLEM: the Reviewer's answer did not fit Verdict. It starts: {result.tasks_output[2].raw[:120]!r}")
        return problems + 1
    print(f"The Reviewer's typed answer: {verdict.model_dump_json(indent=2)}")
    calls = case.runs[lead]["tool_calls"]
    print(f"  {'check_tone ':.<30} {calls.count('check_tone')} call(s)")
    print(f"  {'check_against_brief ':.<30} {calls.count('check_against_brief')} call(s)")
    our_status = "pass"
    if draft_problems:
        our_status = "fix"
    if sentiment is not None and tone_report(draft.body, sentiment)["verdict"] != "ok":
        our_status = "fix"
    agree = verdict.status == our_status
    print(f"  {'Reviewer says ':.<30} {verdict.status}; our checks say {our_status}"
          f"{'' if agree else '   PROBLEM: they disagree'}")
    problems += 0 if agree else 1
    row["reviewer"] = verdict.status
    if calls.count("check_tone") == 0 or calls.count("check_against_brief") == 0:
        print("  PROBLEM: the Reviewer gave a verdict without calling both tools")
        problems += 1
    path = save_pending(lead_file, brief, draft, verdict, model_name)
    row["draft"] = shown(path)
    print(f"\nSaved {shown(path)} (a DRAFT, waiting for a human: nothing was sent)")
    return problems


def save_pending(lead_file: Path, brief: LeadBrief, draft: EmailDraft, verdict: Verdict, model_name: str) -> Path:
    """Write the draft for approve.py. The recipient and the opt-out line are added HERE, by code."""
    name, email = recipient(lead_file)
    PENDING.mkdir(parents=True, exist_ok=True)
    path = PENDING / f"{lead_file.stem}.json"
    path.write_text(json.dumps({
        "lead_file": lead_file.as_posix(),
        "to_name": name,
        "to_email": email,                       # from the lead file, not from the model
        "subject": draft.subject,
        "body": f"{draft.body.rstrip()}\n\n{OPT_OUT}",
        "facts_used": draft.facts_used,
        "brief": brief.model_dump(),
        "verdict": verdict.model_dump(),
        "written_by": model_name,
        "made_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pending: waiting for a human",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 4. Run one lead, or every lead
# ---------------------------------------------------------------------------

def load_sentiment(replay: bool):
    """The tone model. For a replay without the downloaded model, the word list stands in."""
    try:
        return LocalSentimentModel(download=not replay)
    except Exception:
        if not replay:
            raise
        from tools.tone import WordListSentiment
        print("(The sentiment model is not downloaded, so this replay uses the word-list fallback.)")
        return WordListSentiment()


def run(lead_files: list[Path], stop_after: str = "reviewer", llm=None, sentiment=None,
        today: str | None = None, verbose: bool = True, model_name: str | None = None) -> int:
    today = today or date.today().isoformat()
    case = CaseFile()
    try:
        crew = build_crew(llm=llm, case=case, stop_after=stop_after, sentiment=sentiment, verbose=verbose)
    except MissingAPIKeyError as error:
        print(error)                       # the friendly "get your free key" help
        print("No key yet? See the same run offline, with a scripted model: add --replay")
        return 1
    model_name = model_name or getattr(llm, "model", None) or f"{get_provider()}/{get_model_name()}"
    needed = placeholders_in(crew)          # --stop-after researcher has no Writer, so it needs no {sender}
    inputs = [{k: v for k, v in lead_inputs(f, today).items() if k in needed} for f in lead_files]
    for one in inputs:
        missing, unused = check_inputs(crew, one)
        if missing or unused:
            print(f"Inputs do not match the task text: missing={missing} unused={unused}")
            return 1
    team = ", ".join(STAGES[: STAGES.index(stop_after) + 1])
    print(f"Outreach crew: {len(lead_files)} lead(s)   agents: {team}   today: {today}")
    print(f"Model: {model_name}   Process: {crew.process.value}   memory: off   max_rpm: {crew.max_rpm}\n")

    try:
        if len(inputs) == 1:
            results = [crew.kickoff(inputs=inputs[0])]
        else:
            results = crew.kickoff_for_each(inputs=inputs)   # one run per lead, same crew
    except Exception as error:  # no internet, the free-tier limit, a failed tool ...
        print(f"\nThe crew stopped with an error: {type(error).__name__}: {error}")
        print("(429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and run again)")
        return 1

    if sentiment is None:      # a live run: use the model the Reviewer's check_tone tool loaded
        sentiment = next((t.sentiment for a in crew.agents for t in a.tools if t.name == "check_tone"), None)
    problems, previous, rows = 0, None, []
    for lead_file, result in zip(lead_files, results):
        print("\n" + "=" * 70)
        rows.append({})
        problems += report_lead(result, lead_file, case, stop_after, model_name, sentiment, rows[-1])
        # CrewAI 1.15.28 gives each kickoff_for_each run a running TOTAL (we measured 1, 2, 3 requests
        # for three one-request runs), so we subtract the previous total: CrewAI's own delta_since().
        usage = result.token_usage if previous is None else result.token_usage.delta_since(previous)
        previous = result.token_usage
        rows[-1]["requests"], rows[-1]["tokens"] = usage.successful_requests, usage.total_tokens
        print_usage(usage, f"Usage for {lead_file.stem}")
    if len(results) > 1:
        print(f"\nSummary ({len(rows)} leads, one kickoff_for_each)")
        print(f"  {'lead':<24} {'traced':<8} {'outside':<8} {'reviewer':<9} {'requests':<9} {'tokens':<7} draft")
        for r in rows:
            print(f"  {r['lead']:<24} {r['traced']:<8} {r['outside']:<8} {r['reviewer']:<9} "
                  f"{r['requests']:<9} {r['tokens']:<7} {r['draft']}")
        print_usage(results[-1].token_usage, "Usage for all leads")
    if model_name.startswith("scripted"):
        print("(A scripted model reports no tokens, so the usage is 0. A live run shows real numbers here.)")
    if problems:
        print(f"\n{problems} problem(s) found: read the scorecard before you approve anything.")
        return 3
    if stop_after != "reviewer":           # Steps 4 and 5: no Reviewer yet, so no draft was saved
        print(f"\nAll checks passed. No draft was saved: this run stopped after the {stop_after}.")
        return 0
    print("\nAll checks passed. Next: a human reads each draft:  python approve.py")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Crew A: approval-gated email DRAFTS for invented leads.")
    parser.add_argument("--lead", default="kesar-crumb-bakehouse", help="the lead file name, without .md")
    parser.add_argument("--all", action="store_true", help="every lead in leads/ (kickoff_for_each)")
    parser.add_argument("--stop-after", default="reviewer", choices=STAGES, help="run only the first agents")
    parser.add_argument("--replay", action="store_true", help="a scripted model: no key, no internet")
    parser.add_argument("--plant", action="store_true", help="with --replay: the Writer adds a made-up number")
    parser.add_argument("--today", default=None, help="today's date as YYYY-MM-DD (default: your computer's date)")
    parser.add_argument("--quiet", action="store_true", help="hide CrewAI's step-by-step log")
    args = parser.parse_args(argv)

    lead_files = LEADS if args.all else [Path("leads") / f"{args.lead}.md"]
    if not all(f.is_file() for f in lead_files):
        print(f"No such lead file. Pick one of: {', '.join(f.stem for f in LEADS)}")
        return 1
    if not args.replay:
        return run(lead_files, args.stop_after, today=args.today, verbose=not args.quiet)

    from replay import ScriptedLLM, outreach_script
    sentiment = load_sentiment(True)
    script = []
    for f in lead_files:
        script += outreach_script(f, args.stop_after, sentiment, plant=args.plant)
    print("REPLAY: the model is SCRIPTED (replay.py). The texts marked [scripted] were written by us;\n"
          "CrewAI, the tools, the files and every check are real.\n")
    return run(lead_files, args.stop_after, llm=ScriptedLLM(responses=script), sentiment=sentiment,
               today=args.today, verbose=not args.quiet, model_name="scripted replay (no AI model)")


if __name__ == "__main__":
    raise SystemExit(main())
