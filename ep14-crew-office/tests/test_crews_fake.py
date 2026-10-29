"""
test_crews_fake.py - both crews end to end, OFFLINE, with a SCRIPTED model (replay.py).

Everything else is real: CrewAI's loop and processes, our tools, the files, the context hand-offs,
the typed answers, the scorecards, the disclaimer and the claims checks.
"""

import json
from pathlib import Path

import pytest
from crewai.tools import ToolExecutionFailedError

import compare_process
import desk_crew
import outreach_crew
from inputs_check import check_inputs
from office_helpers import KESAR, CountingLLM, ensure_prices
from replay import ScriptedLLM, outreach_script
from testing.fakes import FakeCrewAILLM, react_tool_use
from tools.tone import WordListSentiment

TODAY = "2026-10-23"
WORDS = WordListSentiment()          # fast and the same everywhere; the real model has its own test


@pytest.fixture
def drafts(tmp_path, monkeypatch):
    monkeypatch.setattr(outreach_crew, "PENDING", tmp_path / "pending")
    return tmp_path / "pending"


@pytest.fixture
def reports(tmp_path, monkeypatch):
    ensure_prices()
    monkeypatch.setattr(desk_crew, "REPORTS", tmp_path)
    return tmp_path


def replay_llm(leads, stop_after="reviewer", plant=False) -> ScriptedLLM:
    script = []
    for lead in leads:
        script += outreach_script(lead, stop_after, WORDS, plant=plant)
    return ScriptedLLM(responses=script)


# ---------------------------------------------------------------------------
# Crew A: outreach
# ---------------------------------------------------------------------------

def test_one_lead_end_to_end_saves_a_pending_draft(drafts, capsys):
    code = outreach_crew.run([KESAR], llm=replay_llm([KESAR]), sentiment=WORDS, today=TODAY, verbose=False)

    out = capsys.readouterr().out
    assert code == 0, out
    assert "items traced to a line ....... 6 of 6" in out
    assert "Reviewer says ................ pass; our checks say pass" in out
    draft = json.loads((drafts / "kesar-crumb-bakehouse.json").read_text(encoding="utf-8"))
    assert draft["to_email"] == "nandini.apte@example.com"
    assert draft["status"] == "pending: waiting for a human"
    assert draft["verdict"]["status"] == "pass"


def test_the_writer_sees_the_brief_and_the_reviewer_sees_the_draft(drafts):
    llm = replay_llm([KESAR])
    outreach_crew.run([KESAR], llm=llm, sentiment=WORDS, today=TODAY, verbose=False)

    writer_prompt, reviewer_prompt = str(llm.received[3]), str(llm.received[4])
    assert "leads/kesar-crumb-bakehouse.md:13" in writer_prompt      # context=[research_task]
    assert "Sign off as: Neha Rao, partnerships, Pickup Parrot" in writer_prompt   # {sender} filled in
    assert "Pick-up messages for Kesar Crumb" in reviewer_prompt     # the draft reached the Reviewer
    assert "Today is 2026-10-23" in str(llm.received[0])             # {today} filled in


def test_a_made_up_number_is_caught_and_the_draft_waits_with_a_fix(drafts, capsys):
    code = outreach_crew.run([KESAR], llm=replay_llm([KESAR], plant=True), sentiment=WORDS, today=TODAY,
                             verbose=False)

    out = capsys.readouterr().out
    assert code == 3
    assert "numbers outside the brief .... 1 ['40%']" in out
    draft = json.loads((drafts / "kesar-crumb-bakehouse.json").read_text(encoding="utf-8"))
    assert draft["verdict"]["status"] == "fix"
    assert draft["verdict"]["numbers_outside_brief"] == ["40%"]


def test_kickoff_for_each_writes_one_draft_per_lead(drafts, capsys):
    leads = outreach_crew.LEADS
    code = outreach_crew.run(leads, llm=replay_llm(leads), sentiment=WORDS, today=TODAY, verbose=False)

    assert code == 0, capsys.readouterr().out
    assert sorted(p.stem for p in drafts.glob("*.json")) == [lead.stem for lead in leads]


def test_each_lead_gets_its_own_page_in_the_case_file(drafts):
    from notebook import CaseFile

    case = CaseFile()
    crew = outreach_crew.build_crew(llm=replay_llm(outreach_crew.LEADS), case=case, sentiment=WORDS, verbose=False)
    crew.kickoff_for_each(inputs=[outreach_crew.lead_inputs(f, TODAY) for f in outreach_crew.LEADS])

    assert list(case.runs) == [lead.stem for lead in outreach_crew.LEADS]
    for run in case.runs.values():
        assert run["tool_calls"] == ["read_lead_file", "read_lead_file", "check_tone", "check_against_brief"]
        assert run["brief"] is not None


def test_kickoff_for_each_reports_a_running_total_so_we_subtract(drafts, monkeypatch):
    # Pretend usage: 110 tokens per call, 7 calls per lead (Researcher 3, Writer 1, Reviewer 3).
    leads = outreach_crew.LEADS
    inputs = [outreach_crew.lead_inputs(f, TODAY) for f in leads]
    scripts = [outreach_script(f, "reviewer", WORDS) for f in leads]

    # 1. ONE model object shared by three agents: CrewAI counts it once per agent (Episode 13) ...
    shared = CountingLLM(responses=sum(scripts, []))
    results = outreach_crew.build_crew(llm=shared, sentiment=WORDS, verbose=False).kickoff_for_each(inputs=inputs)
    assert [r.token_usage.successful_requests for r in results] == [21, 42, 63]

    # 2. ... so each agent gets its own (as in a real run). Still a RUNNING TOTAL per lead: 7, 14, 21.
    own = iter([CountingLLM(responses=sum((s[0:3] for s in scripts), [])),       # Researcher
                CountingLLM(responses=sum((s[3:4] for s in scripts), [])),       # Writer
                CountingLLM(responses=sum((s[4:7] for s in scripts), []))])      # Reviewer
    monkeypatch.setattr(outreach_crew, "get_crewai_llm", lambda: next(own))
    results = outreach_crew.build_crew(sentiment=WORDS, verbose=False).kickoff_for_each(inputs=inputs)
    assert [r.token_usage.successful_requests for r in results] == [7, 14, 21]

    # 3. delta_since() gives each lead its own numbers.
    per_lead = [results[0].token_usage] + [b.token_usage.delta_since(a.token_usage)
                                           for a, b in zip(results, results[1:])]
    assert [u.successful_requests for u in per_lead] == [7, 7, 7]
    assert [u.total_tokens for u in per_lead] == [770, 770, 770]


def test_the_scorecard_catches_a_made_up_line_and_a_made_up_number():
    from models import LeadBrief
    from notebook import CaseFile
    from replay import scripted_brief
    from tools.files import lead_file_tool

    case = CaseFile()
    lead_file_tool(case).run(path=KESAR.as_posix())
    lead_file_tool(case).run(path="product.md")
    brief = scripted_brief(KESAR)
    assert outreach_crew.check_brief(LeadBrief(**brief), case) == []
    brief["facts"][1] = {"id": "F2", "text": "Busiest week: about 400 gift-box pre-orders",
                         "source": "leads/kesar-crumb-bakehouse.md:12"}
    brief["facts"][2] = {"id": "F3", "text": "Opened in 2026", "source": "leads/kesar-crumb-bakehouse.md:99"}
    assert outreach_crew.check_brief(LeadBrief(**brief), case) == [
        "F2: ['400'] not on line leads/kesar-crumb-bakehouse.md:12",
        "F3: source 'leads/kesar-crumb-bakehouse.md:99' is not a line the file tool read",
    ]


def test_the_researcher_cannot_read_your_env_file(drafts):
    llm = ScriptedLLM(responses=[react_tool_use("read_lead_file", {"path": "../.env"})])
    crew = outreach_crew.build_crew(llm=llm, stop_after="researcher", verbose=False)

    with pytest.raises(ToolExecutionFailedError, match="outside this project folder"):
        crew.kickoff(inputs=outreach_crew.lead_inputs(KESAR, TODAY))


def test_the_inputs_match_the_placeholders():
    crew = outreach_crew.build_crew(llm=ScriptedLLM(responses=[]), verbose=False)
    assert check_inputs(crew, outreach_crew.lead_inputs(KESAR, TODAY)) == ([], [])
    bad = {"company": "X", "lead_file": "leads/x.md", "budget": 5}
    assert check_inputs(crew, bad) == (["sender", "today"], ["budget"])


def test_each_agent_gets_its_own_llm_object(monkeypatch):
    monkeypatch.setattr(outreach_crew, "get_crewai_llm", lambda: FakeCrewAILLM(responses=[]))
    monkeypatch.setattr(desk_crew, "get_crewai_llm", lambda: FakeCrewAILLM(responses=[]))
    assert len({id(a.llm) for a in outreach_crew.build_crew(verbose=False).agents}) == 3
    assert len({id(a.llm) for a in desk_crew.build_desk(verbose=False).agents}) == 3


def test_without_a_key_every_crew_prints_the_free_key_help(monkeypatch, capsys):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    ensure_prices()

    assert outreach_crew.run([KESAR], verbose=False) == 1
    assert desk_crew.main(["--quiet"]) == 1
    assert compare_process.main([]) == 1
    assert capsys.readouterr().out.count("https://aistudio.google.com/apikey") == 3


def test_excessive_agency_replay_shows_the_unapproved_send():
    import excessive_agency

    excessive_agency.WOULD_HAVE_SENT.clear()
    excessive_agency.part_one()
    assert len(excessive_agency.WOULD_HAVE_SENT) == 1
    assert excessive_agency.WOULD_HAVE_SENT[0]["to"] == "nandini@kesarcrumb.example"   # not the real address


# ---------------------------------------------------------------------------
# Crew B: the research desk
# ---------------------------------------------------------------------------

def test_the_desk_writes_a_checked_report_with_the_disclaimer(reports):
    from replay import desk_sequential_script

    llm = ScriptedLLM(responses=desk_sequential_script("DEMO1", desk_crew.PRICES_FILE, desk_crew.NEWS_FILE))
    run = desk_crew.run_desk("DEMO1", TODAY, llm=llm, verbose=False, written_by="scripted replay")

    text = (reports / "DEMO1.md").read_text(encoding="utf-8")
    assert text.startswith("# DEMO1 research note (invented company: Invented Solar Pumps Co.)\n\n> **Education only.")
    assert run["check"]["unsupported_numbers"] == [] and run["check"]["disclaimer_present"]
    assert run["check"]["advice_phrases"] == [] and run["check"]["unsupported_dates"] == []
    assert run["ledger"].calls == ["price_indicators", "read_data_file", "paper_backtest"]
    assert json.loads((reports / "DEMO1.numbers.json").read_text(encoding="utf-8"))["dates"][-1] >= TODAY
    assert "Today is 2026-10-23" in str(llm.received[0])                 # the date was passed in


def test_a_guessed_number_in_the_desk_report_is_caught(reports):
    from replay import desk_sequential_script

    llm = ScriptedLLM(responses=desk_sequential_script("DEMO1", desk_crew.PRICES_FILE, desk_crew.NEWS_FILE, plant=True))
    run = desk_crew.run_desk("DEMO1", TODAY, llm=llm, verbose=False)
    assert run["check"]["unsupported_numbers"] == ["18.5%"]


def test_hierarchical_adds_a_manager_and_its_calls(reports):
    sequential = compare_process.measure("sequential", "DEMO1", TODAY, replay=True, verbose=False)
    hierarchical = compare_process.measure("hierarchical", "DEMO1", TODAY, replay=True, verbose=False)

    assert (sequential["llm_calls"], sequential["tool_calls"], sequential["delegations"]) == (6, 3, 0)
    assert (hierarchical["llm_calls"], hierarchical["tool_calls"], hierarchical["delegations"]) == (12, 6, 3)
    assert sequential["quality"] == hierarchical["quality"] == "yes: 0 unsupported, disclaimer"


def test_max_iter_caps_a_delegation_ping_pong(capsys):
    assert compare_process.pingpong_replay(max_iter=3) == 0
    out = capsys.readouterr().out
    assert "delegations made:        3" in out
    assert "manager model calls:     4" in out
    assert "CrewAI forced a final answer" in out


def test_the_desk_has_no_trading_tool():
    crew = desk_crew.build_desk(llm=FakeCrewAILLM(responses=[]), verbose=False)
    names = {t.name for agent in crew.agents for t in agent.tools}
    assert names == {"price_indicators", "read_data_file", "calculator", "paper_backtest"}


# ---------------------------------------------------------------------------
# Crew memory (Step 8): runs only if the local MiniLM model is already on this computer
# ---------------------------------------------------------------------------

MINILM = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"


@pytest.mark.skipif(not MINILM.is_file(), reason="run python memory_demo.py once with internet")
def test_shared_memory_carries_one_leads_facts_into_another_run(capsys):
    import memory_demo

    assert memory_demo.main() == 0
    out = capsys.readouterr().out
    assert "<- ANOTHER lead's fact" in out
    scoped = out.split("3. The same search inside one lead's scope:")[1]
    assert "Diwali" not in scoped and "samples" not in scoped
