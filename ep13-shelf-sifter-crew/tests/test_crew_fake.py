"""
test_crew_fake.py - Step 13: the whole crew, offline, with a SCRIPTED model.

FakeCrewAILLM (testing/fakes.py) replies with the lines we write, in order:
  react_tool_use(name, args)  -> "use this tool with these arguments"
  react_final_answer(text)    -> "I am done, here is my answer"
Everything else is real: CrewAI's loop, our tools, the context hand-offs, Picks, the guide file.
"""

import json
import os

import pytest
from crewai import Agent, Crew, Task
from crewai.tools import ToolExecutionFailedError

import broken_tool
import scout
from crew import PRACTICE_NOTE, build_crew
from models import Picks
from shelf_helpers import SALT_AND_SNOW, SLOW_TRAIN, TRAVEL_URL, handmade_fetch, word_embed
from testing.fakes import FakeCrewAILLM, react_final_answer, react_tool_use
from tools import Shelf

# CrewAI keeps the task outputs of the latest run in a small file in your user data folder, inside a
# folder named after the current folder. These tests run in temporary folders, so give them ONE fixed name.
os.environ.setdefault("CREWAI_STORAGE_DIR", "ep13-shelf-sifter-crew-tests")

INPUTS = {"category": "travel", "interest": "a long journey by train", "shelf_url": TRAVEL_URL}
SLOW_TRAIN_URL = "https://books.toscrape.com/catalogue/the-slow-train-to-kanyakumari_901/index.html"
SALT_URL = "https://books.toscrape.com/catalogue/salt-and-snow_902/index.html"

PICKS_JSON = json.dumps({
    "category": "travel",
    "interest": "a long journey by train",
    "picks": [
        {"title": SLOW_TRAIN, "price": 12.5, "rating": 4, "url": SLOW_TRAIN_URL,
         "reason": "[scripted] A train journey, four stars, the lowest price."},
        {"title": SALT_AND_SNOW, "price": 38.0, "rating": 5, "url": SALT_URL,
         "reason": "[scripted] Five stars for a long walk instead of a train."},
    ],
})

GUIDE = f"""# Best-value travel books

## [{SLOW_TRAIN}]({SLOW_TRAIN_URL})
£12.50, 4 stars. [scripted] A train journey, four stars, the lowest price.

## [{SALT_AND_SNOW}]({SALT_URL})
£38.00, 5 stars. [scripted] Five stars for a long walk instead of a train.

{PRACTICE_NOTE}"""


def scripted_full_run() -> list[str]:
    """The four model replies of one full run: Scout 1, Analyst 2, Writer 1."""
    return [
        react_tool_use("scrape_shelf", {"url": TRAVEL_URL}),                          # Scout
        react_tool_use("find_similar_books", {"query": "a long journey by train"}),   # Analyst, lap 1
        react_final_answer(PICKS_JSON),                                               # Analyst, lap 2
        react_final_answer(GUIDE),                                                    # Writer
    ]


def test_one_agent_crew_makes_one_successful_tool_call():
    llm = FakeCrewAILLM(responses=[react_tool_use("scrape_shelf", {"url": TRAVEL_URL})])
    shelf = Shelf()

    result = build_crew(llm=llm, shelf=shelf, stop_after="scout", fetch=handmade_fetch,
                        verbose=False).kickoff(inputs=INPUTS)

    assert json.loads(result.raw) == shelf.books           # the answer IS the tool's JSON
    assert len(shelf.books) == 3
    assert shelf.successful_calls == ["scrape_shelf"]
    assert len(llm.received) == 1                          # result_as_answer: no second model call
    assert not result.has_tool_failures


def test_full_crew_runs_offline_with_a_scripted_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                            # the guide is written to reports/ here
    llm = FakeCrewAILLM(responses=scripted_full_run())
    shelf = Shelf()

    result = build_crew(llm=llm, shelf=shelf, fetch=handmade_fetch, embed=word_embed,
                        verbose=False).kickoff(inputs=INPUTS)

    picks = result.tasks_output[1].pydantic
    assert isinstance(picks, Picks)                        # typed, not text
    assert [pick.title for pick in picks.picks] == [SLOW_TRAIN, SALT_AND_SNOW]
    assert scout.check_picks(picks, shelf.books) == []     # every number matches the scrape
    assert shelf.successful_calls == ["scrape_shelf", "find_similar_books"]
    guide = (tmp_path / "reports" / "travel_guide.md").read_text(encoding="utf-8")
    assert guide == GUIDE
    assert len(llm.received) == 4                          # every scripted reply was used


def test_context_hands_each_answer_to_the_next_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm = FakeCrewAILLM(responses=scripted_full_run())

    build_crew(llm=llm, fetch=handmade_fetch, embed=word_embed, verbose=False).kickoff(inputs=INPUTS)

    analyst_prompt = str(llm.received[1])
    writer_prompt = str(llm.received[3])
    assert "Coffee Stops of the Old Silk Road" in analyst_prompt   # the Scout's JSON reached the Analyst
    assert "Five stars for a long walk" in writer_prompt          # the picks reached the Writer
    assert "a long journey by train" in analyst_prompt            # {interest} was filled in


def test_each_agent_gets_its_own_llm_object(monkeypatch):
    # CrewAI adds up the usage of each agent's LLM object: a shared object would be counted 3 times.
    import crew

    monkeypatch.setattr(crew, "get_crewai_llm", lambda: FakeCrewAILLM(responses=[]))

    agents = build_crew(fetch=handmade_fetch, verbose=False).agents
    shared = build_crew(llm=FakeCrewAILLM(responses=[]), fetch=handmade_fetch, verbose=False).agents

    assert len({id(agent.llm) for agent in agents}) == 3       # the real run: one object per agent
    assert len({id(agent.llm) for agent in shared}) == 1       # a test's scripted fake is shared


def test_crew_stops_loudly_when_the_scrape_fails():
    def no_internet(url):
        raise ConnectionError("no internet")

    llm = FakeCrewAILLM(responses=scripted_full_run())

    with pytest.raises(ToolExecutionFailedError, match="no internet"):
        build_crew(llm=llm, fetch=no_internet, embed=word_embed, verbose=False).kickoff(inputs=INPUTS)
    assert len(llm.received) == 1          # the Analyst and the Writer never started


def test_without_raise_a_failed_tool_still_gives_a_confident_answer():
    # The trap from older tutorials: every tool call fails, yet the crew "succeeds".
    wrong_call = react_tool_use("scrape_shelf", {"url": TRAVEL_URL})   # text for a tool that wants a number
    llm = FakeCrewAILLM(responses=[wrong_call, wrong_call, react_final_answer("[scripted] A list I made up.")])
    agent = Agent(role="Scout", goal="List the books", backstory="Eager.", llm=llm,
                  tools=[broken_tool.broken_scrape_shelf], max_iter=2)
    task = Task(description="List the books on the travel shelf.", expected_output="A list", agent=agent)

    result = Crew(agents=[agent], tasks=[task]).kickoff()     # default tool_failure_policy

    assert result.raw == "[scripted] A list I made up."
    assert result.has_tool_failures                           # only this flag tells the truth


def test_scout_run_prints_the_scorecard(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = scout.run(llm=FakeCrewAILLM(responses=scripted_full_run()), fetch=handmade_fetch,
                     embed=word_embed, verbose=False)

    out = capsys.readouterr().out
    assert code == 0
    lines = {line.split(" .")[0].strip(): line.rsplit(". ", 1)[-1] for line in out.splitlines() if " .." in line}
    assert lines["scrape_shelf"].startswith("1 successful call(s), 3 books")
    assert lines["find_similar_books"] == "1 successful call(s)"
    assert lines["tool failures"] == "0"
    assert lines["picks match the shelf"] == "2 of 2"
    assert lines["pick links in the guide"] == "2 of 2"
    assert "requests 0" in out            # a scripted model reports no tokens; a real one does
    assert "All checks passed." in out


def test_scout_run_flags_a_made_up_price(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    replies = scripted_full_run()
    replies[2] = replies[2].replace('"price": 12.5', '"price": 9.5')    # the Analyst "rounds" a price

    code = scout.run(llm=FakeCrewAILLM(responses=replies), fetch=handmade_fetch, embed=word_embed, verbose=False)

    out = capsys.readouterr().out
    assert code == 3
    assert "PROBLEM:" in out and "price is 9.5, the shelf says 12.5" in out


def test_scout_without_a_key_prints_the_free_key_help(monkeypatch, capsys):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    code = scout.run(verbose=False)

    assert code == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out


def test_broken_tool_replay_shows_the_error_then_a_clean_run(capsys):
    assert broken_tool.replay() == 0
    out = capsys.readouterr().out
    assert "Input should be a valid integer" in out
    assert "Clean run: one tool call, no errors" in out
