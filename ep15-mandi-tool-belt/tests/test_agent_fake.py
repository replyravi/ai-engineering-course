"""
tests/test_agent_fake.py - the agent wiring, with a SCRIPTED model (no key, no internet).

The model's replies are scripted with tool_call(...) and plain text. Everything else is real:
create_agent's loop, our tools, the temporary database, the caps, the trace, the token meter and ask.py.
"""

import pytest
from langchain_core.messages import AIMessage
from mandi_fixtures import fake_store, no_keys, temp_db  # noqa: F401  (pytest fixtures)

import agent
import ask
import token_meter
from agent import final_answer, print_trace, run_agent, sources, stopped_by_cap, system_prompt_for, tool_calls
from testing.fakes import fake_llm, tool_call
from tools import tool_belt
from tools.calc import calculator


@pytest.fixture
def belt(temp_db, fake_store):
    """The whole belt over a temporary database and a fake-embeddings index."""
    return tool_belt(db_path=temp_db, store=fake_store)


def with_usage(message, given_in, out, thinking=0):
    """Give a scripted message the usage numbers a real model would report."""
    message.usage_metadata = {"input_tokens": given_in, "output_tokens": out, "total_tokens": given_in + out,
                              "output_token_details": {"reasoning": thinking}}
    return message


# ---------------------------------------------------------------------------
# Steps 3 and 4 - one tool, and no tool at all
# ---------------------------------------------------------------------------

def test_fare_question_makes_one_calculator_call(capsys):
    llm = fake_llm(tool_call("calculator", {"expression": "30 + (8.5 - 1.5) * 17 + 20"}),
                   "[SCRIPTED] The fare is Rs 169.")
    messages = run_agent(ask.PRESETS["fare"][1], [calculator], llm)

    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert messages[2].text == "169"                     # our Python did the maths
    print_trace(messages)
    assert "Tool calls: 1 (calculator)" in capsys.readouterr().out


def test_general_question_needs_no_tool(capsys):
    messages = run_agent(ask.PRESETS["general"][1], [calculator], fake_llm("[SCRIPTED] A mandi is a market."))
    assert tool_calls(messages) == []
    print_trace(messages)
    assert "Tool calls: 0 (the model answered by itself)" in capsys.readouterr().out


def test_the_model_is_told_about_its_tools_only(belt):
    calc_only = system_prompt_for([calculator])
    assert "calculator" in calc_only and "run_select" not in calc_only and "search_docs" not in calc_only
    everything = system_prompt_for(belt)
    assert "list_tables, then describe_table, then run_select" in everything
    assert "search_docs" in everything and "invented sample data" in everything


# ---------------------------------------------------------------------------
# Steps 7, 10 and 15 - the SQL walk, SQL then maths, docs + SQL + maths
# ---------------------------------------------------------------------------

def test_sql_walk_discover_describe_query_answer(temp_db):
    llm = fake_llm(tool_call("list_tables", {}),
                   tool_call("describe_table", {"table_name": "daily_prices"}),
                   tool_call("run_select", {"query": "SELECT city, price_rs_per_kg FROM daily_prices "
                                                     "WHERE vegetable = 'onion' AND day = 3 ORDER BY price_rs_per_kg"}),
                   "[SCRIPTED] Indore had the cheapest onions on day 3.")
    messages = run_agent(ask.PRESETS["onions"][1], tool_belt(["sql"], db_path=temp_db), llm)

    assert [name for name, _ in tool_calls(messages)] == ["list_tables", "describe_table", "run_select"]
    tool_texts = [m.text for m in messages if m.type == "tool"]
    assert tool_texts[0] == "cities, daily_prices"
    assert tool_texts[1].startswith("Table daily_prices: 150 rows")
    assert tool_texts[2].splitlines()[3].startswith("Indore")
    assert sources(messages)[0].startswith("SQL used: SELECT city, price_rs_per_kg")


def test_a_wrong_query_comes_back_as_text_and_the_agent_tries_again(temp_db):
    llm = fake_llm(tool_call("run_select", {"query": "SELECT price FROM daily_prices"}),
                   tool_call("run_select", {"query": "SELECT MAX(price_rs_per_kg) FROM daily_prices"}),
                   "[SCRIPTED] Rs 38.1.")
    messages = run_agent("Highest price?", tool_belt(["sql"], db_path=temp_db), llm)
    tool_texts = [m.text for m in messages if m.type == "tool"]
    assert tool_texts[0] == "Error: no such column: price"
    assert "38.1" in tool_texts[1]
    assert final_answer(messages) == "[SCRIPTED] Rs 38.1."


def test_percent_change_is_sql_then_calculator(temp_db):
    llm = fake_llm(tool_call("run_select", {"query": "SELECT day, price_rs_per_kg FROM daily_prices WHERE "
                                                     "vegetable = 'tomato' AND city = 'Pune' AND day IN (1, 10)"}),
                   tool_call("calculator", {"expression": "(36.2 - 29.4) / 29.4 * 100"}),
                   "[SCRIPTED] About 23.13 percent.")
    messages = run_agent(ask.PRESETS["percent"][1], tool_belt(["calc", "sql"], db_path=temp_db), llm)
    assert [name for name, _ in tool_calls(messages)] == ["run_select", "calculator"]
    assert [m.text for m in messages if m.type == "tool"][1] == "23.129252"


def test_docs_sql_and_maths_in_one_question(belt):
    llm = fake_llm(tool_call("search_docs", {"query": "commission market fee loading onion bag"}),
                   tool_call("run_select", {"query": "SELECT price_rs_per_kg FROM daily_prices WHERE city = 'Pune' "
                                                     "AND vegetable = 'onion' AND day = 7"}),
                   tool_call("calculator", {"expression": "3 * 50 * 25.7 * 1.05 + 3 * 12"}),
                   "[SCRIPTED] The bill is Rs 4,083.75 (source: fees.html).")
    messages = run_agent(ask.PRESETS["full"][1], belt, llm)
    assert [name for name, _ in tool_calls(messages)] == ["search_docs", "run_select", "calculator"]
    assert [m.text for m in messages if m.type == "tool"][2] == "4083.75"
    found = sources(messages)
    assert any(s.startswith("SQL used:") for s in found)
    assert any(s.startswith("page: https://replyravi.github.io/") for s in found)


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

def test_a_model_that_never_stops_is_stopped_by_the_call_cap():
    endless = fake_llm(*[tool_call("calculator", {"expression": f"{n} + 1"}) for n in range(30)])
    messages = run_agent("Count forever", [calculator], endless)        # no GraphRecursionError
    assert sum(1 for m in messages if m.type == "ai" and m.tool_calls) == agent.MAX_MODEL_CALLS
    assert stopped_by_cap(messages) == f"Model call limits exceeded: run limit ({agent.MAX_MODEL_CALLS}/" \
                                       f"{agent.MAX_MODEL_CALLS})"


def test_the_token_budget_stops_a_long_run():
    replies = [with_usage(tool_call("calculator", {"expression": "1 + 1"}), 9_000, 1_000) for _ in range(6)]
    messages = run_agent("Keep going", [calculator], fake_llm(*replies), max_tokens=25_000)
    assert sum(1 for m in messages if m.type == "ai" and m.tool_calls) == 3     # 10k, 20k, 30k: then stop
    assert stopped_by_cap(messages) == "Stopped: the token budget of 25,000 was used up (30,000 tokens)."


def test_the_recursion_limit_is_never_reached_before_the_call_cap():
    assert agent.RECURSION_LIMIT > 5 * agent.MAX_MODEL_CALLS + 1


# ---------------------------------------------------------------------------
# Step 14 - the token meter
# ---------------------------------------------------------------------------

def test_meter_adds_up_every_model_call():
    messages = [with_usage(AIMessage(content="", tool_calls=[{"name": "list_tables", "args": {}, "id": "1"}]),
                           800, 40, 25),
                with_usage(AIMessage(content="Indore."), 1_200, 60, 30)]
    totals = token_meter.meter(messages)
    assert totals == {"model_calls": 2, "not_reported": 0, "input": 2000, "output": 100, "thinking": 55,
                      "total": 2100}
    assert token_meter.format_meter(totals) == "Tokens: 2,000 in + 100 out = 2,100 (thinking 55 of the out), 2 model calls"


def test_meter_never_turns_missing_usage_into_zero():
    totals = token_meter.meter([AIMessage(content="hi"), AIMessage(content="again")])
    assert token_meter.format_meter(totals) == "Tokens: not reported by this model (2 model calls)"
    mixed = token_meter.meter([with_usage(AIMessage(content="a"), 10, 5), AIMessage(content="b")])
    assert token_meter.format_meter(mixed).endswith("2 model calls, 1 call(s) reported no usage")


def test_cap_notes_are_not_model_calls():
    note = AIMessage(content="Model call limits exceeded: run limit (8/8)")
    assert token_meter.meter([note])["model_calls"] == 0


def test_no_model_call_at_all_is_not_shown_as_zero_tokens():
    assert token_meter.format_meter(token_meter.meter([])) == "Tokens: none counted (no model call finished)"


def test_saved_run_and_the_per_call_table(tmp_path, capsys):
    messages = run_agent("Fare?", [calculator],
                         fake_llm(with_usage(tool_call("calculator", {"expression": "2 + 2"}), 700, 20, 12),
                                  with_usage(AIMessage(content="[SCRIPTED] 4"), 760, 8)))
    path = token_meter.save_run("Fare?", messages, tmp_path / "last_run.json")
    assert token_meter.main(path) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Last run: Fare?"
    assert out[2].split() == ["1", "calculator", "700", "20", "12", "720"]
    assert out[3].split() == ["2", "final", "answer", "760", "8", "0", "768"]
    assert out[4] == "Tokens: 1,460 in + 28 out = 1,488 (thinking 12 of the out), 2 model calls"


# ---------------------------------------------------------------------------
# ask.py
# ---------------------------------------------------------------------------

def test_ask_prints_trace_answer_sources_and_tokens(temp_db, monkeypatch, tmp_path):
    monkeypatch.setattr("tools.sql.DB_PATH", temp_db)
    monkeypatch.setattr(token_meter, "LAST_RUN", tmp_path / "last_run.json")
    llm = fake_llm(tool_call("run_select", {"query": "SELECT city FROM daily_prices WHERE vegetable = 'onion' "
                                                     "AND day = 3 ORDER BY price_rs_per_kg LIMIT 1"}),
                   "[SCRIPTED] Indore.")
    lines = []
    assert ask.main(["onions"], llm=llm, write=lines.append) == 0
    assert lines[0] == "Question: Which city had the cheapest onions on day 3?"
    assert lines[1] == "Tools: list_tables, describe_table, run_select"
    assert "Answer: [SCRIPTED] Indore." in [line.strip() for line in lines]
    assert any(line.startswith("  SQL used: SELECT city FROM daily_prices") for line in lines)
    assert lines[-1] == "Tokens: not reported by this model (2 model calls)"
    assert (tmp_path / "last_run.json").exists()


def test_list_tools_needs_no_key(no_keys):
    lines = []
    assert ask.main(["--list-tools"], write=lines.append) == 0
    assert lines[0] == "The agent holds 5 tools:"
    assert [line.split()[1].split("(")[0] for line in lines[1:6]] == \
        ["calculator", "list_tables", "describe_table", "run_select", "search_docs"]
    assert lines[6].startswith("Caps: at most 8 model calls and 30,000 tokens per question")


def test_ask_stops_before_the_model_when_the_database_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.sql.DB_PATH", tmp_path / "mandi.db")         # not seeded
    llm = fake_llm("[SCRIPTED] never used")
    lines = []
    assert ask.main(["onions"], llm=llm, write=lines.append) == 1
    assert lines[-1] == "mandi.db not found. Create it first with:  python seed_db.py"
    assert llm.received == []                                              # no model call was made


def test_ask_without_a_key_prints_the_help(no_keys):
    lines = []
    assert ask.main(["fare"], write=lines.append) == 1
    assert lines[:2] == [f"Question: {ask.PRESETS['fare'][1]}", "Tools: calculator"]
    assert "Get a FREE Gemini key" in lines[2]


@pytest.mark.parametrize("typed, tool_text", [
    ("y", "SQL used: SELECT COUNT(*) FROM daily_prices LIMIT 20"),
    ("n", "User rejected the tool call for `run_select` with reason: A human said no to this query."),
])
def test_approve_sql_asks_a_human_first(temp_db, monkeypatch, tmp_path, typed, tool_text):
    monkeypatch.setattr("tools.sql.DB_PATH", temp_db)
    monkeypatch.setattr(token_meter, "LAST_RUN", tmp_path / "last_run.json")
    llm = fake_llm(tool_call("run_select", {"query": "SELECT COUNT(*) FROM daily_prices"}), "[SCRIPTED] Done.")
    asked, lines = [], []
    code = ask.main(["How many rows?", "--tools", "sql", "--approve-sql"], llm=llm,
                    read=lambda prompt: asked.append(prompt) or typed, write=lines.append)
    assert code == 0
    assert asked == ["Run this SQL? [y/N] "]
    assert "The agent wants to run:  SELECT COUNT(*) FROM daily_prices" in [line.strip() for line in lines]
    assert any(tool_text in line for line in lines)
