"""
tests/test_agent.py - the agent loop (Steps 10 and 11) with a SCRIPTED model.

The fake model first "asks" for the tool (tool_call), then gives the final answer.
The tool itself is real: our Python runs and its result goes back to the model.
"""

import rung5_tool
from helpdesk_fixtures import fixed_today, no_keys  # noqa: F401
from rung5_tool import AGENT_RULES, ask_agent, print_trace, tools_used
from testing.fakes import fake_llm, tool_call

TOOL_RESULT = "Kitchen deep-clean day is on Tuesday 17 November 2026: 25 days from today (Friday 23 October 2026)."


def scripted_agent_model():
    return fake_llm(
        tool_call("days_until_holiday", {}),                       # model call 1: "run the tool, please"
        "Namaste! The next kitchen holiday is in 25 days.",        # model call 2: the answer
    )


def test_the_loop_is_human_ai_tool_ai(fixed_today):
    messages = ask_agent("How many days until the next kitchen holiday?", scripted_agent_model())
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert messages[1].tool_calls[0]["name"] == "days_until_holiday"
    assert messages[2].text == TOOL_RESULT                          # our real Python ran
    assert messages[3].text == "Namaste! The next kitchen holiday is in 25 days."


def test_the_model_reads_the_tool_result_before_answering(fixed_today):
    llm = scripted_agent_model()
    ask_agent("How many days until the next kitchen holiday?", llm)
    first_call, second_call = llm.received
    assert first_call[0].type == "system" and first_call[0].text == AGENT_RULES
    assert second_call[-1].type == "tool" and second_call[-1].text == TOOL_RESULT


def test_the_trace_names_every_message(fixed_today, capsys):
    messages = ask_agent("How many days until the next kitchen holiday?", scripted_agent_model())
    print_trace(messages)
    output = capsys.readouterr().out
    assert "Message trace (4 messages):" in output
    assert "  1. human  How many days until the next kitchen holiday?" in output
    assert "  2. ai     asks for a tool: days_until_holiday({})" in output
    assert f"  3. tool   days_until_holiday -> {TOOL_RESULT}" in output
    assert "  4. ai     Namaste! The next kitchen holiday is in 25 days." in output
    assert "Tool calls: 1   Model calls: 2" in output


def test_a_question_that_needs_no_tool():
    messages = ask_agent("Say hello.", fake_llm("Namaste!"))
    assert [m.type for m in messages] == ["human", "ai"]
    assert tools_used(messages) == []


def test_main_prints_the_answer_and_the_tools_used(fixed_today, capsys):
    assert rung5_tool.main([], llm=scripted_agent_model()) == 0
    output = capsys.readouterr().out
    assert "Namaste! The next kitchen holiday is in 25 days." in output
    assert "(tools the agent used: days_until_holiday)" in output


def test_a_model_that_never_stops_calling_the_tool_is_stopped(fixed_today, capsys):
    llm = fake_llm(*[tool_call("days_until_holiday", {}) for _ in range(20)])   # never answers
    assert rung5_tool.main([], llm=llm) == 1
    assert "The agent stopped" in capsys.readouterr().out
    assert len(llm.received) < rung5_tool.MAX_AGENT_STEPS        # 5 model calls with the limit of 10, not thousands


def test_main_without_a_key_shows_where_to_get_one(no_keys, capsys):
    assert rung5_tool.main([]) == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out
