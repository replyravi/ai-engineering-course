"""
tests/test_calculator.py - the calculator tool, the agent that uses it (Step 12)
and the LLM-versus-Python experiment (Step 11).

The tool is real Python. The agent's model is SCRIPTED: it first "asks" for the
calculator (tool_call), then answers. The experiment's model replies are scripted too,
one of them wrong on purpose, so we can check that the mismatch is flagged.
"""

import time
from decimal import Decimal

import pytest

import calculator
import math_experiment
from calculator import (AGENT_RULES, MAX_AGENT_STEPS, QUESTION, CalculatorError, ask_agent, calculate,
                        format_number, print_trace)
from costing import SAMPLE_ITEMS, load_prices, scale_and_cost
from party_fixtures import no_keys  # noqa: F401
from testing.fakes import fake_llm, tool_call


# ---------------------------------------------------------------------------
# calculate() and the tool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression, expected", [
    ("12 * 154", "1848"),
    ("938.40 / 12 * 17", "1329.4"),
    ("48,750 * 18 / 100", "8775"),
    ("1,234,567 + 1", "1234568"),
    ("10 + 5 * 2", "20"),
    ("(10 + 5) * 2", "30"),
    ("73986 / 17", "4352.117647"),
    ("2**10", "1024"),
    ("7 // 2", "3"),
    ("-2**2", "-4"),
    ("0.1 + 0.2", "0.3"),
])
def test_the_tool_computes(expression, expected):
    assert calculator.calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize("expression, reason", [
    ("__import__('os').system('ls')", "only numbers"),            # code, not maths
    ("().__class__", "only numbers"),
    ("open('secret.txt').read()", "only numbers"),
    ("sqrt(16)", "only numbers"),                                  # no functions at all
    ("18% of 500", "write a percentage as / 100"),
    ("2^10", "write powers with"),
    ("1 / 0", "division by zero"),
    ("(-8)**0.5", "not a real number"),
    ("99999999 * 99999999", "more than 15 digits"),
    ("1.2.3", "not a complete calculation"),
    ("", "empty"),
    ("1+" * 150, "longer than 200 characters"),
])
def test_unsafe_or_broken_expressions_are_refused(expression, reason):
    with pytest.raises(CalculatorError, match=reason):
        calculate(expression)
    assert calculator.calculator.invoke({"expression": expression}).startswith("Error: ")


def test_a_giant_power_is_refused_quickly():
    # 9**9**9 as whole numbers would make Python build a number with millions of digits.
    start = time.perf_counter()
    assert calculator.calculator.invoke({"expression": "9**9**9"}) == "Error: the result is too big."
    assert time.perf_counter() - start < 1


def test_our_code_never_calls_eval():
    # Read calculator.py as Python code (not as text) and look for a call to eval().
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(calculator))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "eval" not in called and "exec" not in called


def test_format_number():
    assert format_number(1848.0) == "1848"
    assert format_number(4352.117647058823) == "4352.117647"
    assert format_number(-0.0) == "0"


def test_what_the_model_sees():
    assert calculator.calculator.name == "calculator"
    assert list(calculator.calculator.args) == ["expression"]
    assert "Never do maths in your head" in calculator.calculator.description


def test_tool_only_runs_without_a_key(no_keys, capsys):
    assert calculator.main(["--tool-only"]) == 0
    output = capsys.readouterr().out
    assert 'calculator("938.40 / 12 * 17")' in output and "-> 1329.4" in output
    assert "-> Error: the result is too big." in output


def test_the_question_uses_the_step_6_bill():
    total = scale_and_cost(SAMPLE_ITEMS, 12, load_prices()).total
    assert total == Decimal("938.40")
    assert f"Rs {total}" in QUESTION


# ---------------------------------------------------------------------------
# The agent with a scripted model (the old LLMMathChain, the 2026 way)
# ---------------------------------------------------------------------------

def scripted_agent_model():
    return fake_llm(
        tool_call("calculator", {"expression": "938.40 / 12 * 17"}),   # model call 1: use the tool
        "With 17 guests the new total is Rs 1,329.40.",                # model call 2: the answer
    )


def test_the_loop_is_human_ai_tool_ai():
    messages = ask_agent(QUESTION, scripted_agent_model())
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert messages[2].text == "1329.4"                             # our real Python ran
    assert messages[3].text == "With 17 guests the new total is Rs 1,329.40."


def test_the_model_reads_the_tool_result_before_answering():
    llm = scripted_agent_model()
    ask_agent(QUESTION, llm)
    first_call, second_call = llm.received
    assert first_call[0].type == "system" and first_call[0].text == AGENT_RULES
    assert second_call[-1].type == "tool" and second_call[-1].text == "1329.4"


def test_a_bad_expression_goes_back_to_the_model_as_text():
    llm = fake_llm(tool_call("calculator", {"expression": "938.40 / 12 x 17"}),
                   tool_call("calculator", {"expression": "938.40 / 12 * 17"}),
                   "Rs 1,329.40")
    messages = ask_agent(QUESTION, llm)
    tool_texts = [m.text for m in messages if m.type == "tool"]
    assert tool_texts[0].startswith("Error: only numbers")
    assert tool_texts[1] == "1329.4"


def test_the_trace_names_every_message(capsys):
    print_trace(ask_agent(QUESTION, scripted_agent_model()))
    output = capsys.readouterr().out
    assert "Message trace (4 messages):" in output
    assert '  2. ai     asks for a tool: calculator({"expression": "938.40 / 12 * 17"})' in output
    assert "  3. tool   calculator -> 1329.4" in output
    assert "Tool calls: 1   Model calls: 2" in output


def test_a_model_that_never_stops_is_stopped(capsys):
    llm = fake_llm(*[tool_call("calculator", {"expression": "1 + 1"}) for _ in range(20)])
    assert calculator.main([], llm=llm) == 1
    assert "The agent stopped" in capsys.readouterr().out
    assert len(llm.received) < MAX_AGENT_STEPS


def test_agent_without_a_key_shows_where_to_get_one(no_keys, capsys):
    assert calculator.main([]) == 1
    output = capsys.readouterr().out
    assert "https://aistudio.google.com/apikey" in output and "Traceback" not in output


# ---------------------------------------------------------------------------
# Step 11: the model versus Python
# ---------------------------------------------------------------------------

def test_python_answers_are_exact():
    exact = [calculate(q["expression"]) for q in math_experiment.build_questions()]
    assert [format_number(x) for x in exact] == ["938.4", "57525", "11286", "4352.117647", "4386543"]


def test_numbers_are_read_from_messy_replies():
    read = math_experiment.extract_number
    assert read("Rs 57,525.00") == 57525.0
    assert read("The answer is 4352.12") == 4352.12
    assert read("about four thousand") is None


def test_a_wrong_model_answer_is_flagged():
    llm = fake_llm("938.40", "57,525", "11,286", "4352.12", "4,368,543")   # the last one is wrong on purpose
    rows = math_experiment.run_experiment(llm)
    assert [r["result"] for r in rows] == ["match", "match", "match", "match", "OFF by 18000"]
    assert [m[-1].content for m in llm.received][1].startswith("A caterer quotes Rs 48,750")


def test_the_experiment_gives_the_model_no_tools():
    llm = fake_llm("1", "2", "3", "4", "5")
    math_experiment.run_experiment(llm)
    assert all(m[0].type == "system" and "ONLY the final number" in m[0].text for m in llm.received)


def test_experiment_without_a_key_shows_where_to_get_one(no_keys, capsys):
    assert math_experiment.main([]) == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out
