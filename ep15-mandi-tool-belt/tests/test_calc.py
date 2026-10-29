"""
tests/test_calc.py - the calculator tool (Step 2) and the Episode 11 copy it comes from.
"""

import pytest
from mandi_fixtures import EPISODE, without_header

from tools.calc import PRECEDENCE, REFUSED, calculate, calculator, main, what_the_model_sees

def test_the_copy_matches_episode_11():
    original = EPISODE.parent / "ep11-party-planner-chains" / "calculator.py"
    if not original.exists():
        pytest.skip("Episode 11's folder is not here (this folder was copied on its own)")
    copy = EPISODE / "tools" / "ep11_calculator.py"
    assert without_header(copy) == original.read_text(encoding="utf-8")


@pytest.mark.parametrize("story, with_brackets, without", PRECEDENCE)
def test_brackets_change_the_answer(story, with_brackets, without):
    assert calculate(with_brackets) != calculate(without)


def test_the_two_pairs_give_the_numbers_in_the_readme():
    assert [calculator.invoke({"expression": e}) for _, a, b in PRECEDENCE for e in (a, b)] == \
        ["305", "585", "169", "33"]


@pytest.mark.parametrize("expression", REFUSED)
def test_unsafe_or_odd_expressions_come_back_as_text(expression):
    result = calculator.invoke({"expression": expression})
    assert result.startswith("Error:")          # returned, not raised: the agent keeps running


def test_names_and_functions_never_reach_numexpr():
    # numexpr itself would accept round(2.5); our check refuses every letter first
    assert "only numbers" in calculator.invoke({"expression": "round(2.5)"})
    assert "only numbers" in calculator.invoke({"expression": "__import__('os').system('ls')"})


def test_what_the_model_sees_is_name_description_and_schema():
    card = what_the_model_sees()
    assert card["name"] == "calculator"
    assert "brackets" in card["description"]
    assert card["parameters"]["properties"] == {"expression": {"type": "string"}}
    assert card["parameters"]["required"] == ["expression"]


def test_main_runs_one_expression(capsys):
    assert main(["3 * 50 * 25.7"]) == 0
    assert capsys.readouterr().out.strip() == 'calculator("3 * 50 * 25.7") -> 3855'
