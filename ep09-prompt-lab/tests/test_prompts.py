"""
tests/test_prompts.py - the prompt files and the templates they make (Steps 3, 5 and 14).

Rendering a template needs no model, so we can check every message exactly.
"""

import pytest

import prompt_files
from common import PROMPTS_DIR, get_note, load_notes
from models import Expense
from prompt_files import PromptFileError, list_prompts, read_prompt, note_inputs
from settle import same_result
from techniques import parse_transfers

NOTE = get_note("coorg")
# Every prompt file in prompts/, including the ones YOU add (like balance_rtc_v2.toml in Step 14).
ALL_PROMPTS = [p.stem for p in sorted(PROMPTS_DIR.glob("*.toml"))]
# The ten files that come with the lab.
OUR_PROMPTS = ["balance_cot_v1", "balance_cot_v2", "balance_fewshot_v1", "balance_rtc_v1", "balance_zero_v1",
               "expense_fewshot_v1", "expenses_v1", "message_v1", "summary_v1", "summary_v2"]


def fill(prompt, with_examples=True):
    template = prompt.template(with_examples)
    if "note" in template.input_variables:
        return template.format_messages(**note_inputs(NOTE.note))
    return template.format_messages(trip="Coorg weekend", transfers="Kabir owes Meera 740 rupees")


def test_our_ten_prompt_files_are_there():
    # Your own new versions are welcome: they get the checks below too.
    assert len(OUR_PROMPTS) == 10
    assert set(OUR_PROMPTS) <= set(ALL_PROMPTS)


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_every_file_name_matches_its_version(name):
    assert read_prompt(name).version_id == name


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_the_last_message_is_always_human(name):
    # Gemini 3.5 Flash-Lite and newer reject a request that ends with an AI message.
    assert fill(read_prompt(name))[-1].type == "human"


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_every_prompt_has_only_the_slots_we_fill(name):
    variables = read_prompt(name).template().input_variables
    assert variables in (["note"], ["transfers", "trip"])


def test_summary_v1_is_two_messages_with_the_note_in_delimiters():
    messages = fill(read_prompt("summary_v1"))
    assert [m.type for m in messages] == ["system", "human"]
    assert "<note>\nCoorg weekend (Arjun, Meera, Kabir)" in messages[1].text
    assert messages[1].text.endswith("</note>")


@pytest.mark.parametrize("name", ["expense_fewshot_v1", "balance_fewshot_v1"])
def test_few_shot_examples_become_message_pairs(name):
    prompt = read_prompt(name)
    assert [m.type for m in fill(prompt)] == ["system", "human", "ai", "human", "ai", "human"]
    assert [m.type for m in fill(prompt, with_examples=False)] == ["system", "human"]


def test_the_examples_are_not_among_the_ten_test_notes():
    example_names = {"Neha", "Sameer", "Priya", "Aman"}
    for note in load_notes():
        assert not example_names & set(note.people)
    for name in ["expense_fewshot_v1", "balance_fewshot_v1"]:
        example_texts = {e["note"] for e in read_prompt(name).examples}
        assert not example_texts & {n.note for n in load_notes()}


# The two worked examples, written as expenses by hand, so we can prove their answers are right.
LONAVALA = [Expense(payer="Neha", item="cab", amount=900, category="travel", shared_with=["Neha", "Sameer", "Priya"]),
            Expense(payer="Sameer", item="vada pav", amount=600, category="food", shared_with=["Neha", "Sameer"]),
            Expense(payer="Priya", item="fort ticket", amount=300, category="activity", shared_with=["Priya"])]
ALIBAUG = [Expense(payer="Aman", item="villa", amount=4000, category="stay", shared_with=["Neha", "Sameer", "Priya", "Aman"]),
           Expense(payer="Priya", item="ferry", amount=800, category="travel", shared_with=["Neha", "Sameer", "Priya", "Aman"]),
           Expense(payer="Sameer", item="cash to Aman", amount=500, category="other", shared_with=["Aman"])]


def test_the_worked_examples_teach_correct_answers():
    lonavala, alibaug = read_prompt("balance_fewshot_v1").examples
    assert same_result(parse_transfers(lonavala["answer"]), LONAVALA)
    assert same_result(parse_transfers(alibaug["answer"]), ALIBAUG)


def test_braces_in_a_note_do_not_break_the_template():
    messages = read_prompt("balance_rtc_v1").template().format_messages(**note_inputs("Asha paid {500} for {lunch}"))
    assert "Asha paid {500} for {lunch}" in messages[-1].text


def test_a_file_whose_version_does_not_match_its_name_is_refused(tmp_path):
    (tmp_path / "summary_v3.toml").write_text('name = "summary"\nversion = 2\ntechnique = "x"\noutput = "x"\nhuman = "{note}"\n')
    with pytest.raises(PromptFileError, match="must be called summary_v2.toml"):
        read_prompt("summary_v3", tmp_path)


def test_a_file_with_a_missing_field_is_refused(tmp_path):
    (tmp_path / "x_v1.toml").write_text('name = "x"\nversion = 1\n')
    with pytest.raises(PromptFileError, match="missing: technique, output, human"):
        read_prompt("x_v1", tmp_path)


def test_an_unknown_prompt_lists_the_known_ones():
    with pytest.raises(PromptFileError, match="Known prompts: balance_cot_v1"):
        read_prompt("balance_magic_v9")


def test_the_list_command(capsys):
    assert prompt_files.main([]) == 0
    output = capsys.readouterr().out
    assert f"{len(ALL_PROMPTS)} prompt files in prompts/" in output
    assert all(name in output for name in OUR_PROMPTS)
    assert len(list_prompts()) == len(ALL_PROMPTS)


def test_showing_the_message_prompt_fills_it_from_settle(capsys):
    assert prompt_files.main(["message_v1"]) == 0
    assert "Kabir owes Meera 740 rupees\nArjun owes Meera 230 rupees" in capsys.readouterr().out
