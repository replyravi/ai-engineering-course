"""
tests/test_techniques.py - every technique with a scripted fake model (Steps 2-6, 8 and 9).

We script what the "AI" answers and check OUR code: what we send, how we read the reply,
and how every reply is checked against settle().
"""

import pytest

import techniques
from common import get_note, load_notes
from lab_fixtures import key_reply, lines_reply, no_keys  # noqa: F401
from models import BalanceAnswer, Transfer
from settle import settle
from techniques import (ask_transfers, bad_reply_demo, compare_cot, compare_labels, extract_expenses,
                        final_answer_part, parse_category_lines, parse_transfers, run_text_prompt,
                        score_labels, zero_shot_summary)
from testing.fakes import fake_llm, gemini_style_reply, structured_reply, tool_call

NOTE = get_note("coorg")


def test_zero_shot_sends_one_plain_user_string():
    llm = fake_llm(gemini_style_reply("A loose paragraph about the trip."))
    assert zero_shot_summary(NOTE.note, llm) == "A loose paragraph about the trip."
    sent = llm.received[0]
    assert len(sent) == 1 and sent[0].type == "human"
    assert sent[0].text.startswith("Summarise this note") and NOTE.note in sent[0].text


def test_rtc_sends_the_role_as_a_system_message():
    llm = fake_llm("- Meera paid 1800 rupees for homestay (shared by Arjun, Meera, Kabir)")
    assert run_text_prompt("summary_v2", NOTE.note, llm).startswith("- Meera paid 1800")
    assert llm.received[0][0].type == "system"
    assert "careful trip accountant" in llm.received[0][0].text


@pytest.mark.parametrize("reply, expected", [
    ("Kabir owes Meera 740\nArjun owes Meera 230", [("Kabir", "Meera", 740), ("Arjun", "Meera", 230)]),
    ("**Kabir** owes **Meera** ₹740 and Arjun owes Meera Rs. 230.", [("Kabir", "Meera", 740), ("Arjun", "Meera", 230)]),
    ("- Kabir -> Meera: 1,520 rupees", [("Kabir", "Meera", 1520)]),
    ("kabir needs to pay meera 740", [("Kabir", "Meera", 740)]),
    ("Zoya and Tanvi owe Rohan 560 each", [("Tanvi", "Rohan", 560)]),
    ("Everyone is square.", []),
])
def test_parse_transfers(reply, expected):
    assert [(t.from_person, t.to_person, t.amount) for t in parse_transfers(reply)] == expected


def test_a_reply_that_says_nobody_owes_is_an_answer_not_a_blank():
    assert techniques.read_transfers("Everyone is square, nobody owes anything.") == []
    assert techniques.read_transfers("I am not sure what you mean.") is None


def test_only_the_final_answer_part_is_read():
    text = "WORKING:\nKabir owes Meera 830 before his chai...\nFINAL ANSWER:\nKabir owes Meera 740"
    assert final_answer_part(text).strip() == "Kabir owes Meera 740"
    assert final_answer_part("no marker here") == ""


def test_category_lines_are_scored_against_the_answer_key():
    reply = "Meera | 1800 | stay\nArjun | 600 | travel\nKabir | ₹90 | snacks"
    assert parse_category_lines(reply)[2] == ("kabir", 90.0, "snacks")
    assert score_labels(reply, NOTE.expenses) == 2               # snacks is not in our label set


def test_few_shot_comparison_counts_labels_on_every_note(capsys):
    notes = load_notes()
    replies = []
    for note in notes:
        replies.append("\n".join(f"{e.payer} | {e.amount:g} | travel" for e in note.expenses))   # without examples
        replies.append("\n".join(f"{e.payer} | {e.amount:g} | {e.category}" for e in note.expenses))  # with examples
    llm = fake_llm(*replies)
    totals = compare_labels(notes, llm)

    travel_in_key = sum(e.category == "travel" for n in notes for e in n.expenses)
    assert totals == {"without": travel_in_key, "with": 34, "payments": 34, "perfect_without": 0, "perfect_with": 10}
    first_call, second_call = llm.received[0], llm.received[1]
    assert len(first_call) == 2 and len(second_call) == 6        # the examples add two message pairs
    assert "total" in capsys.readouterr().out


def test_structured_output_returns_typed_expenses():
    result = extract_expenses(NOTE.note, fake_llm(key_reply(NOTE)))
    assert type(result).__name__ == "ExpenseList"
    assert result.expenses == NOTE.expenses


def test_a_broken_reply_is_rejected_by_the_schema():
    message = bad_reply_demo(NOTE.note)
    assert message.startswith("3 problems, so the reply was rejected:")
    assert "expenses.0.category" in message and "'hotel'" in message


def test_ask_transfers_reads_free_text_lines():
    answer = ask_transfers("balance_rtc_v1", NOTE, fake_llm(lines_reply(NOTE)))
    assert answer.transfers == settle(NOTE.expenses).transfers


def test_chain_of_thought_text_keeps_the_working_apart():
    reply = "WORKING:\nMeera's balance is +970...\nFINAL ANSWER:\nKabir owes Meera 740\nArjun owes Meera 230"
    answer = ask_transfers("balance_cot_v1", NOTE, fake_llm(reply))
    assert "+970" in answer.hidden and "+970" not in "".join(t.line() for t in answer.transfers)
    assert techniques.verdict(answer, NOTE) == "match"


def test_chain_of_thought_without_a_final_answer_is_no_answer():
    answer = ask_transfers("balance_cot_v1", NOTE, fake_llm("Kabir owes Meera 740 I think"))
    assert answer.transfers is None and techniques.verdict(answer, NOTE) == "no answer"


def test_the_schema_version_hides_the_reasoning():
    final = [Transfer(from_person="Kabir", to_person="Meera", amount=740),
             Transfer(from_person="Arjun", to_person="Meera", amount=230)]
    reply = structured_reply(BalanceAnswer(reasoning="Meera paid 1800, her share is 830...", final_answer=final))
    answer = ask_transfers("balance_cot_v2", NOTE, fake_llm(reply))
    assert answer.transfers == final
    assert answer.hidden.startswith("Meera paid 1800")
    assert "Meera paid 1800" not in answer.reply


def test_a_reply_that_does_not_fit_the_form_is_no_answer_not_a_crash():
    broken = tool_call("ExpenseList", {"expenses": [{"payer": "Meera", "amount": "lots"}]})
    answer = ask_transfers("expenses_v1", NOTE, fake_llm(broken))
    assert answer.transfers is None and answer.error.startswith("reply did not fit the form")


def test_structured_extraction_lets_python_do_the_maths():
    answer = ask_transfers("expenses_v1", NOTE, fake_llm(key_reply(NOTE)))
    assert answer.transfers == NOTE.transfers and answer.held_back == []


def test_compare_cot_logs_a_fix(tmp_path, capsys):
    goa = get_note("goa")
    wrong = "Meera owes Zoya 500\nArjun owes Kabir 375"                         # forgot "except Meera"
    right = "WORKING:\n...\nFINAL ANSWER:\n" + lines_reply(goa)
    log = tmp_path / "results" / "cot_log.jsonl"
    counts = compare_cot([goa], fake_llm(wrong, right), log_path=log)
    assert counts["fixed"] == [3] and counts["direct"] == 0 and counts["cot"] == 1
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2
    assert "chain of thought fixed notes: [3]" in capsys.readouterr().out


def test_commands_without_a_key_show_the_help(no_keys, capsys):
    for command in (["zero-shot"], ["rtc"], ["few-shot"], ["structured"], ["cot"], ["cot-hidden"], ["injection"]):
        assert techniques.main(command) == 1
        output = capsys.readouterr().out
        assert "https://aistudio.google.com/apikey" in output and "Traceback" not in output


def test_show_prompt_works_offline(no_keys, capsys):
    assert techniques.main(["few-shot", "--show-prompt"]) == 0
    assert "the two examples add about" in capsys.readouterr().out
    assert techniques.main(["template"]) == 0
    assert "summary_v1: slots ['note'], 2 messages" in capsys.readouterr().out


def test_cot_hidden_prints_only_the_final_answer(monkeypatch, tmp_path, capsys):
    jaipur = get_note("jaipur")
    reply = structured_reply(BalanceAnswer(reasoning="SECRET WORKING", final_answer=jaipur.transfers))
    monkeypatch.setattr(techniques, "make_llm", lambda thinking=None: fake_llm(reply))
    monkeypatch.setattr(techniques, "RESULTS_DIR", tmp_path)
    assert techniques.main(["cot-hidden"]) == 0
    output = capsys.readouterr().out
    assert "Arjun owes Meera 500 rupees" in output and "Python check: match" in output
    assert "SECRET WORKING" not in output
    assert "SECRET WORKING" in (tmp_path / "reasoning_log.jsonl").read_text(encoding="utf-8")
