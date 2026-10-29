"""
tests/test_settle.py - the maths, checked against answers worked out by hand (Step 7).

No model here at all: settle() is plain Python, so it must be exactly right, every time.
"""

from fractions import Fraction

import pytest

import settle as settle_module
from common import get_note, load_notes
from models import Expense, Transfer
from settle import balances, same_result, settle

NOTES = load_notes()


@pytest.mark.parametrize("note", NOTES, ids=[n.slug for n in NOTES])
def test_settle_matches_the_hand_checked_answer(note):
    assert settle(note.expenses).transfers == note.transfers


@pytest.mark.parametrize("note", NOTES, ids=[n.slug for n in NOTES])
def test_balances_always_add_up_to_zero(note):
    assert sum(balances(note.expenses).values()) == 0


def test_the_cold_open_note():
    lines = settle(get_note("coorg").expenses).lines()
    assert lines == ["Kabir owes Meera 740 rupees", "Arjun owes Meera 230 rupees"]


def test_everyone_except_meera():
    # Goa: Meera skipped parasailing, so she pays no share of the 1500.
    assert balances(get_note("goa").expenses) == {"Zoya": 800, "Kabir": 300, "Arjun": -400, "Meera": -700}


def test_a_payment_for_one_person_only():
    # Udaipur: Meera paid 700 for Tanvi's dinner only.
    assert settle(get_note("udaipur").expenses).lines() == ["Tanvi owes Meera 600 rupees"]


def test_an_uneven_split_stays_exact():
    expenses = [Expense(payer="Asha", item="cab", amount=1000, category="travel", shared_with=["Asha", "Bina", "Chetan"])]
    assert sum(balances(expenses).values()) == 0                     # Fraction: no lost paisa
    assert balances(expenses)["Bina"] == Fraction(-1000, 3)
    result = settle(expenses)
    assert [t.amount for t in result.transfers] == [333.33, 333.33]
    assert same_result(result.transfers, expenses)


def test_nobody_owes_when_everyone_paid_for_themselves():
    expenses = [Expense(payer="Asha", item="ticket", amount=300, category="activity", shared_with=["Asha"]),
                Expense(payer="Bina", item="ticket", amount=300, category="activity", shared_with=["Bina"])]
    assert settle(expenses).transfers == []
    assert settle(expenses).lines() == ["Everyone is square. Nobody owes anything."]


def test_names_match_whatever_the_case():
    expenses = [Expense(payer="arjun ", item="cab", amount=600, category="travel", shared_with=["Arjun", "MEERA"])]
    assert settle(expenses).lines() == ["Meera owes Arjun 300 rupees"]


def test_another_correct_settle_up_is_accepted():
    # Goa can also be settled as: Arjun pays Zoya 400, Meera pays Zoya 400 and Kabir 300.
    other_way = [Transfer(from_person="Arjun", to_person="Zoya", amount=400),
                 Transfer(from_person="Meera", to_person="Zoya", amount=400),
                 Transfer(from_person="Meera", to_person="Kabir", amount=300)]
    assert same_result(other_way, get_note("goa").expenses)


@pytest.mark.parametrize("wrong", [
    [Transfer(from_person="Arjun", to_person="Meera", amount=600), Transfer(from_person="Kabir", to_person="Meera", amount=740)],
    [Transfer(from_person="Kabir", to_person="Meera", amount=740)],                                   # one missing
    [Transfer(from_person="Meera", to_person="Kabir", amount=740), Transfer(from_person="Arjun", to_person="Meera", amount=230)],
    [Transfer(from_person="Everyone", to_person="Dev", amount=5000)],
], ids=["wrong amount", "missing line", "wrong direction", "invented person"])
def test_a_wrong_settle_up_is_rejected(wrong):
    assert not same_result(wrong, get_note("coorg").expenses)


def test_a_rounding_difference_below_one_rupee_is_fine():
    close = [Transfer(from_person="Kabir", to_person="Meera", amount=740.4),
             Transfer(from_person="Arjun", to_person="Meera", amount=229.8)]
    assert same_result(close, get_note("coorg").expenses)


def test_the_command_line_checks_all_ten_notes(capsys):
    assert settle_module.main([]) == 0
    output = capsys.readouterr().out
    assert "settle() agrees with your hand-checked answers on 10 of 10 notes." in output
    assert "Note  1 coorg      Kabir owes Meera 740 rupees | Arjun owes Meera 230 rupees" in output


def test_the_balance_table_for_one_note(capsys):
    assert settle_module.main(["--note", "1"]) == 0
    output = capsys.readouterr().out
    assert "Meera       1800     830     +970" in output
    assert "Kabir         90     830     -740" in output
