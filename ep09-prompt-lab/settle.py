"""
settle.py - who owes whom, worked out by plain Python (Step 7).

Code is the source of truth for maths. The model may read the note for us, but the
money is added up here, the same way every time, and tested.

    balances(expenses)   paid minus share, per person (+ means "gets money back")
    settle(expenses)     the transfers that make everyone square
    same_result(...)     does a proposed list of transfers settle the trip correctly?

How settle() works (the "biggest debt pays the biggest credit" rule):
    1. Each payment is split equally between the people it was for.
    2. Balance = what you paid - your shares.
    3. The person who owes the most pays the person who is owed the most,
       as much as one of them needs. Repeat until everyone is at zero.

Run:  python settle.py              (every note in data/trip_notes.jsonl, offline, no key)
      python settle.py --note 1     (one note, with the paid / share / balance table)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
from fractions import Fraction

from common import get_note, load_notes
from models import Expense, Settlement, Transfer, rupees

# Two answers that differ by less than one rupee count as the same (rounding).
TOLERANCE = 1


def clean_name(name: str) -> str:
    """' arjun ' -> 'Arjun', so 'Arjun' and 'arjun' are the same person."""
    return name.strip().title()


def balances(expenses: list[Expense]) -> dict[str, Fraction]:
    """Paid minus share for every person. Fraction keeps 1000 / 3 exact, so the total is always 0."""
    balance: dict[str, Fraction] = {}
    for expense in expenses:
        amount = Fraction(str(expense.amount))            # str() avoids float surprises like 0.1 + 0.2
        payer = clean_name(expense.payer)
        balance[payer] = balance.get(payer, Fraction(0)) + amount
        sharers = [clean_name(name) for name in expense.shared_with]
        share = amount / len(sharers)
        for person in sharers:
            balance[person] = balance.get(person, Fraction(0)) - share
    return balance


def settle(expenses: list[Expense]) -> Settlement:
    """Turn the payments into the transfers that make everyone square."""
    left = {person: amount for person, amount in balances(expenses).items() if abs(amount) >= Fraction(1, 100)}
    transfers = []
    while left:
        # The biggest debt and the biggest credit (ties: alphabetical, so the result never changes).
        debtor = sorted(left, key=lambda person: (left[person], person))[0]
        creditor = sorted(left, key=lambda person: (-left[person], person))[0]
        if left[debtor] >= 0:      # nobody owes anything any more (only crumbs below one paisa)
            break
        amount = min(-left[debtor], left[creditor])
        transfers.append(Transfer(from_person=debtor, to_person=creditor, amount=round(float(amount), 2)))
        left[debtor] += amount
        left[creditor] -= amount
        left = {person: value for person, value in left.items() if abs(value) >= Fraction(1, 100)}
    return Settlement(transfers=transfers)


def same_result(proposed: list[Transfer], expenses: list[Expense]) -> bool:
    """True if the proposed transfers leave EVERYONE at zero (within one rupee).

    We compare balances, not lines: "Arjun pays Zoya 400" can be just as right as
    "Arjun pays Kabir 300 and Zoya 100". There is often more than one correct settle-up.
    """
    after = dict(balances(expenses))
    for transfer in proposed:
        payer, receiver = clean_name(transfer.from_person), clean_name(transfer.to_person)
        after[payer] = after.get(payer, Fraction(0)) + Fraction(str(transfer.amount))
        after[receiver] = after.get(receiver, Fraction(0)) - Fraction(str(transfer.amount))
    return all(abs(value) <= TOLERANCE for value in after.values())


def balance_table(expenses: list[Expense]) -> list[str]:
    """Paid, share and balance per person, as lines of text (the "working" a person would write)."""
    paid: dict[str, Fraction] = {}
    for expense in expenses:
        payer = clean_name(expense.payer)
        paid[payer] = paid.get(payer, Fraction(0)) + Fraction(str(expense.amount))
    rows = [f"  {'person':<8}{'paid':>8}{'share':>8}{'balance':>9}"]
    for person, balance in balances(expenses).items():
        person_paid = paid.get(person, Fraction(0))
        share = person_paid - balance
        sign = "+" if balance > 0 else ""
        rows.append(f"  {person:<8}{rupees(person_paid):>8}{rupees(share):>8}{sign + rupees(balance):>9}")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Settle trip expenses with plain Python (no AI).")
    parser.add_argument("--note", help="note number or name, e.g. 1 or coorg (default: every note)")
    args = parser.parse_args(argv)

    try:
        notes = load_notes()
        note = get_note(args.note) if args.note else None
    except ValueError as error:        # a typo in data/trip_notes.jsonl, or an unknown note
        print(error)
        return 1
    if note:
        print(f"Note {note.id} ({note.slug}): {len(note.expenses)} payments from your answer key\n")
        print("\n".join(balance_table(note.expenses)))
        print()
        for line in settle(note.expenses).lines():
            print(f"  {line}")
        return 0

    agree = 0
    for note in notes:
        result = settle(note.expenses)
        print(f"Note {note.id:>2} {note.slug:<10} " + " | ".join(result.lines()))
        # Your hand-written answer is right if it ALSO leaves everyone at zero.
        if same_result(note.transfers, note.expenses):
            agree += 1
        else:
            print(f"           your answer key says: {' | '.join(Settlement(transfers=note.transfers).lines())}"
                  " - it does not add up, check it")
    print(f"\nsettle() agrees with your hand-checked answers on {agree} of {len(notes)} notes.")
    return 0 if agree == len(notes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
