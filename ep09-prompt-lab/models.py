"""
models.py - the FORMS (Pydantic schemas) of our prompt lab (Steps 6, 7 and 9).

A schema is a printed form with boxes. We give it to the model with with_structured_output,
and the model has to fill in every box. Python then checks the boxes: a wrong type,
a missing field or a category outside our label set is an error, not a guess.

    Expense        one payment from a trip note (Step 6)
    ExpenseList    all the payments of one note (what the model returns in Step 6)
    Transfer       one line of the settle-up: "Kabir owes Meera 740 rupees" (Step 7)
    Settlement     all the transfers of one trip (what settle() returns in Step 7)
    BalanceAnswer  a chain-of-thought answer: hidden working + clean final answer (Step 9)
    TripNote       one line of data/trip_notes.jsonl: a note plus its correct answers

You do not run this file. The other files import it.

Author: Ravi Shanker Singh (TechWithRavi)
"""

from typing import Literal

from pydantic import BaseModel, Field

# Our label set. Five words, nothing else is allowed.
Category = Literal["travel", "stay", "food", "activity", "other"]


class Expense(BaseModel):
    """One payment written in a group-trip note."""

    payer: str = Field(description="The first name of the person who paid.")
    item: str = Field(description="What the money was for, in a few words, for example homestay or cab.")
    # ge=1 means "at least 1 rupee". (Gemini's schema supports minimum, so we use it.)
    amount: float = Field(ge=1, description="How much was paid, in rupees, as a plain number.")
    category: Category = Field(description="One of: travel, stay, food, activity, other.")
    shared_with: list[str] = Field(
        min_length=1,
        description="Everyone this payment was for. They split it equally. "
                    "Include the payer only if they shared it too.",
    )


class ExpenseList(BaseModel):
    """All the payments in one trip note, in the order they appear."""

    expenses: list[Expense]


class Transfer(BaseModel):
    """One settle-up line: from_person pays to_person this amount."""

    from_person: str = Field(description="The person who owes money.")
    to_person: str = Field(description="The person who gets the money.")
    amount: float = Field(ge=0.01, description="Rupees to pay, as a plain number.")

    def line(self) -> str:
        return f"{self.from_person} owes {self.to_person} {rupees(self.amount)} rupees"


class Settlement(BaseModel):
    """Who owes whom at the end of a trip."""

    transfers: list[Transfer]

    def lines(self) -> list[str]:
        return [t.line() for t in self.transfers] or ["Everyone is square. Nobody owes anything."]


class BalanceAnswer(BaseModel):
    """A chain-of-thought answer. The working comes first, the clean answer second."""

    # The working goes in its own box, so our app can keep it backstage (Step 9).
    reasoning: str = Field(
        description="Your step-by-step working: each payment, each person's share, "
                    "each person's balance (paid minus share). Never shown to the group."
    )
    final_answer: list[Transfer] = Field(
        description="Who owes whom, one transfer per line. Empty if everyone is square."
    )


class TripNote(BaseModel):
    """One line of data/trip_notes.jsonl: a messy note and the answers YOU worked out by hand."""

    id: int
    slug: str
    trip: str
    people: list[str] = Field(min_length=2)
    tags: list[str] = []
    note: str
    expenses: list[Expense]
    transfers: list[Transfer]


def rupees(amount: float) -> str:
    """740.0 -> '740', 333.333 -> '333.33', 1520 -> '1520' (no commas, so tests can find the number)."""
    amount = round(float(amount), 2)
    return str(int(amount)) if amount == int(amount) else f"{amount:.2f}"
