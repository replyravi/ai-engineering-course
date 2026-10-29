"""
models.py - the SHAPES of the answers our agents hand to each other (Steps 4, 5, 6 and 12).

A shape is a promise. "A Fact always has an id, a text and a source line." CrewAI turns each
agent's answer into one of these classes (output_pydantic), and Pydantic refuses anything
that breaks the promise, for example a brief with only one fact.

Notice what is NOT here: no email address in any model answer (our code copies it from the
lead file), and no buy, sell or hold field in the research report (the desk never advises).

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Crew A: the outreach office
# ---------------------------------------------------------------------------

class Fact(BaseModel):
    """One fact, with the exact line it came from."""

    id: str = Field(description="F1, F2, F3 ... for lead facts; P1, P2 ... for product points")
    text: str = Field(description="the fact in plain words, taken from ONE line of a file")
    source: str = Field(description="file and line number exactly as read_lead_file shows them, "
                                    "for example leads/kesar-crumb-bakehouse.md:10")


class LeadBrief(BaseModel):
    """The Researcher's typed answer: what we know about one lead, and where it says so."""

    company: str = Field(description="the business name from the lead file")
    contact_name: str = Field(description="the contact person's name from the lead file")
    facts: list[Fact] = Field(min_length=3, max_length=3, description="exactly 3 facts about the lead")
    product_points: list[Fact] = Field(min_length=1, max_length=3,
                                       description="1 to 3 points from product.md that fit this lead")


class EmailDraft(BaseModel):
    """The Writer's typed answer. A DRAFT: nothing here is ever sent by code."""

    subject: str = Field(max_length=90, description="a short, plain subject line")
    body: str = Field(description="the email text, signed by the sender named in product.md")
    facts_used: list[str] = Field(description="ids of the brief facts the email uses, for example ['F1', 'P2']")


class Verdict(BaseModel):
    """The Reviewer's typed answer: pass, or a list of fixes for the human to see."""

    status: Literal["pass", "fix"] = Field(description="pass only when both checks found no problem")
    tone: str = Field(description="what check_tone said, for example 'POSITIVE 0.99, warm'")
    numbers_outside_brief: list[str] = Field(default_factory=list,
                                             description="numbers check_against_brief could not find in the brief")
    fixes: list[str] = Field(default_factory=list, description="what to change; empty when status is pass")


# ---------------------------------------------------------------------------
# Crew B: the research desk (education only)
# ---------------------------------------------------------------------------

class DeskReport(BaseModel):
    """The Risk Reviewer's typed answer: an explanation, never a recommendation."""

    ticker: str = Field(description="the invented ticker, for example DEMO1")
    what_moved: str = Field(description="what happened to the price, with numbers copied exactly from the tools")
    what_the_numbers_mean: str = Field(description="each number explained in plain words for a beginner")
    paper_backtest: str = Field(description="what the paper test of the moving-average rule showed, and its limits")
    risks: list[str] = Field(min_length=2, max_length=5,
                             description="limits and risks a learner should keep in mind")
