"""
tools/brief_check.py - Step 6: "no fact outside the brief", checked by Python.

A journalist may only write what the researcher's notes support. Our Writer may only use
what the Researcher's brief (LeadBrief) says. This tool gives the Reviewer hard evidence:

  numbers_outside_brief   every number in the draft that does not appear in the brief
                          (a made-up "40% more pick-ups" is caught here)
  links_or_addresses      web links or email addresses in the draft (our code adds the
                          recipient later; the model never types an address)

It reads the brief from the CaseFile, where CrewAI's task callback saved it, so the model
cannot hand the checker a different brief.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import json
import re

from crewai.tools import BaseTool, ToolFailure
from pydantic import BaseModel, ConfigDict, Field

from claims_check import as_value, find_numbers, unsupported_numbers
from models import LeadBrief
from notebook import CaseFile

LINK_OR_ADDRESS = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def brief_numbers(brief: LeadBrief) -> list[float]:
    """Every number written in the brief's facts and product points."""
    texts = [fact.text for fact in brief.facts + brief.product_points]
    return [as_value(number)[0] for text in texts for number in find_numbers(text)]


def check_against_brief(text: str, brief: LeadBrief) -> dict:
    outside = unsupported_numbers(text, brief_numbers(brief))
    links = LINK_OR_ADDRESS.findall(text)
    return {
        "numbers_in_draft": find_numbers(text),
        "numbers_outside_brief": outside,
        "links_or_addresses": links,
        "ok": not outside and not links,
    }


class DraftInput(BaseModel):
    text: str = Field(description="The draft's subject line and body, as plain text")


class CheckAgainstBriefTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "check_against_brief"
    description: str = (
        "Check an email draft against the Researcher's brief. Returns JSON: the numbers in the draft, "
        "the numbers NOT found in the brief, any links or email addresses, and ok (true or false). "
        "Call it once with the subject and the body together."
    )
    args_schema: type[BaseModel] = DraftInput
    case: CaseFile

    def _run(self, text: str):
        brief = self.case.run.get("brief")
        if brief is None:
            return ToolFailure(message="No brief saved yet: the Researcher's task has not finished.")
        self.case.note_call(self.name)
        return json.dumps(check_against_brief(text, brief))
