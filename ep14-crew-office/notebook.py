"""
notebook.py - what the tools REALLY did, written down by our code (the Shelf idea from Episode 13).

Agents talk. Tools act. After a run we never trust what an agent SAYS it did; we read
this notebook instead:

  CaseFile  (outreach crew)  every file the file tool read, line by line, and the brief of each lead
  Ledger    (research desk)  every number and date a desk tool returned

The claims checks compare the agents' words with these records.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import json
import re
from pathlib import Path

ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class CaseFile:
    """The outreach crew's notice board. One 'run' per lead."""

    def __init__(self):
        self.lines: dict[str, list[str]] = {}   # every file read: "leads/x.md" -> its lines
        self.runs: dict[str, dict] = {}         # "kesar-crumb-bakehouse" -> {"tool_calls": [...], "brief": ...}
        self.current: str = "no-lead"

    # --- called by CrewAI ---------------------------------------------------
    def start(self, inputs: dict) -> dict:
        """Crew(before_kickoff_callbacks=[case.start]): CrewAI calls this at the start of EVERY kickoff,
        also once per lead inside kickoff_for_each. It must hand the inputs back."""
        self.current = Path(str(inputs.get("lead_file", "no-lead"))).stem
        self.runs[self.current] = {"tool_calls": [], "brief": None}
        return inputs

    def save_brief(self, output) -> None:
        """Task(callback=case.save_brief): CrewAI calls this when the Researcher's task is done."""
        self.run["brief"] = output.pydantic

    # --- called by our tools ------------------------------------------------
    @property
    def run(self) -> dict:
        return self.runs.setdefault(self.current, {"tool_calls": [], "brief": None})

    def note_call(self, tool_name: str) -> None:
        self.run["tool_calls"].append(tool_name)

    def remember_file(self, path: str, lines: list[str]) -> None:
        self.lines[path] = lines

    def line_at(self, source: str) -> str | None:
        """'leads/x.md:10' -> the text of line 10 of that file, if the file tool really read it."""
        path, _, number = source.rpartition(":")
        if not number.isdigit() or path not in self.lines:
            return None
        lines = self.lines[path]
        index = int(number) - 1
        return lines[index] if 0 <= index < len(lines) else None


class Ledger:
    """Every number and date the research desk's tools returned. The claims checker's evidence."""

    def __init__(self):
        self.numbers: list[float] = []
        self.dates: set[str] = set()
        self.calls: list[str] = []

    def record(self, tool_name: str, value) -> None:
        """Walk a tool's answer (dicts, lists, numbers, text) and write down every number and date."""
        if tool_name:
            self.calls.append(tool_name)
        if isinstance(value, dict):
            for item in value.values():
                self.record("", item)
        elif isinstance(value, list):
            for item in value:
                self.record("", item)
        elif isinstance(value, bool):
            return                                   # True/False are not numbers here
        elif isinstance(value, (int, float)):
            self.numbers.append(float(value))
        elif isinstance(value, str):
            self.dates.update(ISO_DATE.findall(value))

    def record_text(self, tool_name: str, text: str) -> None:
        """For tools that return plain text (the file tool): keep its dates and numbers too."""
        from claims_check import find_numbers      # imported here: claims_check imports nothing from us

        self.calls.append(tool_name)
        self.dates.update(ISO_DATE.findall(text))
        self.numbers.extend(abs(float(n.replace(",", "").rstrip("%"))) for n in find_numbers(text))

    def to_dict(self) -> dict:
        return {"numbers": sorted(set(self.numbers)), "dates": sorted(self.dates), "tool_calls": self.calls}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
