"""
inputs_check.py - Steps 2 and 8: one crew, many leads, and no blank left empty.

Think of a mail-merge letter: "Dear {name}, congratulations on {news}". CrewAI does the same
with task text: kickoff(inputs={...}) fills every {placeholder} in the agents' and tasks' text.
Two quiet mistakes are common:

  a MISSING input   a {placeholder} with no value  -> CrewAI stops with a ValueError (good, loud)
  an UNUSED input   a value no text asks for       -> CrewAI ignores it in silence (bad, quiet)

Older tutorials passed inputs like "initial capital" that no task ever used, so the crew
looked like it considered them and never did. This file finds both mistakes BEFORE kickoff.

    python inputs_check.py          (Step 2: our leads, the inputs we build, and the check)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import re

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def placeholders_in(crew) -> dict[str, list[str]]:
    """Every {name} in the crew's text, and where it appears: {'company': ['Researcher goal', ...]}."""
    found: dict[str, list[str]] = {}

    def scan(text, where):
        for name in PLACEHOLDER.findall(text or ""):
            places = found.setdefault(name, [])
            if where not in places:
                places.append(where)

    for agent in crew.agents:
        for field in ("role", "goal", "backstory"):
            scan(getattr(agent, field), f"{agent.role} {field}")
    for number, task in enumerate(crew.tasks, start=1):
        for field in ("description", "expected_output", "output_file"):
            scan(getattr(task, field), f"task {number} {field}")
    return found


def check_inputs(crew, inputs: dict) -> tuple[list[str], list[str]]:
    """(missing, unused): placeholders with no input, and inputs no placeholder asks for."""
    wanted = placeholders_in(crew)
    missing = sorted(name for name in wanted if name not in inputs)
    unused = sorted(name for name in inputs if name not in wanted)
    return missing, unused


def main() -> int:
    from datetime import date

    from outreach_crew import LEADS, PRODUCT_FILE, build_crew, lead_inputs
    from replay import ScriptedLLM

    print("Step 2: the leads and the product sheet (all invented, all local files)\n")
    for lead in LEADS:
        print(f"  {lead.as_posix():<40} {len(lead.read_text(encoding='utf-8').splitlines())} lines")
    print(f"  {PRODUCT_FILE.as_posix():<40} {len(PRODUCT_FILE.read_text(encoding='utf-8').splitlines())} lines")

    # We only READ the crew's text here, so a scripted model with no replies is enough: nothing is sent.
    crew = build_crew(llm=ScriptedLLM(responses=[]), verbose=False)
    print("\nPlaceholders in the crew's text:")
    for name, places in placeholders_in(crew).items():
        label = "{" + name + "}"
        print(f"  {label:<12} {', '.join(places)}")

    today = date.today().isoformat()
    print(f"\nInputs we build for each lead (today = {today}):")
    for lead in LEADS:
        inputs = lead_inputs(lead, today)
        missing, unused = check_inputs(crew, inputs)
        status = "OK, every input used, no blank left" if not (missing or unused) else \
            f"PROBLEM missing={missing} unused={unused}"
        print(f"  {inputs['company']:<26} {status}")

    print("\nThe classic mistakes, on purpose:")
    bad = dict(lead_inputs(LEADS[0], today), budget="Rs 50,000")      # an input no task uses
    del bad["today"]                                                   # and a blank with no value
    missing, unused = check_inputs(crew, bad)
    print(f"  missing: {missing}   -> CrewAI would stop: Missing required template variable")
    print(f"  unused:  {unused}  -> CrewAI would ignore it without a word")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
