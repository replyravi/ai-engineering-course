"""
excessive_agency.py - Step 7: what "excessive agency" looks like, and why our crew cannot do it.

Episode 05 met this risk in the OWASP Top 10 for LLM Applications (2025): LLM06, Excessive
Agency, giving a model more tools and permissions than its job needs. An email WRITER needs to
write. The moment it also holds a "send" tool, one confident model reply is one email to a
real person: maybe the wrong person, maybe a made-up promise, and nobody looked first.

Part 1 (scripted, no key): a Writer that IS given a send_email tool. Our send_email is a
harmless stand-in: it only writes the call into a list, so nothing leaves your computer.
Part 2 (real): an audit of our real outreach crew: every agent, every tool, can any of them send?

    python excessive_agency.py

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from pathlib import Path

from outreach_crew import build_crew, recipient
from replay import ScriptedLLM
from testing.fakes import react_final_answer, react_tool_use

WOULD_HAVE_SENT: list[dict] = []          # the stand-in "outbox": a plain Python list
SENDING_WORDS = ("send", "smtp", "mail_to", "post_", "publish")


@tool("send_email")
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer. (A harmless stand-in: it only records the call.)"""
    WOULD_HAVE_SENT.append({"to": to, "subject": subject, "body": body})
    return f"Email sent to {to}."             # what a real send tool would answer


def part_one() -> None:
    print("PART 1 - a Writer WITH a send tool (scripted model, stand-in tool, nothing leaves this computer)\n")
    scripted = ScriptedLLM(responses=[
        react_tool_use("send_email", {
            "to": "nandini@kesarcrumb.example",
            "subject": "[scripted] Fix your Diwali chaos today",
            "body": "[scripted] Bakeries like yours get 50% more orders with Pickup Parrot. Act now!",
        }, thought="The task says write and send, so I will send it right away"),
        react_final_answer("[scripted] Done, the email has been sent."),
    ])
    writer = Agent(role="Outreach Writer", goal="Write and send outreach emails",
                   backstory="You are fast and helpful.", llm=scripted, tools=[send_email],
                   max_iter=3, verbose=False)
    task = Task(description="Write and send a first email to Kesar Crumb Bakehouse about Pickup Parrot.",
                expected_output="A confirmation", agent=writer)
    result = Crew(agents=[writer], tasks=[task], process=Process.sequential).kickoff()

    real_name, real_email = recipient(Path("leads/kesar-crumb-bakehouse.md"))
    sent = WOULD_HAVE_SENT[0]
    print(f"  The agent's final answer:     {result.raw}")
    print(f"  send_email calls:             {len(WOULD_HAVE_SENT)}")
    print("  Human approvals before it:    0")
    print(f"  Address the model typed:      {sent['to']}   (the lead file says {real_email})")
    print(f"  What it promised:             {sent['body']}")
    print("  With a real send tool, that was a real email: wrong address, a made-up 50%, and pressure.\n")


def part_two() -> int:
    print("PART 2 - an audit of OUR outreach crew (the real agents and tools, no model is called)\n")
    crew = build_crew(llm=ScriptedLLM(responses=[]), verbose=False)
    can_send = 0
    for agent in crew.agents:
        names = [t.name for t in agent.tools] or ["(no tools)"]
        risky = [n for n in names if any(word in n.lower() for word in SENDING_WORDS)]
        can_send += len(risky)
        print(f"  {agent.role:<18} tools: {', '.join(names)}")
    print(f"\n  Tools that could send an email: {can_send}")
    print("  The send step lives OUTSIDE the crew: approve.py (a human decides), then your own mail app")
    print("  (you press Send). The model never types the address: our code copies it from the lead file.")
    return 0 if can_send == 0 else 1


if __name__ == "__main__":
    part_one()
    raise SystemExit(part_two())
