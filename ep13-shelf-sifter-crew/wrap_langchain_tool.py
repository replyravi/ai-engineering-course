"""
wrap_langchain_tool.py - Step 6 (extra): use a LangChain tool inside a CrewAI agent.

CrewAI is its own framework: it does not run on top of LangChain. Its agents accept only
CrewAI tools, so a tool written for LangChain (like the ones in Episodes 06 and 11) must be
wrapped first with Tool.from_langchain. The tool keeps its name, description and arguments.

    python wrap_langchain_tool.py      (no key, no internet: nothing is sent to a model)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

from crewai import LLM, Agent
from crewai.tools.base_tool import Tool
from langchain_core.tools import tool as langchain_tool
from pydantic import ValidationError


@langchain_tool
def price_per_star(price: float, rating: int) -> float:
    """Pounds paid for each star of rating. Lower means better value."""
    return round(price / rating, 2)


def make_agent(tools: list) -> Agent:
    # Building an agent sends nothing, so no key is needed for this demo.
    return Agent(role="Value Analyst", goal="Find good value", backstory="Careful with money.",
                 tools=tools, llm=LLM(model="gemini/gemini-3.5-flash-lite"))


def main() -> int:
    print("1. A LangChain tool, given to a CrewAI agent as it is:")
    try:
        make_agent([price_per_star])
        print("   accepted (a different CrewAI version?)")
    except ValidationError as error:
        problem = error.errors()[0]           # pydantic says which field failed, and why
        print(f"   rejected: {problem['msg']} (field: {'.'.join(map(str, problem['loc']))}, "
              f"got a {type(price_per_star).__name__})")

    print("\n2. The same tool, wrapped with Tool.from_langchain:")
    crew_tool = Tool.from_langchain(price_per_star)
    make_agent([crew_tool])
    print(f"   accepted: {type(crew_tool).__name__} named {crew_tool.name!r}")
    print(f"   description: {crew_tool.description}")
    print(f"   crew_tool.run(price=26.08, rating=5) -> {crew_tool.run(price=26.08, rating=5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
