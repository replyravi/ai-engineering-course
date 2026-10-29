"""
crew.py - Steps 7, 10, 11 and 12: the Shelf Sifter crew.

Three agents work one after the other (Process.sequential), like a relay team:

  Scout    reads one shelf of the practice bookshop with scrape_shelf     -> a JSON list of books
  Analyst  picks the best-value books for the reader (find_similar_books) -> a typed Picks object
  Writer   turns the picks into a short Markdown buying guide              -> reports/<shelf>_guide.md

Every builder takes llm=None: the real model from llm.py (Gemini by default), or a fake one in tests.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

from crewai import Agent, Crew, Process, Task

from llm import get_crewai_llm
from models import Picks
from tools import FindSimilarBooksTool, ScrapeShelfTool, Shelf

STAGES = ["scout", "analyst", "writer"]      # build_crew(stop_after=...) can stop early (Steps 7 and 10)
GUIDE_FILE = "reports/{category}_guide.md"   # CrewAI fills in {category} from the kickoff inputs
PRACTICE_NOTE = ("_Prices and ratings come from books.toscrape.com, a practice site "
                 "where they were assigned at random._")


def make_agents(llm, shelf: Shelf, fetch=None, embed=None, verbose: bool = True) -> dict:
    """The three team members. role + goal + backstory = the job description the model reads."""

    def model():
        # Each agent gets its OWN crewai.LLM object (Gemini by default; LLM_PROVIDER in .env switches it).
        # CrewAI adds up the token usage of every agent's LLM object, so one object shared by three
        # agents would be counted three times. Tests pass one scripted fake, which counts nothing.
        return llm if llm is not None else get_crewai_llm()
    scrape_tool = ScrapeShelfTool(
        shelf=shelf,
        result_as_answer=True,    # the tool's JSON IS the Scout's answer: no model retypes 11 prices
        **({"fetch": fetch} if fetch else {}),
    )
    similar_tool = FindSimilarBooksTool(shelf=shelf, embed=embed)

    scout = Agent(
        role="Shelf Sifter",
        goal="Read the {category} shelf of the practice bookshop and report its books exactly as scraped",
        backstory="You are careful and polite. You only report what the scrape_shelf tool returned. "
                  "You never invent a book, a price or a rating.",
        tools=[scrape_tool],
        llm=model(),              # Gemini by default, from get_crewai_llm(); never leave it out
        max_iter=3,               # at most 3 think-act-observe laps, then it must answer
        allow_delegation=False,   # the Scout does its own job (False is also CrewAI's default)
        verbose=verbose,          # print every step in the terminal
    )
    analyst = Agent(
        role="Value Analyst",
        goal="Choose the best-value books on the {category} shelf for a reader who likes {interest}",
        backstory="You shop like a careful student: a high star rating for a low price wins. "
                  "You check which titles match the reader with the find_similar_books tool. "
                  "You copy titles, prices, ratings and links exactly; you never change a number.",
        tools=[similar_tool],
        llm=model(),
        max_iter=3,
        allow_delegation=False,
        verbose=verbose,
    )
    writer = Agent(
        role="Guide Writer",
        goal="Write a short, friendly buying guide in Markdown",
        backstory="You write clear guides for students. Every book title is a link to its page, "
                  "and every price and rating is copied exactly from the picks you receive.",
        tools=[],                 # the Writer only writes
        llm=model(),
        max_iter=2,
        allow_delegation=False,
        verbose=verbose,
    )
    return {"scout": scout, "analyst": analyst, "writer": writer}


def make_tasks(agents: dict) -> dict:
    """The three work tickets. expected_output tells the agent what "done" looks like."""
    scout_task = Task(
        description="Call the scrape_shelf tool once, with this url: {shelf_url}",
        expected_output="The JSON list of books that scrape_shelf returned. "
                        "Each book has title, price, rating, stock and url.",
        agent=agents["scout"],
    )
    analyst_task = Task(
        description="The reader likes: {interest}.\n"
                    "1. Call find_similar_books once with the reader's interest.\n"
                    "2. From the Scout's list choose up to 3 best-value books for this reader. Best value "
                    "means a high star rating for a low price: prefer 4 or 5 stars, and between equal "
                    "stars the lower price wins. A book that find_similar_books ranked high wins a tie.\n"
                    "Copy title, price, rating and url exactly from the Scout's list. Category: {category}.",
        expected_output="Up to 3 picks, best value first, each with title, price, rating, url and a "
                        "one-sentence reason, plus the category and the reader's interest.",
        agent=agents["analyst"],
        context=[scout_task],      # the Scout's answer is pasted into this task's prompt
        output_pydantic=Picks,     # CrewAI turns the answer into a Picks object (.pydantic)
    )
    writer_task = Task(
        description="Write a buying guide for the {category} shelf for a reader who likes {interest}.\n"
                    "Start with the heading '# Best-value {category} books'. For each pick write a level-2 "
                    "heading with the title as a Markdown link to its url, then one line with the price "
                    "in pounds and the stars, then the reason. Use only the picks you received.\n"
                    f"End with this line, exactly: {PRACTICE_NOTE}\n"
                    "Answer with the Markdown only, without code fences.",
        expected_output="A Markdown guide: one heading, then one section per pick with a linked title, "
                        "price, stars and reason, then the practice-site note.",
        agent=agents["writer"],
        context=[analyst_task],    # the Writer receives the typed picks
        output_file=GUIDE_FILE,    # "reports/{category}_guide.md": CrewAI saves the Writer's answer here
    )
    return {"scout": scout_task, "analyst": analyst_task, "writer": writer_task}


def build_crew(llm=None, shelf: Shelf | None = None, stop_after: str = "writer",
               fetch=None, embed=None, verbose: bool = True) -> Crew:
    """Build the crew. stop_after="scout" gives a one-agent crew (Step 7), "analyst" two agents."""
    shelf = shelf if shelf is not None else Shelf()

    agents = make_agents(llm, shelf, fetch=fetch, embed=embed, verbose=verbose)
    tasks = make_tasks(agents)
    keep = STAGES[: STAGES.index(stop_after) + 1]

    return Crew(
        agents=[agents[name] for name in keep],
        tasks=[tasks[name] for name in keep],
        process=Process.sequential,     # one after the other; each task can read the earlier answers
        max_rpm=10,                     # at most 10 model requests per minute (protects your free quota)
        tool_failure_policy="raise",    # a failed tool call STOPS the crew instead of letting it guess
        verbose=verbose,
    )
