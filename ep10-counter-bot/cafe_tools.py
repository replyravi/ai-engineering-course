"""
cafe_tools.py - the counter bot's two tools (Step 14). Both are read-only.

    search_cafe_docs   RETRIEVAL: searches the menu, about page and policy (the index from cafe_index.py)
    todays_specials    a TOOL for live data: reads kitchen/specials_today.csv at the moment it is asked

Why not just put the menu in the bot's memory? Memory is for the conversation. Facts that live in
files are looked up when needed, so the answer can cite the row or page, and an updated file is
used at once. The model only sees each tool's name, its docstring and its arguments.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import csv
from functools import lru_cache

from langchain.tools import tool

from cafe_index import citation, load_index, search
from common import SPECIALS_FILE

TOOL_K = 4      # chunks per search. For "Is the kachori nut-free?" the "kitchen is not nut-free" rule ranks 4th.


@lru_cache(maxsize=1)
def _store():
    """Open the index once and reuse it (loading the embeddings model takes a few seconds)."""
    return load_index()


def warm_up():
    """Open the index now (counter_bot.py calls this at start), so the first question is not slow."""
    _store()


def format_hits(hits):
    """Each chunk with its citation tag in front, the way the model should cite it."""
    return "\n\n".join(f"[{citation(doc)}]\n{doc.page_content}" for doc, _distance in hits)


@tool
def search_cafe_docs(question: str) -> str:
    """Search Kadak Kettle's menu, about page and policy.

    Use it for menu items, prices, ingredients, nuts and other allergens, pickup orders and refunds.
    Returns the best matching passages, each starting with its citation tag in square brackets.
    """
    try:
        return format_hits(search(_store(), question, k=TOOL_K))
    except Exception as error:      # return the problem as text: an exception would stop the whole agent
        return f"Error: could not search the cafe files ({error})."


def read_specials(path=SPECIALS_FILE):
    """Today's specials as text lines (plain Python, easy to test)."""
    try:
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except OSError as error:
        return f"Error: could not read today's specials ({error})."
    if not rows:
        return "No specials today."
    return "\n".join(f"{r['special']}: Rs {r['price_rs']}, contains nuts: {r['contains_nuts']}, {r['note']}"
                     for r in rows)


@tool
def todays_specials() -> str:
    """Today's specials from the kitchen, with prices. The list changes every morning."""
    return read_specials()


TOOLS = [search_cafe_docs, todays_specials]
