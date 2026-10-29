"""
common.py - small things the Kadak Kettle programs share.

  - where the files live
  - the cafe rules: the system prompt (fixed facts live HERE, not in the chat memory)
  - a short, friendly message when a model call fails

You do not run this file. The other programs import it.

Author: Ravi Shanker Singh (TechWithRavi). Kadak Kettle is an invented tea counter for this course.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"                         # the cafe's own files: menu.csv, about.html, policy.pdf
KITCHEN_DIR = HERE / "kitchen"                   # live data the kitchen updates every morning
SPECIALS_FILE = KITCHEN_DIR / "specials_today.csv"
INDEX_DIR = HERE / "index"                       # the Chroma index (made by cafe_index.py, git-ignored)
STATE_DIR = HERE / "state"                       # saved conversations (made by counter_bot.py --save, git-ignored)
OUTPUTS_DIR = HERE / "outputs"                   # exports (git-ignored)

CAFE_NAME = "Kadak Kettle"

# Fixed facts that rarely change go in the SYSTEM PROMPT. The model reads it on every call,
# and no memory setting can ever trim it away.
CAFE_RULES = (
    "You are the counter assistant of Kadak Kettle, a small tea counter next to a city bus stop. "
    "Be warm and brief, like friendly counter staff: at most three short sentences. "
    "Kadak Kettle is open from 7 am to 9 pm and closed on Tuesdays. Prices are in rupees. "
    "Use the customer's name when you know it. "
    "Never promise that a food is safe for an allergy. Ask the customer to tell the staff before paying."
)

# Extra rules when the bot has its tools (Step 14). Facts come from the tools, never from memory.
GROUNDING_RULES = (
    " For any question about menu items, prices, ingredients, nuts or other allergens, pickup orders or refunds, "
    "call search_cafe_docs first and answer only from what it returns. "
    "For today's specials, call todays_specials. "
    "After each fact, add its tag in square brackets, for example [menu.csv row 7]. "
    "If the tools do not have the answer, say you don't know and suggest asking the staff."
)


def friendly_error(error: Exception) -> str:
    """A short explanation of a failed model call, with the fix. Never a long traceback."""
    text = f"{type(error).__name__}: {error}"
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if "huggingface.co" in text:
        return (
            "Could not download the free local embeddings model (about 90 MB) from huggingface.co.\n"
            "It downloads once. Connect to the internet and run the command again."
        )
    if type(error).__name__ == "GraphRecursionError":
        return "The bot stopped: it kept calling tools without answering. Ask again, more simply."
    if provider == "ollama" and ("Connect" in type(error).__name__ or "refused" in text.lower()):
        return "Could not reach Ollama. Start the Ollama app, run  ollama pull qwen3.5:4b , then try again."
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            "The free-tier limit was reached. Wait a minute and try again.\n"
            "Your limits: https://aistudio.google.com/rate-limit"
        )
    return f"The model call failed: {text[:300]}"
