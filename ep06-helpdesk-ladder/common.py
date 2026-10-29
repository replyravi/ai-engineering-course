"""
common.py - small things every rung of the Help Desk Ladder shares.

  - where the data files are
  - the house rules (the system prompt of our help desk)
  - turning a model error into a short, friendly message
  - one line that tells you if LangSmith tracing is on (Step 14)

You do not run this file. The rung scripts import it.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
HANDBOOK_FILE = HERE / "data" / "handbook.md"
HOLIDAYS_FILE = HERE / "data" / "holidays.csv"

# The house rules: who the assistant is and how it talks. Every rung starts with these.
# (Dabbaloop is a company we invented for this course.)
HOUSE_RULES = (
    "You are the help desk assistant of Dabbaloop, a meal-box delivery service in Pune. "
    "Start every reply with 'Namaste!'. Be warm and brief: at most three short sentences. "
    "Write numbers as digits."
)

# The exact sentence the grounded rungs use when the handbook has no answer.
REFUSAL = "I don't know based on the handbook."


def is_refusal(text: str) -> bool:
    """True when an answer says 'I don't know' (in any of the ways our prompts ask for)."""
    text = text.lower().replace("’", "'")   # models often write a curly apostrophe: don’t
    return "don't know" in text or "do not know" in text


def friendly_error(error: Exception) -> str:
    """A short explanation of a failed model call, with the fix. Never a long traceback."""
    text = f"{type(error).__name__}: {error}"
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if "huggingface.co" in text:
        # Not a chat model problem: the free local embeddings model could not be downloaded.
        return (
            "Could not download the free local embeddings model (about 90 MB) from huggingface.co.\n"
            "It downloads once, the first time you search the handbook. Connect to the internet and run\n"
            f"this script again. ({text.splitlines()[0]})"
        )
    if type(error).__name__ == "GraphRecursionError":
        return (
            "The agent stopped: it used all its steps without a final answer (our safety limit,\n"
            "MAX_AGENT_STEPS in rung5_tool.py). The model kept calling tools. Ask a clearer question,\n"
            "or try LLM_MODEL=gemini-3.8-flash in .env."
        )
    if provider == "ollama" and ("Connect" in type(error).__name__ or "refused" in text.lower()):
        return (
            f"Could not reach Ollama ({text}).\n"
            "Ollama runs models on your own computer, and it is not running yet:\n"
            "  1. Install it from https://ollama.com/download and start it\n"
            "  2. In a terminal:  ollama pull qwen3.5:4b\n"
            "  3. Run this script again"
        )
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (
            f"The free-tier limit was reached ({text[:120]}...).\n"
            "Wait a minute and try again. Your limits: https://aistudio.google.com/rate-limit"
        )
    return f"The model call failed: {text}"


def tracing_status() -> str:
    """One line about LangSmith tracing (Step 14). Never prints the key itself."""
    value = os.getenv("LANGSMITH_TRACING", "")
    if value != "true":
        if value.strip().lower() == "true":
            # LangSmith switches on only for the exact word true. True or TRUE traces nothing.
            return f"LangSmith tracing: off (write LANGSMITH_TRACING=true in small letters, not {value.strip()})"
        return "LangSmith tracing: off"
    project = os.getenv("LANGSMITH_PROJECT", "").strip() or "default"
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        return "LangSmith tracing: ON, but LANGSMITH_API_KEY is missing in .env (no trace will be saved)"
    return f"LangSmith tracing: ON (project: {project}) - open https://smith.langchain.com to see the run"
