"""
decoder_cloud.py - the same prompt through Gemini, a CLOSED model behind an API (step 6).

Gemini's weights never leave Google's servers. We send text over the internet and get
text back, and Google does not publish how many parameters the model has.
At the end we put its reply next to the small local model's reply from step 5.

Step 12: put LLM_PROVIDER=ollama in the root .env file and run this file again.
The same code then answers from a model on your own computer.

Run:  python decoder_cloud.py      (needs your free Gemini key in the root .env file)
"""

import json
import re
import time

import httpx   # the web library the Ollama client uses (installed with LangChain)
from langchain_core.messages import HumanMessage, SystemMessage

from common import LOCAL_RESULT_FILE, OWNER_PROMPT, OWNER_SYSTEM, console, make_table
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

# Where each provider's model runs, and what it needs.
PROVIDER_FACTS = {
    "gemini": ("Google's servers", "closed (stays with Google)", "not published", "internet + free key"),
    "openai": ("OpenAI's servers", "closed (stays with OpenAI)", "not published", "internet + paid key"),
    "anthropic": ("Anthropic's servers", "closed (stays with Anthropic)", "not published", "internet + paid key"),
    "ollama": ("this computer (Ollama)", "open (a file on your disk)", "see ollama.com/library", "the Ollama app, no key"),
}

# A rough check, not a real quality score: does the reply talk about the customer's actual problem?
PROBLEM_WORDS = {"coffee", "lukewarm", "hot", "warm", "temperature", "sticky", "cup", "clean", "wipe"}


def ask(llm, system: str = OWNER_SYSTEM, prompt: str = OWNER_PROMPT) -> str:
    """Send the chat to the model and return the reply as plain text."""
    reply = llm.invoke([SystemMessage(system), HumanMessage(prompt)])
    return reply.text   # .text, not .content: Gemini 3 returns content as a list of blocks


def mentions_problem(reply: str) -> str:
    """'yes (coffee, sticky)' if the reply names the real problem, else 'no'."""
    found = sorted(PROBLEM_WORDS & set(re.findall(r"[a-z]+", reply.lower())))
    return f"yes ({', '.join(found)})" if found else "no"


def comparison_rows(local: dict | None, cloud: dict) -> list[list[str]]:
    """Rows for the side-by-side table: [what we compare, local model, cloud model]."""
    if local is None:
        local = {"model": "run decoder_local.py first", "parameters": None, "seconds": None, "reply": ""}
    runs_on, weights, parameters, needs = PROVIDER_FACTS[cloud["provider"]]
    return [
        ["Model", local["model"], cloud["model"]],
        ["Runs on", "this computer", runs_on],
        ["Weights", "open (a file on your disk)", weights],
        ["Parameters", f"{local['parameters']:,}" if local["parameters"] else "-", parameters],
        ["Needs", "nothing after the download", needs],
        ["Time for this reply", f"{local['seconds']} s" if local["seconds"] is not None else "-", f"{cloud['seconds']} s"],
        ["Names the real problem?", mentions_problem(local["reply"]) if local["reply"] else "-",
         mentions_problem(cloud["reply"])],
        ["Reply", local["reply"] or "-", cloud["reply"]],
    ]


def main(llm=None) -> int:
    try:
        provider = get_provider()              # reads LLM_PROVIDER from the root .env (gemini by default)
        model = get_model_name(provider)
        print(f"Provider: {provider}   Model: {model}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")        # friendly help (no key, or a typo in LLM_PROVIDER)
        return 1

    print(f"\nPrompt: {OWNER_PROMPT}")
    start = time.perf_counter()
    try:
        reply = ask(llm)
    except (ConnectionError, httpx.ConnectError) as error:   # no internet, or Ollama is not running
        print(f"\nCould not reach the model: {error}")
        if provider == "ollama":
            print("Start the Ollama app (https://ollama.com), run 'ollama pull qwen3.5:4b', then try again.")
        else:
            print("Check your internet connection, then try again.")
        return 1
    seconds = round(time.perf_counter() - start, 1)
    print(f"\nReply ({seconds} s):\n{reply}\n")

    local = json.loads(LOCAL_RESULT_FILE.read_text(encoding="utf-8")) if LOCAL_RESULT_FILE.exists() else None
    cloud = {"provider": provider, "model": model, "seconds": seconds, "reply": reply}
    table = make_table(["", "small local model (step 5)", f"{provider} model"],
                       comparison_rows(local, cloud), title="Same prompt, two models")
    console.print(table)
    print("Which reply would you put on the shop's page? That judgement is yours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
