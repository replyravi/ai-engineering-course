"""
rung1_messages.py - Rung 1: talk to a chat model with MESSAGES (Steps 2 and 3).

A chat model reads a conversation, not one loose string. Every message has a role:
    system  the rules: who the assistant is and how it talks
    human   what the customer asks
    ai      what the model answers (the reply we get back)
We print the reply with .text and look at the token usage the provider reports.

Run:  python rung1_messages.py
      python rung1_messages.py "Explain in two sentences what a delivery slot is."
Step 3: change LLM_PROVIDER in the root .env (for example to ollama) and run it again.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

from common import HOUSE_RULES, friendly_error
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

QUESTION = "Explain in two sentences what a meal-box subscription is."


def build_messages(question: str) -> list:
    """The conversation we send: the house rules first, then the customer's question."""
    return [
        SystemMessage(HOUSE_RULES),
        HumanMessage(question),
    ]


def model_id(reply) -> str:
    """The model name the provider says it used (each provider files it a little differently)."""
    meta = reply.response_metadata or {}
    return meta.get("model_name") or meta.get("model") or "not reported"


def main(argv=None, llm=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    question = " ".join(argv).strip() or QUESTION

    messages = build_messages(question)
    print(f"Sending {len(messages)} messages:")
    for message in messages:
        print(f"  {message.type:<6} {message.text}")

    try:
        provider = get_provider()                # reads LLM_PROVIDER from the root .env
        print(f"\nProvider: {provider}   Model: {get_model_name(provider)}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")         # friendly help, not a traceback
        return 1

    start = time.perf_counter()
    try:
        reply = llm.invoke(messages)
    except Exception as error:                   # no internet, Ollama not running, free-tier limit ...
        print(f"\n{friendly_error(error)}")
        return 1
    seconds = time.perf_counter() - start

    # .text, not .content: Gemini 3 returns content as a list of blocks.
    print(f"\nReply ({reply.type}):\n{reply.text}\n")
    print(f"Model id reported by the provider: {model_id(reply)}")
    print(f"Token usage: {reply.usage_metadata}")   # a plain dict: input, output and total tokens
    print(f"Time: {seconds:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
