"""
first_call.py - your first call to an AI model (step 5).

get_llm() reads LLM_PROVIDER from the .env file (gemini by default) and returns a chat model.
We send one message, print the answer with .text, and show how many tokens it used.

Run:  python first_call.py
Try:  LLM_PROVIDER=openai in .env (with no OpenAI key) to see the friendly help message.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage

from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

QUESTION = "In exactly two sentences, explain to a college student what a large language model is."


def main(llm=None) -> int:
    """Send one question to the chat model and print the answer plus token counts.

    Why this exists: step 5 is the first proof that the key and llm.py work,
    before any feedback sorting.

    What it does: get_provider() and get_model_name() from llm.py, then
    get_llm(), then llm.invoke with a system message and a human message.
    Prints reply.text (plain text). reply.content on Gemini 3 is a list of
    blocks, so this file does not print .content.
    Returns 1 if the key is missing or LLM_PROVIDER is unknown.
    Tests pass llm=fake_llm(...) and never call the real provider.

    Called by: the bottom of this file, and tests/test_genai.py.
    """
    try:
        provider = get_provider()          # reads LLM_PROVIDER from .env
        print(f"Provider: {provider}   Model: {get_model_name(provider)}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")   # friendly help (no key, or a typo in LLM_PROVIDER), not a traceback
        return 1

    messages = [
        SystemMessage("You are a patient teacher. Use simple words."),
        HumanMessage(QUESTION),
    ]
    start = time.perf_counter()
    reply = llm.invoke(messages)
    seconds = time.perf_counter() - start

    print(f"\nYou asked: {QUESTION}")
    # .text, not .content: Gemini 3 returns content as a list of blocks.
    print(f"\nAI replied:\n{reply.text}\n")

    usage = reply.usage_metadata   # how many tokens went in and came out
    if usage:
        print(f"Tokens: {usage['input_tokens']} in, {usage['output_tokens']} out, {usage['total_tokens']} total")
    else:
        print("Tokens: not reported by this model")
    print(f"Time: {seconds:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
