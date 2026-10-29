"""
step09_two_calls.py - Step 9: two separate calls to the model. Does the second call remember the first?

Each call to a chat model is a brand-new request. The model keeps nothing between calls:
it is STATELESS. Whatever it should "remember", your program has to send again.

Run:  python step09_two_calls.py        (needs your free Gemini key, 2 calls)

Author: Ravi Shanker Singh (TechWithRavi)
"""

from common import friendly_error
from llm import MissingAPIKeyError, get_llm
from token_meter import reported_tokens

FIRST = "Hi, I'm Asha. Please remember my name."
SECOND = "What is my name?"


def ask_once(llm, text):
    """One call. The model receives exactly ONE message: this text. Nothing else."""
    return llm.invoke(text)


def main(llm=None):
    try:
        llm = llm or get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())       # friendly help instead of a traceback
        return 1

    try:
        for number, text in enumerate([FIRST, SECOND], start=1):
            reply = ask_once(llm, text)
            print(f"Call {number}  You:   {text}")
            print(f"        Model: {reply.text}")        # .text, not .content (Gemini 3+ sends a list of blocks)
            counts = reported_tokens(reply)
            print(f"        (this call sent 1 message; input tokens: {counts[0] if counts else 'not reported'})\n")
    except Exception as error:
        print(friendly_error(error))
        return 1
    print("Call 2 carried only its own message, so the model had no way to know your name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
