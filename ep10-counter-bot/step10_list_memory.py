"""
step10_list_memory.py - Step 10: memory by hand. A Python list, sent again on every call.

The model is stateless (Step 9). So we keep the conversation in a list and send the WHOLE list
each time: the system prompt, every customer message and every reply so far.
The token meter shows the bill: the input gets bigger on every turn.

Run:  python step10_list_memory.py      (needs your free Gemini key, 4 calls)

Author: Ravi Shanker Singh (TechWithRavi)
"""

from langchain_core.messages import HumanMessage, SystemMessage

from common import CAFE_RULES, friendly_error
from llm import MissingAPIKeyError, get_llm
from token_meter import describe, estimate_tokens, reported_tokens

CUSTOMER_LINES = [
    "Hi, I'm Asha.",
    "One masala chai, please. No nuts for me, I'm allergic.",
    "Add one khasta kachori too.",
    "What is my name, and what have I ordered so far?",
]


def chat_with_list_memory(llm, lines, write=print):
    """Talk through `lines`. Returns the history list (our whole 'memory')."""
    history = [SystemMessage(CAFE_RULES)]         # the memory: one plain Python list
    for turn, line in enumerate(lines, start=1):
        history.append(HumanMessage(line))
        sent = len(history)
        estimate = estimate_tokens(history)        # counted before sending
        reply = llm.invoke(history)                # the WHOLE list goes to the model, every time
        history.append(reply)                      # remember the answer too, or the next call cannot see it
        write(f"Turn {turn}  You: {line}")
        write(f"        Bot: {reply.text}")
        write(f"        meter: {describe(sent, estimate, reported_tokens(reply))}\n")
    return history


def main(llm=None):
    try:
        llm = llm or get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    try:
        history = chat_with_list_memory(llm, CUSTOMER_LINES)
    except Exception as error:
        print(friendly_error(error))
        return 1
    print(f"The list now holds {len(history)} messages. Every new turn sends all of them again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
