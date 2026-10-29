"""
memory_modes.py - three kinds of chatbot memory, built the 2026 way (Steps 11 to 13).

    buffer    remember every turn                         (classic: ConversationBufferMemory)
    window    remember only the last 2 exchanges          (classic: ConversationBufferWindowMemory(k=2))
    summary   squeeze old turns into one short note       (classic: ConversationSummaryMemory)

All three are the SAME agent: create_agent + a checkpointer. The checkpointer saves the messages
of each conversation under its thread_id (one thread per customer). Window and summary add one
middleware: a small function that runs just before every model call and edits the saved messages.

    python memory_modes.py --window-demo     Step 12: a window that cuts in the wrong place, then the fix
                                             (no API key needed, no model is called)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import os

# Load saved conversations safely: LangGraph then only rebuilds known, safe types from a saved
# file (its security notes recommend this switch). It must be set before LangGraph is imported.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langchain.agents import AgentState, create_agent  # noqa: E402
from langchain.agents.middleware import SummarizationMiddleware, before_model  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, trim_messages  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES  # noqa: E402

from common import CAFE_RULES  # noqa: E402

MODES = ("buffer", "window", "summary")

# Window: keep the new question plus the 2 exchanges (question + answer) before it.
WINDOW_EXCHANGES = 2

# Summary: once the saved conversation reaches 8 messages, squeeze all but the newest 4 into one note.
SUMMARY_TRIGGER_MESSAGES = 8
SUMMARY_KEEP_MESSAGES = 4
SUMMARY_PROMPT = """Write a short note about this cafe conversation for the next person at the counter.
Keep: the customer's name, every allergy, the current order with quantities, and any pickup time.
Leave out greetings and small talk. At most 60 words. Write only the note.

Conversation:
{messages}"""


# ---------------------------------------------------------------------------
# Window memory: trim_messages, always starting on a customer (human) message
# ---------------------------------------------------------------------------

def count_turns(messages):
    """Our 'token counter' for trim_messages: it counts customer messages, not tokens.

    trim_messages keeps as much as fits under max_tokens, measured by the counter you give it.
    len counts messages, count_tokens_approximately counts tokens, and this one counts turns.
    """
    return sum(1 for message in messages if message.type == "human")


def last_exchanges(messages, k=WINDOW_EXCHANGES, start_on="human"):
    """The new question plus the k exchanges before it.

    start_on="human" makes the window begin with a customer message. Without it, the window can
    begin with an old REPLY whose question was cut off (see python memory_modes.py --window-demo).
    """
    return trim_messages(messages, max_tokens=k + 1, token_counter=count_turns,
                         strategy="last", start_on=start_on)


@before_model
def keep_recent_turns(state: AgentState, runtime) -> dict | None:
    """Runs before every model call: forget everything older than the window."""
    messages = state["messages"]
    recent = last_exchanges(messages)
    if len(recent) == len(messages):
        return None                                 # nothing to forget yet
    # Replace the saved messages with the recent ones (the old ones are deleted, not just hidden).
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent]}


# ---------------------------------------------------------------------------
# Summary memory: SummarizationMiddleware
# ---------------------------------------------------------------------------

def summary_memory(llm, summary_llm=None):
    """Squeeze old messages into one summary message. Each squeeze is one EXTRA model call.

    summary_llm: the model that writes the note. None = the bot's own model.
    """
    return SummarizationMiddleware(
        model=llm,
        trigger=("messages", SUMMARY_TRIGGER_MESSAGES),     # when to squeeze (no trigger = never)
        keep=("messages", SUMMARY_KEEP_MESSAGES),           # how many recent messages stay word for word
        summary_prompt=SUMMARY_PROMPT,
        summarizer=summary_llm,
    )


def is_summary(message):
    """True for the note SummarizationMiddleware wrote ('Here is a summary of the conversation to date')."""
    return message.additional_kwargs.get("lc_source") == "summarization"


# ---------------------------------------------------------------------------
# One builder for all three
# ---------------------------------------------------------------------------

def build_bot(mode="buffer", llm=None, checkpointer=None, tools=(), system_prompt=CAFE_RULES, summary_llm=None):
    """The counter bot with the memory you ask for.

    llm:          the chat model. None = the real one from llm.py (tests pass fake_llm(...)).
    checkpointer: where conversations are saved. None = InMemorySaver (in RAM, gone when the program stops).
    tools:        the tools the bot may call (Step 14). None by default.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown memory mode {mode!r}. Use one of: {', '.join(MODES)}")
    if llm is None:
        from llm import get_llm

        llm = get_llm()
    middleware = []
    if mode == "window":
        middleware = [keep_recent_turns]
    elif mode == "summary":
        middleware = [summary_memory(llm, summary_llm)]
    return create_agent(
        model=llm,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=middleware,          # buffer: none. Window and summary add one (Steps 12 and 13)
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# Step 12: the window bug, with no model at all
# ---------------------------------------------------------------------------

DEMO_CHAT = [
    HumanMessage("Hi, I'm Asha."),
    AIMessage("Hello Asha! What can I get you?"),
    HumanMessage("One masala chai, please."),
    AIMessage("One masala chai coming up."),
    HumanMessage("Add a vada pav."),
    AIMessage("Done: one masala chai and one vada pav."),
    HumanMessage("Who am I?"),
]


def show(messages):
    for message in messages:
        print(f"    {message.type:<5} {message.text}")


def window_demo():
    print(f"The whole chat ({len(DEMO_CHAT)} messages, the last one is the new question):")
    show(DEMO_CHAT)
    wrong = last_exchanges(DEMO_CHAT, start_on=None)
    print(f"\nWindow of {WINDOW_EXCHANGES} exchanges WITHOUT start_on ({len(wrong)} messages):")
    show(wrong)
    print("    -> it starts with a reply that uses the name Asha, but the message that gave the name is gone.")
    right = last_exchanges(DEMO_CHAT)
    print(f'\nWindow of {WINDOW_EXCHANGES} exchanges WITH start_on="human" ({len(right)} messages):')
    show(right)
    print("    -> it starts on a customer message. The name is simply gone, so the model can say it does not know.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Memory modes for the counter bot.")
    parser.add_argument("--window-demo", action="store_true", help="show the window cut, wrong and right")
    args = parser.parse_args(argv)
    if args.window_demo:
        window_demo()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
