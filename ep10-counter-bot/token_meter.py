"""
token_meter.py - count what we send to the model on every call (Steps 10 to 15).

A model has no memory of its own. "Memory" means: we send the conversation AGAIN on every call.
This meter shows the price of that, with two numbers:

    estimate   counted on YOUR computer before sending: LangChain's count_tokens_approximately
               (about 4 characters per token, plus a little per message). Free and offline, but a rough guess.
    reported   what the provider counted, from reply.usage_metadata. Gemini fills this in on every
               reply. It also counts things the estimate does not see, like the tool descriptions.

TokenMeter is a LangChain callback: LangChain calls it every time ANY chat model starts or ends,
including the extra call the summary memory makes. So it sees exactly what the model saw.

Author: Ravi Shanker Singh (TechWithRavi)
"""

from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.utils import count_tokens_approximately


def estimate_tokens(messages, counter=count_tokens_approximately):
    """About how many tokens these messages are (no API call)."""
    return counter(list(messages))


def reported_tokens(reply):
    """(input tokens, output tokens) as the provider reported them, or None if it did not."""
    usage = getattr(reply, "usage_metadata", None)
    if not usage:
        return None
    return usage["input_tokens"], usage["output_tokens"]


def describe(sent, estimate, reported):
    """One readable line about one model call."""
    line = f"sent {sent} message{'' if sent == 1 else 's'}, about {estimate} tokens (estimate)"
    if reported:
        line += f" | model counted {reported[0]} in, {reported[1]} out"
    else:
        line += " | model counted: not reported"
    return line


@dataclass
class ModelCall:
    """One call to a chat model, as the meter saw it."""

    kind: str                       # "reply" (the bot answering) or "summary" (summary memory squeezing)
    sent: list = field(repr=False)  # the exact messages the model received, system prompt included
    estimate: int
    reported: tuple | None = None


class TokenMeter(BaseCallbackHandler):
    """Writes down every chat-model call: what was sent, and the token counts."""

    def __init__(self, counter=count_tokens_approximately):
        self.counter = counter      # tests pass a fake counter here
        self.calls = []
        self._turn_start = 0

    # LangChain calls this just before a chat model runs. messages[0] is what the model will read.
    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        sent = list(messages[0])
        kind = "summary" if (metadata or {}).get("lc_source") == "summarization" else "reply"
        self.calls.append(ModelCall(kind, sent, estimate_tokens(sent, self.counter)))

    # ... and this when it has answered. Gemini's answer carries usage_metadata.
    def on_llm_end(self, response, **kwargs):
        message = getattr(response.generations[0][0], "message", None)
        counts = reported_tokens(message)
        if counts and self.calls:
            self.calls[-1].reported = counts

    def new_turn(self):
        """Call before each customer message, so this_turn() only shows that turn's calls."""
        self._turn_start = len(self.calls)

    def this_turn(self):
        return self.calls[self._turn_start:]

    def turn_lines(self):
        """Text lines about this turn's calls, for counter_bot.py.

        A turn can have several calls: with tools the model is called again after each tool result,
        and summary memory adds one summary call when it squeezes.
        """
        calls = self.this_turn()
        replies = [call for call in calls if call.kind == "reply"]
        lines = []
        number = 0
        for call in calls:
            if call.kind == "summary":
                label = "summary call"
            else:
                number += 1
                label = "model call" if len(replies) == 1 else f"model call {number}"
            lines.append(f"{label}: {describe(len(call.sent), call.estimate, call.reported)}")
        return lines

    def total_estimate(self, kind=None):
        """All estimated input tokens so far (only of one kind if given)."""
        return sum(call.estimate for call in self.calls if kind is None or call.kind == kind)
