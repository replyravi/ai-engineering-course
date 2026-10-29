"""
router.py - Step 11: the LLM router. The model fills in a Route form: destination + reason.

    python router.py --show-schema                     the form the model must fill in (no key needed)
    python router.py "How many Jain thalis were sold last week?"
    python router.py "q1" "q2" "q3"                    several questions, one model call each

Like a railway signal box: it reads each train's label and sets the points. It does not answer
the question. It only picks the line: policy, orders or chat.

  - Fixed labels:  destination is a Literal, so the model can only answer one of the three words.
  - A reason:      one short line, printed every time. When a route is wrong, the reason shows you why.
  - A fallback:    if the router call fails (bad output, a network error), with_fallbacks sends the
                   question to chat, the one destination that cannot touch any data.
  - The cost:      one extra model call per question, before the real answer.

Classic names: MultiPromptChain and LLMRouterChain did this with a JSON-in-text prompt and a parser,
and never showed you the decision. Both are deprecated since LangChain 0.2.12 (see classic_demo.py).

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from llm import MissingAPIKeyError, get_llm

ROUTES = ["policy", "orders", "chat"]
FALLBACK_ROUTE = "chat"        # the safe default: chat cannot read the handbook or the database


class Route(BaseModel):
    """Where the Dabbaloop help desk should send this question."""

    destination: Literal["policy", "orders", "chat"] = Field(description="the one desk that should answer")
    reason: str = Field(description="why, in at most 12 words")


ROUTER_RULES = """You are the router of the Dabbaloop help desk. Dabbaloop delivers meal boxes (thalis) in Pune.
Do not answer the question. Pick the ONE desk that should answer it:

- policy: rules, prices and how things work, from the Dabbaloop handbook. Refunds, delivery zones and
  slots, cut-off times, meal plans and their prices, pausing, the steel box deposit, food safety,
  allergens, support hours.
- orders: numbers about what was actually ordered, from the orders database. How many were sold,
  revenue, the most popular item, orders per day, week, zone or meal.
- chat: greetings, thanks, jokes and anything that is not about Dabbaloop.

A question about a rule or a price is policy, even when it says "how many" or "how much".
If you are not sure, pick chat."""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([("system", ROUTER_RULES), ("human", "{question}")])


def fallback_route(_inputs):
    """Used when the router itself fails: send the question somewhere safe, and say so."""
    return Route(destination=FALLBACK_ROUTE, reason="fallback: the router call failed")


def build_llm_router(llm=None):
    """{"question": str} -> Route. One model call."""
    if llm is None:
        llm = get_llm()
    classify = ROUTER_PROMPT | llm.with_structured_output(Route)
    return classify.with_fallbacks([RunnableLambda(fallback_route)])


class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls (and tokens, when the provider reports them) in one run."""

    def __init__(self):
        self.calls = 0
        self.tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.calls += 1

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.tokens += usage.get("total_tokens", 0)


def route_tag(route):
    """The line we print for every question, like: route=orders  (asks for a count of thalis sold)"""
    return f"route={route.destination}  ({route.reason})"


def main():
    parser = argparse.ArgumentParser(description="Route help-desk questions with structured output.")
    parser.add_argument("questions", nargs="*", default=["How many Jain thalis were sold last week?"])
    parser.add_argument("--show-schema", action="store_true", help="print the Route form (JSON schema)")
    args = parser.parse_intermixed_args()

    if args.show_schema:
        print(json.dumps(Route.model_json_schema(), indent=2))
        return
    try:
        router = build_llm_router()
    except MissingAPIKeyError as error:
        print(str(error).strip())
        raise SystemExit(1)
    for question in args.questions:
        route = router.invoke({"question": question})
        print(f"{question}\n  {route_tag(route)}")


if __name__ == "__main__":
    main()
