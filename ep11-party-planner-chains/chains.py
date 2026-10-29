"""
chains.py - the six small chains of the Party Planner (Steps 2, 3, 4, 5 and 9).

A chain is a pipe of steps. Each step takes something in and hands something on:

    prompt          dict of named slots  -> messages
    model           messages             -> an AI message (text + metadata)
    parser          AI message           -> plain text   (or a typed object, Step 5)

    menu_chain        = MENU_PROMPT | llm | StrOutputParser()                      Step 2
    notes_chain       = NOTES_PROMPT | llm | StrOutputParser()                     Step 4
    ingredients_chain = INGREDIENTS_PROMPT | llm.with_structured_output(...)       Step 5
    swaps_chain, invite_chain, checklist_chain                                     Steps 9-10

Every builder takes llm=None: the real model from llm.py, or a fake one in tests.

Run:  python chains.py menu                    (Step 2)
      python chains.py menu --show-prompt      (the filled prompt only, no key needed)
      python chains.py raw                     (Step 3: without the parser, then with it)
      python chains.py notes                   (Step 4: the menu is handed to a second chain)
      python chains.py ingredients             (Step 5: a typed grocery list)
      python chains.py ingredients --show-schema   (the form the model fills in, no key needed)
      python chains.py parallel                (Step 9: invite and checklist side by side)
      Add --request "Kerala sadya for a housewarming" --guests 20 to plan your own party.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
import os
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from costing import EmptyListError, Ingredients, load_prices
from llm import MissingAPIKeyError, get_llm

REQUEST = "South Indian lunch"
GUESTS = 12

# ---------------------------------------------------------------------------
# 1. The prompts. Every {slot} is filled from the dict we pass to invoke().
#    Every prompt ends with a human message (Gemini rejects a last AI message).
# ---------------------------------------------------------------------------

MENU_SYSTEM = ("You plan home-cooked party menus in India. Reply with exactly four dishes, one per line, "
               "written as 'Dish - a few words about it'. No prices, no quantities, no extra text.")
MENU_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MENU_SYSTEM),
    ("human", "Party: {request}\nGuests: {guests}"),
])

NOTES_SYSTEM = ("You are a friendly home cook. For each dish in the menu, write one line: the dish name, "
                "then one practical tip, like what to start first or prepare the day before.")
NOTES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NOTES_SYSTEM),
    ("human", "Menu:\n{menu}"),
])

INGREDIENTS_SYSTEM = ("You turn a party menu into a grocery list. Give the amount ONE guest needs. "
                      "Use item names from this price list, spelled exactly as written, with the unit shown:\n"
                      "{price_list}\n"
                      "If the menu needs something that is not in the list, still add it with a short "
                      "plain name. Leave out water and salt.")
INGREDIENTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INGREDIENTS_SYSTEM),
    ("human", "Menu:\n{menu}"),
])

SWAPS_SYSTEM = ("You help a host cut the grocery bill of a party. Suggest up to three swaps, one per line, "
                "written as 'Swap X for Y - why it still works'. Do NOT write any prices, amounts or "
                "totals: our code does all the maths.")
SWAPS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SWAPS_SYSTEM),
    ("human", "Menu:\n{menu}\n\nThe bill is over budget by {over_by}. "
              "The most expensive items are: {expensive}."),
])

INVITE_SYSTEM = ("You write short, warm party invitations for a messaging app, at most 60 words. "
                 "Leave [date], [time] and [address] as blanks for the host to fill in. Do not invent names.")
INVITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INVITE_SYSTEM),
    ("human", "Party: {request}\nMenu:\n{menu}"),
])

CHECKLIST_SYSTEM = ("You help a host get ready. Write a checklist of exactly six short lines, in time order, "
                    "from the day before the party to serving time. Start each line with '[ ]'.")
CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CHECKLIST_SYSTEM),
    ("human", "Party: {request}\nGuests: {guests}\nMenu:\n{menu}"),
])

# Which chain sent a prompt (used to label model calls in Step 8 and in the tests).
CHAIN_NAMES = {MENU_SYSTEM: "menu", NOTES_SYSTEM: "notes", INGREDIENTS_SYSTEM.split("{")[0]: "grocery list",
               SWAPS_SYSTEM: "swaps", INVITE_SYSTEM: "invite", CHECKLIST_SYSTEM: "checklist"}


# ---------------------------------------------------------------------------
# 2. The chain builders. Pass llm=fake_llm(...) in tests; None means the real model.
# ---------------------------------------------------------------------------

def _model(llm):
    return llm if llm is not None else get_llm()


def build_menu_chain(llm=None):
    """{'request', 'guests'} -> four dishes as plain text (Step 2)."""
    return MENU_PROMPT | _model(llm) | StrOutputParser()


def build_notes_chain(llm=None):
    """{'menu'} -> one cooking tip per dish (Step 4)."""
    return NOTES_PROMPT | _model(llm) | StrOutputParser()


def price_list_text(prices) -> str:
    """'- rice (g)' lines from prices.csv, so the model uses names our costing can find."""
    per_guest_unit = {"kg": "g", "litre": "ml", "piece": "piece"}
    return "\n".join(f"- {p.item} ({per_guest_unit[p.unit]})" for p in prices.values())


def build_ingredients_chain(llm=None, prices=None):
    """{'menu'} -> an Ingredients object, not text (Step 5)."""
    prompt = INGREDIENTS_PROMPT.partial(price_list=price_list_text(prices or load_prices()))
    return prompt | _model(llm).with_structured_output(Ingredients)


def build_swaps_chain(llm=None):
    """{'menu', 'over_by', 'expensive'} -> up to three cheaper swaps, no numbers (Step 10)."""
    return SWAPS_PROMPT | _model(llm) | StrOutputParser()


def build_invite_chain(llm=None):
    """{'request', 'menu'} -> a short invite. Printed only: our code never sends it."""
    return INVITE_PROMPT | _model(llm) | StrOutputParser()


def build_checklist_chain(llm=None):
    """{'request', 'guests', 'menu'} -> a six-line host checklist."""
    return CHECKLIST_PROMPT | _model(llm) | StrOutputParser()


def build_extras(llm=None):
    """Step 9: invite and checklist do not need each other, so they run side by side."""
    llm = _model(llm)
    return RunnableParallel(invite=build_invite_chain(llm), checklist=build_checklist_chain(llm))


# ---------------------------------------------------------------------------
# 3. Small helpers
# ---------------------------------------------------------------------------

class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls in one run:  chain.invoke(x, config={"callbacks": [counter]})."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()      # parallel steps call the model at the same time

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self._lock:
            self.calls += 1


def chain_name(messages) -> str:
    """Which of our chains sent these messages (looks at the system message)."""
    system = messages[0].content if messages else ""
    for start, name in CHAIN_NAMES.items():
        if system.startswith(start):
            return name
    return "other"


def friendly_error(error: Exception) -> str:
    """A short explanation of a failed model call, with the fix. Never a long traceback."""
    text = f"{type(error).__name__}: {error}"
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if type(error).__name__ in ("OutputParserException", "ValidationError", "EmptyListError"):
        return ("The model's answer did not fit the Ingredients form (Step 5). Run it again, or try\n"
                f"LLM_MODEL=gemini-3.8-flash in .env. ({text.splitlines()[0][:160]})")
    if provider == "ollama" and ("Connect" in type(error).__name__ or "refused" in text.lower()):
        return (f"Could not reach Ollama ({text}).\n"
                "Start the Ollama app, run  ollama pull qwen3.5:4b  and try again.")
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return (f"The free-tier limit was reached ({text[:120]}...).\n"
                "Wait a minute and try again. Your limits: https://aistudio.google.com/rate-limit")
    if "Connect" in type(error).__name__:
        return f"Could not reach the model's server ({text}). Check your internet connection."
    return f"The model call failed: {text}"


def print_items(items: Ingredients) -> None:
    print(f"{'Item':<22} {'Per guest':>10}  Unit")
    for i in items.items:
        print(f"{i.item:<22} {i.qty_per_person:>10g}  {i.unit}")


# ---------------------------------------------------------------------------
# 4. One small demo per step
# ---------------------------------------------------------------------------

def show_prompt(step: str, inputs: dict) -> None:
    """Print the messages a step would send, filled in. No model call, no key."""
    menu_placeholder = "<the menu written by the menu chain>"
    if step in ("menu", "raw"):
        filled = [MENU_PROMPT.invoke(inputs)]
        print(f"Template slots: {', '.join(sorted(MENU_PROMPT.input_variables))}")
        print(f"Input dict:     {inputs}\n")
    elif step == "notes":
        filled = [NOTES_PROMPT.invoke({"menu": menu_placeholder})]
    elif step == "ingredients":
        prompt = INGREDIENTS_PROMPT.partial(price_list=price_list_text(load_prices()))
        filled = [prompt.invoke({"menu": menu_placeholder})]
    else:  # parallel
        filled = [INVITE_PROMPT.invoke({**inputs, "menu": menu_placeholder}),
                  CHECKLIST_PROMPT.invoke({**inputs, "menu": menu_placeholder})]
    for prompt_value in filled:
        for message in prompt_value.to_messages():
            print(f"----- {message.type} message -----\n{message.text}")
        print()


def demo_menu(llm, inputs, config=None) -> None:
    menu_chain = build_menu_chain(llm)
    print(f"Input dict: {inputs}\n")
    print(menu_chain.invoke(inputs, config=config))


def demo_raw(llm, inputs, config=None) -> None:
    raw_chain = MENU_PROMPT | llm                     # no parser at the end
    message = raw_chain.invoke(inputs, config=config)
    print("Without a parser:  MENU_PROMPT | llm")
    print(f"  type(result)          -> {type(message).__name__}")
    print(f"  type(result.content)  -> {type(message.content).__name__}")
    print(f"  metadata it carries   -> {', '.join(sorted(message.response_metadata)) or 'none'}")
    handed_on = NOTES_PROMPT.invoke({"menu": message}).to_messages()[-1].text
    print("\nHand that object to the next prompt and it gets ALL of this (first 300 characters):")
    print("  " + handed_on[:300].replace("\n", " ") + " ...")

    text = StrOutputParser().invoke(message)          # what the parser does, on its own
    print("\nWith the parser:  StrOutputParser().invoke(result)")
    print(f"  isinstance(..., str)  -> {isinstance(text, str)}   "
          f"(type name {type(text).__name__}: LangChain's own kind of str)")
    print("  " + text.replace("\n", "\n  "))


def demo_notes(llm, inputs, config=None) -> None:
    # The old SimpleSequentialChain. The dict step runs menu_chain and puts its text
    # into the slot called "menu", which is exactly the slot notes_chain needs.
    menu_then_notes = {"menu": build_menu_chain(llm)} | build_notes_chain(llm)
    print('Chain: {"menu": menu_chain} | notes_chain\n')
    print(menu_then_notes.invoke(inputs, config=config))


def demo_ingredients(llm, inputs, config=None) -> None:
    menu_then_items = {"menu": build_menu_chain(llm)} | build_ingredients_chain(llm)
    items = menu_then_items.invoke(inputs, config=config)
    if items is None:                       # the model answered without filling in the form
        raise EmptyListError("the model sent no grocery list.")
    print('Chain: {"menu": menu_chain} | ingredients_chain')
    print(f"type(result) -> {type(items).__name__} with {len(items.items)} items (an object, not text)\n")
    print_items(items)


def demo_parallel(llm, inputs, config=None) -> None:
    extras = RunnablePassthrough.assign(menu=build_menu_chain(llm)) | build_extras(llm)
    result = extras.invoke(inputs, config=config)
    print("Chain: RunnablePassthrough.assign(menu=menu_chain) | RunnableParallel(invite=..., checklist=...)")
    print(f"Keys in the result: {', '.join(result)}\n")
    print("== invite (printed only, never sent) ==")
    print(result["invite"])
    print("\n== checklist ==")
    print(result["checklist"])


DEMOS = {"menu": demo_menu, "raw": demo_raw, "notes": demo_notes,
         "ingredients": demo_ingredients, "parallel": demo_parallel}


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="The chains of the Party Planner, one step at a time.")
    parser.add_argument("step", choices=list(DEMOS))
    parser.add_argument("--request", default=REQUEST, help="the party, in a few words")
    parser.add_argument("--guests", type=int, default=GUESTS)
    parser.add_argument("--show-prompt", action="store_true", help="print the filled prompt only (no key)")
    parser.add_argument("--show-schema", action="store_true", help="print the Ingredients form (no key)")
    args = parser.parse_args(argv)
    inputs = {"request": args.request, "guests": args.guests}

    if args.show_schema:
        print(json.dumps(Ingredients.model_json_schema(), indent=2))
        return 0
    if args.show_prompt:
        show_prompt(args.step, inputs)
        return 0

    try:
        llm = _model(llm)
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1

    counter = CallCounter()
    start = time.perf_counter()
    try:
        # The callback in the config counts every model call inside the run.
        DEMOS[args.step](llm, inputs, config={"callbacks": [counter]})
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1
    print(f"\n({counter.calls} model call{'s' if counter.calls != 1 else ''}, "
          f"{time.perf_counter() - start:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
