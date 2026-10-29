"""
party.py - the whole Party Planner pipeline (Steps 7, 8 and 10).

One sentence in. Out come a menu, a costed shopping list, a budget check, cheaper
swaps (only when the bill is over budget), an invite and a host checklist.

    request -> menu (model) -> items (model, typed) -> bill (Python) -> check (Python)
            -> swaps (model, only if over budget) -> invite + checklist (model, side by side)

RunnablePassthrough.assign adds ONE key per step to a running dict, like a backpack that
collects something at every stop. Nothing is thrown away, and every model call happens once.

Run:  python party.py "South Indian lunch for 12 guests, budget 6000 rupees"
      python party.py "..." --plan-only     (Step 7: only menu, items and bill)
      python party.py --count-calls         (Step 8: count model calls with a SCRIPTED fake, no key)

Python reads the guest count and the budget from your sentence, so write them as digits.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re
import time

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from chains import (CallCounter, build_checklist_chain, build_ingredients_chain, build_invite_chain,
                    build_menu_chain, build_swaps_chain, chain_name, friendly_error)
from costing import Ingredients, SAMPLE_ITEMS, budget_step, cost_step, print_bill, rupees
from llm import MissingAPIKeyError, get_llm

REQUEST = "South Indian lunch for 12 guests, budget 6000 rupees"
NO_SWAPS = "Not needed: the bill fits the budget."


# ---------------------------------------------------------------------------
# 1. Python reads the numbers from the sentence (the model never touches them)
# ---------------------------------------------------------------------------

class RequestError(ValueError):
    """The sentence has no guest count or no budget we can read."""


def read_request(text: str, guests: int | None = None, budget: int | None = None) -> dict:
    """'South Indian lunch for 12 guests, budget 6000 rupees' -> {'request', 'guests', 'budget'}."""
    lower = text.lower()
    if guests is None:
        found = (re.search(r"(\d+)\s*(?:guests?|people|persons?|pax)\b", lower)
                 or re.search(r"\bfor\s+(\d+)\b", lower))
        if not found:
            raise RequestError("How many guests? Write it as digits, like 'for 12 guests', "
                               "or add --guests 12.")
        guests = int(found.group(1))
    if budget is None:
        found = (re.search(r"budget\D{0,15}?(\d[\d,]*)", lower)
                 or re.search(r"(\d[\d,]*)\s*(?:rupees|rs\b|inr\b)", lower))
        if not found:
            raise RequestError("What is the budget? Write it as digits, like 'budget 6000 rupees', "
                               "or add --budget 6000.")
        budget = int(found.group(1).replace(",", ""))
    if guests < 1 or budget < 1:
        raise RequestError("The guest count and the budget must both be at least 1.")
    return {"request": text, "guests": guests, "budget": budget}


# ---------------------------------------------------------------------------
# 2. The pipeline
# ---------------------------------------------------------------------------

def build_plan_chain(llm=None):
    """Step 7: request -> menu -> items -> bill, keeping every key in one dict."""
    if llm is None:
        llm = get_llm()
    return (
        RunnablePassthrough.assign(menu=build_menu_chain(llm))           # model: 1 call
        | RunnablePassthrough.assign(items=build_ingredients_chain(llm))  # model: 1 call, typed
        | RunnablePassthrough.assign(bill=RunnableLambda(cost_step))      # Python: no call
    )


def swap_inputs(state: dict) -> dict:
    """Python picks what the swaps prompt needs: the menu, the gap and the 3 costliest items."""
    costliest = sorted(state["bill"].lines, key=lambda line: line.cost, reverse=True)[:3]
    return {"menu": state["menu"],
            "over_by": rupees(state["check"].over_by),
            "expensive": ", ".join(line.item for line in costliest)}


def build_party_chain(llm=None):
    """Steps 7-10: the plan, then the budget check, swaps if needed, invite and checklist."""
    if llm is None:
        llm = get_llm()
    swaps_chain = RunnableLambda(swap_inputs) | build_swaps_chain(llm)

    def swaps_if_over_budget(state):
        # Python decides. A Runnable returned from a RunnableLambda is run next with the
        # same input; plain text is just the answer. (Episode 12 builds a full router this way.)
        return swaps_chain if state["check"].is_over else NO_SWAPS

    return (
        build_plan_chain(llm)
        | RunnablePassthrough.assign(check=RunnableLambda(budget_step))           # Python
        | RunnablePassthrough.assign(swaps=RunnableLambda(swaps_if_over_budget))  # model, maybe
        # Two keys in ONE assign run side by side: assign builds a RunnableParallel inside (Step 9).
        | RunnablePassthrough.assign(invite=build_invite_chain(llm), checklist=build_checklist_chain(llm))
    )


# ---------------------------------------------------------------------------
# 3. Printing
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    print("== Menu (model) ==")
    print(result["menu"])
    print(f"\n== Grocery list (model, typed): {len(result['items'].items)} items -> "
          "shopping list (Python) ==")
    print_bill(result["bill"])
    if "check" not in result:
        return
    print("\n== Budget check (Python) ==")
    print(result["check"].message())
    print("\n== Cheaper swaps (model, only when over budget) ==")
    print(result["swaps"])
    print("\n== Invite (model; printed only, never sent) ==")
    print(result["invite"])
    print("\n== Host checklist (model) ==")
    print(result["checklist"])


# ---------------------------------------------------------------------------
# 4. Step 8: count model calls with a SCRIPTED fake model (no key, no internet)
# ---------------------------------------------------------------------------

def scripted_menu(n: int) -> str:
    return f"[scripted menu {n}] Sambar rice - Beans poriyal - Tomato rasam - Semiya payasam"


def count_calls_demo() -> None:
    from testing.fakes import fake_llm, structured_reply

    inputs = read_request(REQUEST)
    items = structured_reply(Ingredients(items=SAMPLE_ITEMS))
    print("Counting model calls with a SCRIPTED fake model (no key, no internet).")
    print("Every model reply below is scripted by us, not written by an AI.\n")

    # A) The old habit: to print each stage, re-run the chain from the start up to it.
    llm = fake_llm(scripted_menu(1), scripted_menu(2), items, scripted_menu(3), items)
    menu_chain = build_menu_chain(llm)
    with_menu = RunnablePassthrough.assign(menu=menu_chain)
    with_items = with_menu | RunnablePassthrough.assign(items=build_ingredients_chain(llm))
    with_bill = with_items | RunnablePassthrough.assign(bill=RunnableLambda(cost_step))
    print("A) Print each stage by re-running the chain up to it (what older tutorials did):")
    stages = [("print the menu", menu_chain), ("print the grocery list", with_items),
              ("print the bill", with_bill)]
    for label, chain in stages:
        before = len(llm.received)
        result = chain.invoke(inputs)
        menu = result if isinstance(result, str) else result["menu"]
        calls = len(llm.received) - before
        print(f"   {label:<24} -> {calls} model call{'s' if calls > 1 else ' '}   "
              f"running total {len(llm.received)}   (menu used: {menu.split(']')[0][1:]})")
    print(f"   total: {len(llm.received)} model calls, and the three printouts came from 3 different menus\n")

    # B) RunnablePassthrough.assign: one run, every stage kept.
    llm = fake_llm(scripted_menu(1), items)
    result = build_plan_chain(llm).invoke(inputs)
    print("B) One run of build_plan_chain() with RunnablePassthrough.assign:")
    print(f"   keys: {', '.join(result)}")
    print(f"   total: {len(llm.received)} model calls, one menu all the way through "
          f"(bill {rupees(result['bill'].total)})\n")

    # C) The whole party pipeline. max_concurrency=1 keeps the scripted replies in order.
    print("C) The whole party pipeline, build_party_chain():")
    for budget, replies in [(6000, ["[scripted invite]", "[scripted checklist]"]),
                            (800, ["[scripted swaps]", "[scripted invite]", "[scripted checklist]"])]:
        llm = fake_llm(scripted_menu(1), items, *replies)
        result = build_party_chain(llm).invoke({**inputs, "budget": budget}, config={"max_concurrency": 1})
        calls = ", ".join(chain_name(messages) for messages in llm.received)
        print(f"   budget {rupees(budget, whole=True):<9} {result['check'].message()}")
        print(f"   {'':<16} {len(llm.received)} model calls: {calls}")


# ---------------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------------

def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="The Party Planner pipeline.")
    parser.add_argument("request", nargs="?", default=REQUEST)
    parser.add_argument("--guests", type=int, help="overrides the number in the sentence")
    parser.add_argument("--budget", type=int, help="in rupees; overrides the number in the sentence")
    parser.add_argument("--plan-only", action="store_true", help="Step 7: only menu, items and bill")
    parser.add_argument("--count-calls", action="store_true", help="Step 8: scripted fake model, no key")
    args = parser.parse_args(argv)

    if args.count_calls:
        count_calls_demo()
        return 0

    try:
        inputs = read_request(args.request, args.guests, args.budget)
    except RequestError as error:
        print(error)
        return 1
    print(f"Party request: {inputs['request']}")
    print(f"Read by Python: {inputs['guests']} guests, budget {rupees(inputs['budget'], whole=True)}\n")

    try:
        chain = build_plan_chain(llm) if args.plan_only else build_party_chain(llm)
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1

    counter = CallCounter()
    start = time.perf_counter()
    try:
        result = chain.invoke(inputs, config={"callbacks": [counter]})
    except Exception as error:
        print(friendly_error(error))
        return 1

    print_result(result)
    print(f"\nKeys in the result: {', '.join(result)}")
    print(f"Model calls this run: {counter.calls}   ({time.perf_counter() - start:.1f} s; "
          "every rupee above was added up by Python)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
