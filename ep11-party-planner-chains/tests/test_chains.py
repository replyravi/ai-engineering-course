"""
tests/test_chains.py - the chains and the whole pipeline (Steps 2-5 and 7-10) with a FAKE model.

We script what the "AI" answers and check OUR code: the dicts and prompts we send,
the hand-offs between steps, the typed grocery list, the number of model calls,
the budget decision and the printed report.
"""

from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

import chains
import party
from chains import (MENU_PROMPT, NOTES_PROMPT, CallCounter, build_extras, build_ingredients_chain,
                    build_menu_chain, build_notes_chain, chain_name)
from costing import SAMPLE_ITEMS, Ingredients
from party import NO_SWAPS, RequestError, build_party_chain, build_plan_chain, read_request
from party_fixtures import ITEMS, MENU, no_keys  # noqa: F401
from testing.fakes import fake_llm, gemini_style_reply, structured_reply

INPUTS = {"request": "South Indian lunch", "guests": 12}
PARTY = {"request": "South Indian lunch for 12 guests, budget 6000 rupees", "guests": 12, "budget": 6000}
SAMPLE = Ingredients(items=SAMPLE_ITEMS)


# ---------------------------------------------------------------------------
# Step 2: prompt | llm | parser, invoked with a dict
# ---------------------------------------------------------------------------

def test_menu_chain_fills_both_slots():
    llm = fake_llm(MENU)
    assert build_menu_chain(llm).invoke(INPUTS) == MENU
    system, human = llm.received[0]
    assert system.type == "system" and "exactly four dishes" in system.text
    assert human.text == "Party: South Indian lunch\nGuests: 12"


def test_a_bare_value_only_works_for_a_one_slot_prompt():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    one_slot = ChatPromptTemplate.from_template("Plan a menu for {request}") | fake_llm("ok") | StrOutputParser()
    assert one_slot.invoke("a picnic") == "ok"                    # LangChain wraps it as {"request": ...}
    with pytest.raises(TypeError, match="Expected mapping type as input"):
        build_menu_chain(fake_llm("never used")).invoke("a picnic")   # two slots need a dict


def test_a_missing_slot_names_the_slot():
    with pytest.raises(KeyError, match="missing variables {'guests'}"):
        build_menu_chain(fake_llm("never used")).invoke({"request": "a picnic"})


def test_menu_show_prompt_needs_no_key(no_keys, capsys):
    assert chains.main(["menu", "--show-prompt"]) == 0
    output = capsys.readouterr().out
    assert "Template slots: guests, request" in output
    assert "Party: South Indian lunch\nGuests: 12" in output


# ---------------------------------------------------------------------------
# Step 3: the parser
# ---------------------------------------------------------------------------

def test_without_a_parser_you_get_a_message_object():
    message = (MENU_PROMPT | fake_llm(gemini_style_reply(MENU))).invoke(INPUTS)
    assert isinstance(message, AIMessage)
    assert isinstance(message.content, list)                       # Gemini 3: a list of blocks
    assert message.text == MENU


def test_a_raw_message_leaks_into_the_next_prompt():
    message = (MENU_PROMPT | fake_llm(gemini_style_reply(MENU))).invoke(INPUTS)
    leaked = NOTES_PROMPT.invoke({"menu": message}).to_messages()[-1].text
    assert "content=[{'type': 'text'" in leaked and "signature" in leaked


def test_with_the_parser_you_get_plain_text():
    text = build_menu_chain(fake_llm(gemini_style_reply(MENU))).invoke(INPUTS)
    assert isinstance(text, str) and text == MENU


def test_raw_demo_prints_both_types(capsys):
    assert chains.main(["raw"], llm=fake_llm(gemini_style_reply(MENU))) == 0
    output = capsys.readouterr().out
    assert "type(result)          -> AIMessage" in output
    assert "type(result.content)  -> list" in output
    assert "isinstance(..., str)  -> True" in output
    assert "(1 model call," in output


# ---------------------------------------------------------------------------
# Step 4: the named hand-off (the old SimpleSequentialChain)
# ---------------------------------------------------------------------------

def test_the_menu_fills_the_menu_slot_of_the_notes_chain():
    llm = fake_llm(MENU, "Start the lemon rice first.")
    chain = {"menu": build_menu_chain(llm)} | build_notes_chain(llm)
    assert chain.invoke(INPUTS) == "Start the lemon rice first."
    assert llm.received[1][-1].text == f"Menu:\n{MENU}"            # exactly the menu of THIS run
    assert len(llm.received) == 2


# ---------------------------------------------------------------------------
# Step 5: typed ingredients
# ---------------------------------------------------------------------------

def test_ingredients_chain_returns_a_typed_object():
    items = build_ingredients_chain(fake_llm(structured_reply(ITEMS))).invoke({"menu": MENU})
    assert isinstance(items, Ingredients)
    assert items.items[2].item == "coconut" and items.items[2].unit == "piece"


def test_the_price_list_is_in_the_ingredients_prompt():
    llm = fake_llm(structured_reply(ITEMS))
    build_ingredients_chain(llm).invoke({"menu": MENU})
    system = llm.received[0][0].text
    assert "- rice (g)" in system and "- milk (ml)" in system and "- banana leaf (piece)" in system


def test_ingredients_schema_needs_no_key(no_keys, capsys):
    assert chains.main(["ingredients", "--show-schema"]) == 0
    output = capsys.readouterr().out
    assert '"qty_per_person"' in output and '"piece"' in output


# ---------------------------------------------------------------------------
# Step 7: keep every step with RunnablePassthrough.assign
# ---------------------------------------------------------------------------

def test_plan_chain_keeps_every_key_with_two_calls():
    llm = fake_llm(MENU, structured_reply(SAMPLE))
    result = build_plan_chain(llm).invoke(PARTY)
    assert list(result) == ["request", "guests", "budget", "menu", "items", "bill"]
    assert result["menu"] == MENU
    assert result["bill"].total == Decimal("938.40")
    assert len(llm.received) == 2                                   # one call per model step
    assert llm.received[1][-1].text == f"Menu:\n{MENU}"             # the grocery list saw THIS menu


def test_re_running_stages_costs_more_calls():
    llm = fake_llm(MENU, MENU, structured_reply(SAMPLE))
    build_menu_chain(llm).invoke(PARTY)                             # stage 1: 1 call
    build_plan_chain(llm).invoke(PARTY)                             # stages 1-3 again: 2 more
    assert len(llm.received) == 3


def test_count_calls_demo(capsys):
    assert party.main(["--count-calls"]) == 0
    output = capsys.readouterr().out
    assert "total: 5 model calls, and the three printouts came from 3 different menus" in output
    assert "keys: request, guests, budget, menu, items, bill" in output
    assert "total: 2 model calls, one menu all the way through (bill Rs 938.40)" in output
    assert "4 model calls: menu, grocery list, invite, checklist" in output
    assert "5 model calls: menu, grocery list, swaps, invite, checklist" in output


def test_the_call_counter_counts_like_the_fake():
    llm = fake_llm(MENU, structured_reply(SAMPLE))
    counter = CallCounter()
    build_plan_chain(llm).invoke(PARTY, config={"callbacks": [counter]})
    assert counter.calls == len(llm.received) == 2


# ---------------------------------------------------------------------------
# Step 9: invite and checklist in parallel
# ---------------------------------------------------------------------------

def test_extras_gives_two_outputs_from_one_run():
    llm = fake_llm("INVITE", "CHECKLIST")
    result = build_extras(llm).invoke({**INPUTS, "menu": MENU}, config={"max_concurrency": 1})
    assert result == {"invite": "INVITE", "checklist": "CHECKLIST"}
    assert [chain_name(m) for m in llm.received] == ["invite", "checklist"]


def test_the_invite_leaves_blanks_and_is_never_sent():
    llm = fake_llm("INVITE", "CHECKLIST")
    build_extras(llm).invoke({**INPUTS, "menu": MENU}, config={"max_concurrency": 1})
    assert "[date], [time] and [address]" in llm.received[0][0].text


# ---------------------------------------------------------------------------
# Step 10: Python decides the budget, the model only suggests swaps
# ---------------------------------------------------------------------------

def run_party(budget, *replies):
    llm = fake_llm(MENU, structured_reply(SAMPLE), *replies)
    result = build_party_chain(llm).invoke({**PARTY, "budget": budget}, config={"max_concurrency": 1})
    return result, llm


def test_within_budget_skips_the_swaps_call():
    result, llm = run_party(6000, "INVITE", "CHECKLIST")
    assert result["swaps"] == NO_SWAPS
    assert [chain_name(m) for m in llm.received] == ["menu", "grocery list", "invite", "checklist"]
    assert list(result) == ["request", "guests", "budget", "menu", "items", "bill", "check",
                            "swaps", "invite", "checklist"]


def test_over_budget_asks_for_swaps_with_python_numbers():
    result, llm = run_party(800, "Swap cashew nuts for raisins", "INVITE", "CHECKLIST")
    assert result["check"].over_by == Decimal("138.40")
    assert result["swaps"] == "Swap cashew nuts for raisins"
    swaps_prompt = llm.received[2]
    assert chain_name(swaps_prompt) == "swaps"
    assert "over budget by Rs 138.40" in swaps_prompt[-1].text
    assert "rice, milk, banana leaf" in swaps_prompt[-1].text      # the 3 costliest lines, chosen by Python
    assert "Do NOT write any prices" in swaps_prompt[0].text


def test_party_main_prints_every_section(capsys):
    llm = fake_llm(MENU, structured_reply(SAMPLE), "INVITE TEXT", "CHECKLIST TEXT")
    assert party.main([], llm=llm) == 0
    output = capsys.readouterr().out
    assert "Read by Python: 12 guests, budget Rs 6,000" in output
    assert "Total for 19 items" in output and "Rs 938.40" in output
    assert "Within budget: Rs 5,061.60 left" in output
    assert NO_SWAPS in output
    assert "INVITE TEXT" in output and "CHECKLIST TEXT" in output
    assert "Keys in the result: request, guests, budget, menu, items, bill, check, swaps, invite, checklist" in output
    assert "Model calls this run: 4" in output


def test_party_plan_only(capsys):
    assert party.main(["--plan-only"], llm=fake_llm(MENU, structured_reply(SAMPLE))) == 0
    output = capsys.readouterr().out
    assert "Keys in the result: request, guests, budget, menu, items, bill" in output
    assert "Model calls this run: 2" in output
    assert "Budget check" not in output


# ---------------------------------------------------------------------------
# Reading the request, and the friendly paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, guests, budget", [
    ("South Indian lunch for 12 guests, budget 6000 rupees", 12, 6000),
    ("Kerala sadya for 25 people with a budget of Rs 15,000", 25, 15000),
    ("Diwali snacks for 8, 2,500 rupees", 8, 2500),
])
def test_python_reads_guests_and_budget(text, guests, budget):
    assert read_request(text) == {"request": text, "guests": guests, "budget": budget}


def test_numbers_in_words_get_a_clear_message():
    with pytest.raises(RequestError, match="How many guests"):
        read_request("Lunch for twelve, budget 6000 rupees")
    with pytest.raises(RequestError, match="What is the budget"):
        read_request("Lunch for 12 guests")


def test_flags_override_the_sentence():
    assert read_request("Lunch for twelve", guests=12, budget=6000)["guests"] == 12


@pytest.mark.parametrize("module, argv", [(chains, ["menu"]), (party, []), (party, ["--plan-only"])])
def test_without_a_key_you_get_help_not_a_traceback(no_keys, capsys, module, argv):
    assert module.main(argv) == 1
    output = capsys.readouterr().out
    assert "https://aistudio.google.com/apikey" in output and "Traceback" not in output


def test_a_broken_grocery_list_gives_a_friendly_message(capsys):
    llm = fake_llm(MENU, "this is not a grocery list")             # no structured reply
    assert chains.main(["ingredients"], llm=llm) == 1
    assert "did not fit the Ingredients form" in capsys.readouterr().out


def test_no_internet_gives_a_friendly_message():
    import httpx

    from calculator import explain_error

    error = httpx.ConnectError("[Errno 8] nodename nor servname provided")
    assert "Check your internet connection" in chains.friendly_error(error)
    assert "Check your internet connection" in explain_error(error)


def test_a_rate_limit_points_to_the_limits_page():
    error = RuntimeError("429 RESOURCE_EXHAUSTED: quota")
    assert "https://aistudio.google.com/rate-limit" in chains.friendly_error(error)
