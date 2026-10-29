"""
Memory tests (Steps 9 to 13): what the model SEES on every turn, with a fake model.

FakeChatModel keeps every list of messages it was sent in llm.received, so a test can read
exactly what each memory sent: llm.received[0] is turn 1, llm.received[1] is turn 2, ...
"""

import pytest
from langchain_core.messages import AIMessage

import step09_two_calls
import step10_list_memory
import token_race
from memory_modes import DEMO_CHAT, build_bot, is_summary, last_exchanges
from testing.fakes import fake_llm
from token_meter import TokenMeter


def say(bot, text, thread="asha", meter=None):
    """One customer turn. Returns the agent's result."""
    config = {"configurable": {"thread_id": thread}}
    if meter is not None:
        config["callbacks"] = [meter]
    return bot.invoke({"messages": [{"role": "user", "content": text}]}, config)


def seen(llm, call):
    """(type, text) of every message the fake model received in one call."""
    return [(message.type, message.text) for message in llm.received[call]]


# --- Step 9: two bare calls ---

def test_two_bare_calls_each_send_only_their_own_message(capsys):
    llm = fake_llm("Nice to meet you, Asha!", "I'm sorry, I don't know your name.")

    assert step09_two_calls.main(llm=llm) == 0

    assert seen(llm, 0) == [("human", step09_two_calls.FIRST)]
    assert seen(llm, 1) == [("human", step09_two_calls.SECOND)]     # nothing about Asha is sent
    assert "Call 2" in capsys.readouterr().out


# --- Step 10: memory by hand ---

def test_list_memory_sends_the_whole_list_every_time():
    llm = fake_llm("Hi Asha!", "One masala chai, noted.", "Added.", "You are Asha.")

    history = step10_list_memory.chat_with_list_memory(llm, step10_list_memory.CUSTOMER_LINES, write=lambda _: None)

    assert [len(sent) for sent in llm.received] == [2, 4, 6, 8]       # system + everything so far
    assert len(history) == 9
    assert llm.received[3][1].text == "Hi, I'm Asha."                 # turn 4 still carries turn 1


# --- Step 11: buffer memory (create_agent + InMemorySaver) ---

def test_buffer_memory_sends_every_earlier_turn():
    llm = fake_llm("Hi Asha!", "One masala chai.", "You are Asha.")
    bot = build_bot("buffer", llm=llm)

    for text in ["Hi, I'm Asha.", "One masala chai, please.", "Who am I?"]:
        say(bot, text)

    assert [message_type for message_type, _ in seen(llm, 2)] == ["system", "human", "ai", "human", "ai", "human"]
    assert seen(llm, 2)[1] == ("human", "Hi, I'm Asha.")


def test_the_system_prompt_is_added_on_every_call_but_never_saved():
    llm = fake_llm("Hi!", "Hello again!")
    bot = build_bot("buffer", llm=llm, system_prompt="You are the counter bot.")
    say(bot, "Hi")
    say(bot, "Hi again")

    assert seen(llm, 1)[0] == ("system", "You are the counter bot.")
    saved = bot.get_state({"configurable": {"thread_id": "asha"}}).values["messages"]
    assert [message.type for message in saved] == ["human", "ai", "human", "ai"]


# --- Step 12: window memory ---

def test_a_window_without_start_on_keeps_a_reply_whose_question_is_gone():
    wrong = last_exchanges(DEMO_CHAT, start_on=None)
    right = last_exchanges(DEMO_CHAT)

    assert wrong[0].type == "ai" and "Asha" in wrong[0].text          # the name, without the line that gave it
    assert right[0].type == "human" and right[0].text == "One masala chai, please."
    assert right[-1].text == "Who am I?"
    assert len(right) == 5                                            # 2 exchanges + the new question


def test_window_memory_forgets_the_first_order_by_turn_five():
    lines = ["Hi, I'm Asha.", "One masala chai, please.", "Add one vada pav.", "Add one samosa plate too.",
             "What was my first order, and what is my name?"]
    llm = fake_llm("Hi Asha!", "One masala chai.", "Added a vada pav.", "Added a samosa plate.", "I'm not sure.")
    bot = build_bot("window", llm=llm)
    for text in lines:
        say(bot, text)

    customer_lines = [[text for kind, text in seen(llm, turn) if kind == "human"] for turn in range(5)]
    assert "One masala chai, please." in customer_lines[3]            # turn 4: still in the window
    assert "One masala chai, please." not in customer_lines[4]        # turn 5: gone
    assert customer_lines[4] == lines[2:]                             # turn 5 sees turns 3, 4 and 5 only
    assert [len(sent) for sent in llm.received] == [2, 4, 6, 6, 6]    # the meter flattens from turn 3
    saved = bot.get_state({"configurable": {"thread_id": "asha"}}).values["messages"]
    assert len(saved) == 6                                            # the old turns are deleted, not hidden


# --- Step 13: summary memory ---

def test_summary_memory_squeezes_old_turns_into_one_note_at_turn_five():
    llm = fake_llm("Hi Asha!", "One masala chai.", "Added.", "Pickup at 6:15 pm.", "Yes, UPI is fine.")
    summary_llm = fake_llm("Asha, allergic to nuts. Order: 1 masala chai, 1 vada pav.")
    bot = build_bot("summary", llm=llm, summary_llm=summary_llm)
    lines = ["Hi, I'm Asha. I'm allergic to nuts.", "One masala chai, please.", "Add one vada pav.",
             "I'll pick it up at 6:15 pm.", "Do you take UPI?"]
    for text in lines[:4]:
        say(bot, text)
    assert summary_llm.received == []                                 # 7 saved messages: no squeeze yet

    say(bot, lines[4])                                                # 9 messages >= trigger of 8: squeeze

    assert len(summary_llm.received) == 1                             # ONE extra model call
    asked = summary_llm.received[0][0].text
    assert "every allergy" in asked and "allergic to nuts" in asked   # our prompt + the old turns
    turn5 = llm.received[4]
    assert turn5[0].type == "system"
    assert is_summary(turn5[1])
    assert turn5[1].type == "human"                                   # the note is saved with the human role
    assert turn5[1].text.startswith("Here is a summary of the conversation to date:")
    assert "allergic to nuts" in turn5[1].text
    assert [m.text for m in turn5[3:] if m.type == "human"] == lines[3:]   # the newest turns stay word for word
    saved = bot.get_state({"configurable": {"thread_id": "asha"}}).values["messages"]
    assert sum(1 for message in saved if is_summary(message)) == 1
    assert is_summary(saved[0])


def test_an_unknown_memory_mode_is_refused():
    with pytest.raises(ValueError):
        build_bot("goldfish", llm=fake_llm())


# --- The token meter ---

def test_the_meter_counts_what_was_sent_with_any_counter():
    meter = TokenMeter(counter=lambda messages: 10 * len(messages))   # a fake counter: 10 per message
    bot = build_bot("buffer", llm=fake_llm("a", "b", "c"))

    totals = []
    for text in ["one", "two", "three"]:
        meter.new_turn()
        say(bot, text, meter=meter)
        totals.append(meter.this_turn()[0].estimate)

    assert totals == [20, 40, 60]                                     # system + 1, 3, 5 messages
    assert meter.total_estimate() == 120


def test_the_meter_reads_the_token_counts_the_provider_reports():
    reply = AIMessage("Hello Asha!", usage_metadata={"input_tokens": 57, "output_tokens": 4, "total_tokens": 61})
    meter = TokenMeter()
    say(build_bot("buffer", llm=fake_llm(reply)), "Hi", meter=meter)

    assert meter.calls[0].reported == (57, 4)
    assert "model counted 57 in, 4 out" in meter.turn_lines()[0]


def test_the_meter_sees_the_extra_summary_calls():
    meter = TokenMeter()
    bot = build_bot("summary", llm=fake_llm(*[f"reply {n}" for n in range(1, 6)]), summary_llm=fake_llm("note"))
    for number in range(1, 6):
        meter.new_turn()
        say(bot, f"turn {number}", meter=meter)

    assert [call.kind for call in meter.this_turn()] == ["summary", "reply"]
    assert meter.turn_lines()[0].startswith("summary call: sent 1 message,")


# --- Step 13b: the token race ---

def test_the_token_race_has_the_expected_shape():
    turns = {mode: token_race.run(mode)[1] for mode in ("buffer", "window", "summary")}
    sent = {mode: [token_race.tokens(turn, "reply") for turn in turns[mode]] for mode in turns}
    squeezes = [n for n, turn in enumerate(turns["summary"], start=1) if token_race.tokens(turn, "summary")]

    assert all(a < b for a, b in zip(sent["buffer"], sent["buffer"][1:]))   # buffer climbs every turn
    assert max(sent["window"][3:]) < sent["buffer"][3]                      # window stays flat and low
    assert squeezes == [5, 7, 9, 11, 13, 15, 17, 19]                         # 8 extra summary calls
    assert sum(sent["window"]) < sum(sent["summary"]) < sum(sent["buffer"])
