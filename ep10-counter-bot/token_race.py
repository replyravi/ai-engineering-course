"""
token_race.py - Step 13b: one 20-turn chat through all three memories. No API key needed.

No model is called. The bot's replies (and the summary notes) are written by hand and played back
by the fake model from testing/fakes.py, so every computer prints the same numbers.
What IS real: the agent, the checkpointer, the window and summary middleware, and the token meter.
The meter counts what each memory sends to the model on every turn, with LangChain's offline
estimate (count_tokens_approximately, about 4 characters per token). Gemini's own count will differ.

    python token_race.py                  tokens sent on every turn, for each memory
    python token_race.py --show window    what the model sees on every turn (buffer, window or summary)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse

from memory_modes import MODES, build_bot, is_summary
from testing.fakes import fake_llm
from token_meter import TokenMeter

# (what the customer types, the reply we wrote by hand for the race)
SCRIPT = [
    ("Hi, I'm Asha. I'm allergic to nuts.", "Hi Asha! Thanks for telling me about the nut allergy. What would you like?"),
    ("One masala chai, please.", "One masala chai, noted. Please also tell the staff about your nut allergy."),
    ("Add one vada pav.", "Added one vada pav. It is quite spicy, with dry garlic chutney."),
    ("Is the samosa spicy too?", "The samosa is medium spicy, and a plate has two pieces."),
    ("Okay, add one samosa plate.", "Added one samosa plate. So far: masala chai, vada pav and samosa."),
    ("I'll pick it up at 6:15 pm.", "Pickup at 6:15 pm, noted. We hold pickup orders for 30 minutes."),
    ("Do you take UPI?", "Yes, you can pay by UPI or in cash when you collect."),
    ("Can you make the chai less sweet?", "Sure, one masala chai with less sugar."),
    ("How much is my order so far?", "Masala chai 20, vada pav 25 and samosa 30: 75 rupees in all."),
    ("Add a bun maska.", "Added one bun maska for 25 rupees. Your total is now 100 rupees."),
    ("Actually, remove the samosa.", "Removed the samosa. Your total is now 70 rupees."),
    ("What time do you close today?", "We close at 9 pm today."),
    ("Are the snacks fresh?", "Yes, we fry snacks in small batches and never keep them overnight."),
    ("Make it two masala chai.", "Two masala chai, both with less sugar. Your total is 90 rupees."),
    ("Do you have a tea without milk?", "Yes, our lemon ginger tea has no milk."),
    ("Then add one lemon ginger tea.", "Added one lemon ginger tea for 25 rupees. Your total is 115 rupees."),
    ("What was my pickup time again?", "Your pickup time is 6:15 pm."),
    ("Please read my whole order back.",
     "Two masala chai with less sugar, one vada pav, one bun maska and one lemon ginger tea."),
    ("Do I pay now or at pickup?", "You pay when you collect, by UPI or in cash."),
    ("Great. Who am I, and what am I allergic to?", "You are Asha, and you are allergic to nuts."),
]

# The notes the summary call "writes" (by hand, in order). The summary memory squeezes 8 times in 20 turns.
SUMMARY_NOTES = [
    "Asha, allergic to nuts. Order: 1 masala chai, 1 vada pav.",
    "Asha, allergic to nuts. Order: 1 masala chai, 1 vada pav, 1 samosa plate.",
    "Asha, allergic to nuts. Order: 1 masala chai, 1 vada pav, 1 samosa plate. Pickup 6:15 pm, pays by UPI.",
    "Asha, allergic to nuts. Order: 1 masala chai with less sugar, 1 vada pav, 1 samosa plate. "
    "Pickup 6:15 pm, pays by UPI.",
    "Asha, allergic to nuts. Order: 1 masala chai with less sugar, 1 vada pav, 1 bun maska (samosa removed). "
    "Pickup 6:15 pm, pays by UPI.",
    "Asha, allergic to nuts. Order: 1 masala chai with less sugar, 1 vada pav, 1 bun maska. "
    "Pickup 6:15 pm, pays by UPI.",
    "Asha, allergic to nuts. Order: 2 masala chai with less sugar, 1 vada pav, 1 bun maska. "
    "Pickup 6:15 pm, pays by UPI. Asked for a tea without milk.",
    "Asha, allergic to nuts. Order: 2 masala chai with less sugar, 1 vada pav, 1 bun maska, "
    "1 lemon ginger tea. Pickup 6:15 pm, pays by UPI.",
]


def run(mode, script=SCRIPT, notes=SUMMARY_NOTES):
    """Play the script through one memory. Returns (meter, turns): turns[i] = the meter's calls of turn i+1."""
    llm = fake_llm(*[reply for _, reply in script])
    summary_llm = fake_llm(*notes) if mode == "summary" else None
    bot = build_bot(mode, llm=llm, summary_llm=summary_llm)
    meter = TokenMeter()                     # counts with count_tokens_approximately
    config = {"configurable": {"thread_id": f"race-{mode}"}, "callbacks": [meter]}
    turns = []
    for line, _ in script:
        meter.new_turn()
        bot.invoke({"messages": [{"role": "user", "content": line}]}, config)
        turns.append(meter.this_turn())
    return meter, turns


def tokens(calls, kind):
    return sum(call.estimate for call in calls if call.kind == kind)


def race():
    results = {mode: run(mode)[1] for mode in MODES}
    print(f"The same {len(SCRIPT)}-turn chat in all three memories. Tokens sent to the model on each turn")
    print("(an offline estimate: count_tokens_approximately, about 4 characters per token).")
    print("The replies are written by hand and played back by a fake model; no AI is called.\n")
    print(f"{'turn':>4}  {'buffer':>6}  {'window':>6}  {'summary':>7}   extra summary call")
    for index in range(len(SCRIPT)):
        row = [tokens(results[mode][index], "reply") for mode in MODES]
        extra = tokens(results["summary"][index], "summary")
        note = f"+ {extra}" if extra else ""
        print(f"{index + 1:>4}  {row[0]:>6}  {row[1]:>6}  {row[2]:>7}   {note}".rstrip())
    totals = {mode: sum(tokens(turn, "reply") for turn in results[mode]) for mode in MODES}
    squeeze = sum(tokens(turn, "summary") for turn in results["summary"])
    calls = sum(1 for turn in results["summary"] for call in turn if call.kind == "summary")
    print(f"{'all':>4}  {totals['buffer']:>6,}  {totals['window']:>6,}  {totals['summary']:>7,}   + {squeeze:,}")
    print(f"\nModel calls: {len(SCRIPT)} in every memory, plus {calls} summary calls in summary memory.")
    print(f"Summary memory in all, with its summary calls: {totals['summary'] + squeeze:,} tokens "
          f"(buffer {totals['buffer']:,}, window {totals['window']:,}).")


def label(message, script=SCRIPT):
    """'customer, turn 3' for a scripted message, so you can see which turns the model still sees."""
    if is_summary(message):
        return "summary note"
    for number, (line, reply) in enumerate(script, start=1):
        if message.type == "human" and message.text == line:
            return f"customer, turn {number}"
        if message.type == "ai" and message.text == reply:
            return f"bot, turn {number}"
    return message.type


def show(mode):
    _, turns = run(mode)
    print(f"What the model sees on every turn in {mode} memory (replies written by hand, played by a fake model):\n")
    for number, calls in enumerate(turns, start=1):
        for call in calls:
            kind = "summary call" if call.kind == "summary" else "model call"
            if call.kind == "summary":
                print(f"turn {number:>2}  {kind}: 1 message asking for a note, about {call.estimate} tokens")
                continue
            seen = [label(m) for m in call.sent if m.type != "system"]
            customer_turns = [s.split()[-1] for s in seen if s.startswith("customer")]
            extra = " + summary note" if "summary note" in seen else ""
            print(f"turn {number:>2}  {kind}: system prompt{extra} + customer turns {', '.join(customer_turns)}"
                  f"  ({len(call.sent)} messages, about {call.estimate} tokens)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="One scripted chat through buffer, window and summary memory.")
    parser.add_argument("--show", choices=MODES, help="print what the model sees on every turn in this memory")
    args = parser.parse_args(argv)
    if args.show:
        show(args.show)
    else:
        race()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
