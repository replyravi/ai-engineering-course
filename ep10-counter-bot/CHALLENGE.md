# Your Turn: a customer profile that never scrolls away

Window memory is cheap, but on turn 20 it has forgotten that Asha is allergic to nuts. Summary memory remembers, but it pays for a squeeze every two turns and its note can drop details. Build a fourth memory that keeps the few facts a counter must never lose, and race it against the other three.

## The task

1. Make a small Pydantic class, `CustomerProfile`, with three fields: `name`, `allergies` (a list) and `favourite_item`.
2. After each customer turn, ask the model to fill it in with structured output (`llm.with_structured_output(CustomerProfile)`, Episode 01 and Episode 06), and merge it with the profile you already have. Keep one profile per `thread_id`.
3. Pin the profile: every model call must see it, whatever the window cut away. Put it in the system prompt for each call, not in the chat.
4. Pair it with window memory (`keep_recent_turns` from `memory_modes.py`).
5. Play the 20-turn script from `token_race.py` through all four memories and print one table: total tokens for buffer, window, summary and **profile-plus-window**. Count the profile calls too, the same way the race counts the summary calls.
6. Write one offline test: with fake models, the model call on turn 20 must see `Asha` and `nuts` in the profile, while the window no longer holds turn 1.

Post your four totals in the video comments.

## Done when

- [ ] The turn-20 model call shows the profile (`name: Asha`, `allergies: nuts`) and only customer turns 18, 19 and 20
- [ ] Your table has four lanes, and the profile lane includes its extraction calls
- [ ] `python -m pytest` passes, including your new test, with no key and no internet
- [ ] The profile is never saved as a fake chat message (that is the old `save_context` mistake from Step 14)

<details>
<summary>Hint</summary>

- `langchain.agents.middleware` has a `@dynamic_prompt` decorator: a function that builds the system prompt fresh before every model call. `CAFE_RULES` plus one line with the profile is enough.
- In tests, `structured_reply(CustomerProfile(...))` from `testing/fakes.py` scripts what `with_structured_output` returns. Use one fake for the bot's replies and a second fake for the profile calls, like `token_race.py` does for the summary notes.
- Pass the `TokenMeter` in the profile call's `config` too, so it sees those calls. Give them their own label through the call's `metadata` and teach `TokenMeter` to tell them apart.
- Do not ask for a profile on turns that add nothing new? That saves calls, but then you need a rule for when to ask. Measure both.

</details>

## Stretch

Keep the **full** chat on disk but send only the window to the model. Our `@before_model` window deletes old turns from the saved thread. A `@wrap_model_call` middleware can trim only the messages of the model request instead. Check with `/history` that every turn is still saved, and with the meter that the model still gets only the window.

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is live, so try it yourself first.
