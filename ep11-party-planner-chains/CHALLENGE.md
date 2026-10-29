# Your Turn: a dietary filter and a second-language invite

Two small changes to the Party Planner, one in Python and one in the parallel block.

## The task

1. **A dietary filter in Python.** Some guests eat Jain food (no onion, garlic, potato or other root vegetables)
   or vegan food (no milk, curd, ghee or other dairy). Write a plain function in `costing.py`:

   ```python
   def dietary_filter(state: dict) -> Ingredients:
       """Remove the items the diet does not allow, before costing."""
   ```

   It reads `state["items"]` (the typed grocery list) and `state["diet"]` (`"jain"`, `"vegan"` or `"none"`),
   and returns a new `Ingredients` without the items that diet does not allow. Keep the lists of forbidden
   items in two Python sets at the top of the file, written by you. The model must not decide what is
   allowed: Python does.
2. **Put it in the pipeline.** In `party.py`, add one more `RunnablePassthrough.assign(...)` in
   `build_plan_chain`, **after** `items` and **before** `bill`, that wraps your function in `RunnableLambda`
   and writes the result back to `items`. Add a `--diet` option to `main()` and put it in the input dict.
3. **Print what was removed.** The shopping list should say which items the filter took out, so the host can
   ask the model for a replacement dish if needed.
4. **A second-language invite.** In `chains.py`, write `build_invite_chain_in(llm, language)` (or add a
   `{language}` slot to a copy of the invite prompt), and add a third key to the parallel block in
   `build_party_chain`: `invite_local=...`. Pick any language you speak: Hindi, Tamil, Telugu, Kannada,
   Malayalam, Marathi, Bengali... The invite in that language is printed under its own heading.
5. **Test both offline.** In `tests/test_costing.py`, test `dietary_filter` on a fixed `Ingredients` list
   (one test for Jain, one for vegan, one for `"none"`). In `tests/test_chains.py`, run `build_party_chain`
   with a scripted fake model and check that the result has the new `invite_local` key and that the model was
   called one more time than before.
6. **Run `python -m pytest`.** A few old tests now fail, and that is good news: they check the exact keys and
   the exact number of model calls, so they noticed your extra key and your extra call. Update their expected
   keys and counts, and give them (and `count_calls_demo` in `party.py`) one more scripted reply for the second
   invite. Then everything passes again.

Post your diet rule, your language and the model-call count of your run in the video comments.

## Done when

- [ ] `python party.py "Jain lunch for 10 guests, budget 3000 rupees" --diet jain` prints a shopping list with no onion, garlic or potato in it, and says what it removed
- [ ] `--diet vegan` removes milk, curd and ghee; `--diet none` (the default) changes nothing
- [ ] The run prints two invites, one in English and one in your language
- [ ] `Model calls this run` is one more than before (the extra invite), and the filter added **no** model call
- [ ] `python -m pytest` passes, with your new tests included

<details>
<summary>Hint</summary>

- A `RunnableLambda` step only sees the running dict, so `diet` must be in the dict you pass to `invoke`.
- Compare names the way `find_price` in `costing.py` does (small letters, extra spaces removed), so
  `"Onion"` and `"onions"` are caught too.
- Return a **new** `Ingredients(items=[...])` instead of changing the old list: it keeps each step easy to test.
- Several keys in one `RunnablePassthrough.assign(...)` run side by side, so the second invite does not make the
  run noticeably longer. Check `Model calls this run`, not the clock.
- In a test, pass `config={"max_concurrency": 1}` to `invoke` so the scripted replies reach the branches in a
  fixed order (look at `run_party` in `tests/test_chains.py`).

</details>

## Stretch

Re-cost the swaps. Ask the swaps chain for a typed answer (a Pydantic `Swap` with `replace` and `with_item`,
names from `prices.csv`), apply each swap to the grocery list in Python, and run `scale_and_cost` and
`budget_check` again. Does the new bill fit the budget? Python should say, not the model.

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is
live, so try it yourself first.
