# Your Turn: add a Fact-Checker to the crew

The Writer copies prices and ratings by hand, like a person retyping a bill. A tired person makes mistakes, and so does a model. Add a fourth team member who checks the guide against what the Scout **really** scraped, and make the crew **fail loudly** when something does not match.

## The task

1. **A comparing tool** in `tools.py`: `compare_guide`. It takes the shelf name (for example `travel`), reads `reports/<shelf>_guide.md`, and checks every book section of the guide against the `Shelf` (the books `scrape_shelf` pinned there):
   - the title in the `## [title](url)` heading must be on the shelf, with the same url,
   - the `£` price and the number of stars in that section must match the scraped row.

   No mismatch: return a short JSON summary, for example `{"checked": 3, "mismatches": []}`. Any mismatch: return a `ToolFailure` that names each book and field, for example `"1,000 Places to See Before You Die: guide says £26.00, the shelf says £26.08"`. Python does the comparing, not the model (remember Episode 11: let Python do the maths).
2. **A Fact-Checker agent and task** in `crew.py`: a fourth agent that only has `compare_guide`, and a `fact_check_task` that runs after the Writer (`context=[writer_task]`) and calls the tool once. Keep `tool_failure_policy="raise"`, so a mismatch stops the crew with a red error instead of a happy ending.
3. **The scorecard** in `scout.py`: add a line for `compare_guide`, and make `--stop-after` accept `checker`.
4. **Two offline tests** in `tests/test_crew_fake.py`, with `FakeCrewAILLM`:
   - a scripted Writer that copies everything correctly: the crew finishes and the checker reports 0 mismatches,
   - a scripted Writer that writes `£9.99` for a book that costs `£12.50`: `kickoff()` raises `ToolExecutionFailedError`, and the message names that book and the price.
5. **One live run**: `python scout.py` on any shelf. Post your Fact-Checker's line in the video comments.

## Done when

- [ ] `python tools.py` can run `compare_guide` by hand on a saved guide (add a `check` command next to `scrape` and `similar`)
- [ ] A correct guide gives `0 mismatches`; a guide you edit by hand (change one price) stops the crew with a red error that names the book
- [ ] The first line of `python scout.py` says `agents: scout, analyst, writer, checker`, and the scorecard shows the `compare_guide` call
- [ ] `python -m pytest` passes, including your two new tests, with the internet blocked
- [ ] The README of your copy says in one line which numbers the checker compares

<details>
<summary>Hint</summary>

- One regular expression per thing you need: the heading `## [title](url)`, a price like `£26.08`, a rating like `5 stars`. Split the guide at each `## ` and check one section at a time.
- Compare prices as numbers rounded to 2 decimals, not as text: `26.1` and `26.10` are the same price.
- The Fact-Checker never needs to retype the guide: give the tool the shelf name, and let the tool read the file. Every number the model does not retype is a number it cannot get wrong.
- Three existing tests in `tests/test_crew_fake.py` expect three agents and four scripted replies (`test_full_crew_runs_offline_with_a_scripted_model`, `test_context_hands_each_answer_to_the_next_task`, `test_each_agent_gets_its_own_llm_object`). When the default crew gets a fourth agent they fail: that is the tests doing their job. Update them, or pass `stop_after="writer"` there.
- In the tests, the guide file is written into `tmp_path` (look at `monkeypatch.chdir(tmp_path)` in the existing tests), so your tool must read `reports/...` relative to the current folder, like CrewAI's `output_file` does.

</details>

## Stretch

Make the checker also compare the stock text (`In stock`). Then think about bigger shelves: a shop page shows at most 20 books, and `scrape_shelf` reads only the first page of a shelf. Find a shelf with a "next" button, and make the Writer say in the guide how many books were compared.

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is live, so try it yourself first.
