# Your Turn: a calendar for the desk, and a fourth lead in Hindi

Two small jobs, one for each crew. Both are about the same habit: **give the model the facts it cannot know, and test the tool before you trust it.**

## Part 1 - A `today` tool for the research desk

A model has no clock. Its "sense of time" stops where its training data stops, so a crew that is not told the date may write "as of 2024" or search for news from an old year. Our desk avoids that by passing `{today}` in the task text. Now make the date a **tool** and measure the difference.

1. **See the problem first.** Make a copy of the Market Analyst's task text in `desk_crew.py` without the sentence `Today is {today}.`, and ask for one extra sentence: "Say in which month and year this note is written." Run `python desk_crew.py --ticker DEMO2` with your key, then `python claims_check.py reports/DEMO2.md`. Does the line `Dates or years not from a tool` show a year? Write down what you see (it may be fine, or it may be a stale year: models differ).
2. **Add the tool** in a new file `tools/today.py`: a CrewAI tool named `today` that returns the date as `YYYY-MM-DD` and writes it into the Ledger (`ledger.record("today", value)`), so the claims checker accepts it. Give it to the Market Analyst, and tell the task to call it once.
3. **Run again** and compare the claims check. The year should now come from a tool.
4. **Test it offline** in `tests/test_today.py`: the tool returns today's date, and a scripted run (see `replay.py`) that calls `today` first gives `0 unsupported` dates.

## Part 2 - A fourth lead, written in Hindi

1. **Write the lead**: `leads/<a-name>.md` with the same `- Label: value` lines as the other three, but the values in Hindi (an invented business, an invented person, an address on `example.com`). Keep the labels (`- Contact:`, `- Email:`, `- Recent news:` ...) in English: our code reads them.
2. **Test the tone tool with known Hindi sentences first.** Write three test sentences in `tests/test_tone.py`: a warm one, a pushy one and a cold one. Run them through `tone_report` with the real model and with the word list. What happens? (Hint: look at what the model was trained on, and at the regular expression in `WordListSentiment`.)
3. **Make the tone check work for Hindi**, one of two ways:
   - **Word list**: add Hindi warm and cold words, and change the word pattern so it also matches Devanagari letters. Small, fully tested, and honest about its limits.
   - **A multilingual model**: find a small multilingual sentiment model on Hugging Face that has an ONNX copy and a licence you may use, and add it as a second `LocalSentimentModel` (a different `MODEL_ID`). Check its scores on your known sentences, like we did in Step 3, before you trust it.
4. **Run the crew** for your new lead: `python outreach_crew.py --lead <a-name>` with your key, then `python approve.py`. Does the Writer answer in Hindi or English? Should it? Decide, and put the rule in the Writer's task text.

## Done when

- [ ] `python claims_check.py reports/DEMO2.md` shows `0` dates or years not from a tool, with the `today` tool in the Tool calls line
- [ ] `python -m tools.tone "<your warm Hindi sentence>"` (or `--words`) gives `ok`, and your pushy Hindi sentence gives `check`
- [ ] `python inputs_check.py` lists four leads, each `OK, every input used, no blank left`
- [ ] Your new lead's draft is in `drafts/approved/` or `drafts/rejected/`, decided by you
- [ ] `python -m pytest` passes, with your new tests, while the internet is blocked
- [ ] Post in the video comments: which way you chose for Hindi, and one known sentence it got wrong

<details>
<summary>Hint</summary>

- A CrewAI tool with no arguments is fine: `@tool("today")` on a function `def today() -> str:` with a clear docstring. To write into the Ledger, make it a small `BaseTool` class with a `ledger` field, like `tools/calculator.py`.
- `datetime.date.today().isoformat()` gives `2026-10-26` style text. In tests, compare with the same expression, not with a fixed date.
- `re.findall(r"[a-z']+", ...)` only finds English letters. `re.findall(r"\w+", text.lower())` finds Hindi words too.
- `--replay` uses `LEAD_SCRIPTS` in `replay.py`, one entry per lead. Add one for your lead, or the replay stops with a `KeyError`.
- A model card tells you which languages the model saw. If Hindi is not on the list, a confident score means nothing.

</details>

## Stretch

Add the `today` tool to the **outreach** crew too, and let the Writer say "this week" instead of a date. Then think: which facts in a lead file go stale (a "recent news" line from last year)? Add a `- Checked on: YYYY-MM-DD` line to each lead, and make the Researcher skip facts older than six months. How would you test that offline?

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is live, so try it yourself first.
