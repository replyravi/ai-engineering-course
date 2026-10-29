# Your Turn: teach the bot a new policy

Dabbaloop is announcing its **festival holiday schedule**. Add it to the handbook and prove the bot can use it.

## The task

1. Write `handbook/festival_holidays.md` yourself. Invent clear, checkable details, for example:
   - which festival days the kitchen is closed (Ganesh Chaturthi, Diwali Padwa, ...),
   - whether those days are credited automatically or moved to the end of the plan,
   - the last date to order a festival special box, and its price.
2. Re-index: `python ingest.py`. The new chunks are added, and **nothing else** is removed or duplicated.
3. Add **two** golden questions about the new file to `golden.json` (with `sources` and `answer_contains`).
4. Run `python eval_retrieval.py`. Keep the hit rate at **100 percent** (now 10 out of 10).
5. Run `python ask.py` and ask one of your new questions. The answer must cite your file, like `[festival_holidays.md#2]`.
6. Run `python -m pytest`. Everything still passes (`test_every_golden_answer_is_found_at_k3` now checks your questions too).

Post your hit-rate line and one cited answer in the video comments.

## Done when

- [ ] `python ingest.py` shows `new this run:` equal to your new file's chunks and `removed: 0`
- [ ] Running `python ingest.py` a second time shows `new this run: 0`
- [ ] `Hit rate at k=3: 10/10 = 100%`
- [ ] A real answer from `ask.py` cites `festival_holidays.md`
- [ ] `python -m pytest` passes

<details>
<summary>Hint</summary>

- Write each rule as a full sentence with its own number, like the other handbook files. A chunk that says "Closed on 21 October" without saying what is closed matches questions badly.
- If one of your golden questions misses, first run `python search.py "<your question>"` and read the top 3 chunks. Is the answer cut away from the words of the question? Try rewording the file, or look at what `--chunk-size` and `--k` do in `eval_retrieval.py`.
- Your new file can change other questions too: a near-duplicate sentence can push an older chunk out of the top 3. That is why the scoreboard checks all of them.

</details>

## Stretch

Make the schedule a PDF instead, using `tools/make_food_safety_pdf.py` as the pattern. Check that the citation shows the page number.

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is live, so try it yourself first.
