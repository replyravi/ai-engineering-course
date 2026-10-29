# Your Turn: a fourth route for complaints (drafts only)

Customers do not only ask questions. Sometimes they are upset: "my dal was cold again", "the rider was rude".
Give the desk a fourth route, `complaint`, that **drafts** a polite reply for a person on the Dabbaloop team to
check. The desk never sends anything by itself.

## The task

1. **A new label.** In `router.py`, add `"complaint"` to `ROUTES` and to the `Literal[...]` of `Route.destination`.
   Add one line about it to `ROUTER_RULES`, for example: complaint means the customer is unhappy about something
   that already happened and wants it put right.
2. **New examples.** In `embed_router.py`, add a `"complaint"` list to `ROUTE_EXAMPLES` with at least six examples
   written by you. Do not copy the questions you will test with (step 5).
3. **The complaint path.** In a new file `complaint.py`, write `build_complaint_chain(llm, drafts_dir)`. It takes
   `{"question": ...}` and:
   - asks the model for a short, polite reply (two to four sentences) that says sorry, says what happens next,
     and promises **no** refund, credit or amount of money (only a person may decide that),
   - saves the draft as a new Markdown file in `drafts/` with the customer's message, the draft and the line
     `Status: DRAFT - a person must review this. Nothing was sent.`,
   - returns `{"answer": "Draft saved to drafts/<file name> for a person to review. Nothing was sent.",
     "draft_path": ...}`.
   No email, no chat message, no API call to anything but the model. `drafts/` is already in `.gitignore`.
4. **Wire it in.** In `desk.py`, add `"complaint": build_complaint_chain(llm, ...)` to `paths`. Nothing else in the
   dispatch changes: that is the point of a dispatch table.
5. **Label four more questions.** Add four complaint questions of your own to `data/router_questions.json`
   (24 questions now). Make at least one of them tricky, for example a complaint that mentions a refund, which
   looks like a policy question.
6. **Measure.** Run `python eval_router.py --router both`. Keep the accuracy at **90 percent or more** (at least 22
   of 24) for the router you would put in front of customers. The confusion table is now 4 by 4.
7. **Test it offline.** In a new `tests/test_complaint.py`: run your chain with `fake_llm("Sorry ...")` and
   `tmp_path / "drafts"`, and check that exactly one file was written, that it contains the `Status: DRAFT` line,
   and that the model was called once. Add a desk test with a fixed router (see `fixed_router` in
   `tests/test_desk.py`) that sends a question to `complaint`.
8. **Run `python -m pytest`.** Three old tests now fail, and that is good news: `test_the_route_form_allows_exactly_three_labels`,
   `test_twenty_labelled_questions` and `test_semantic_router_scores_18_of_20` check the exact labels, the exact number
   of questions and the exact score. Update them to your new labels and numbers. Then everything passes again.

Post your confusion table and one of your drafts in the video comments.

## Done when

- [ ] `python desk.py "My dal was cold again today and nobody answered my call"` prints `route=complaint` and
      `Draft saved to drafts/...`
- [ ] The draft file says `Status: DRAFT - a person must review this. Nothing was sent.`
- [ ] The draft promises no refund, credit or amount of money
- [ ] `python eval_router.py --router both` shows at least `Accuracy: 22/24 = 92%` for your chosen router
- [ ] `python -m pytest` passes

<details>
<summary>Hint</summary>

- The complaint path is the chat path plus one file write. Start from `build_chat_chain` in `desk.py`.
- Name the files so two drafts never overwrite each other, for example a running number:
  `complaint-001.md`, `complaint-002.md`.
- If complaints and policy questions get mixed up, read the reasons under each MISS. For the semantic router, add an
  example that looks like the miss (never the test question itself). For the LLM router, add one sentence to
  `ROUTER_RULES` that says how to tell them apart.

</details>

## Stretch

- Episode 08's winning splitter ("headers + recursive") put a right chunk first for every question on its
  scoreboard. Build a second index with it, rerun `python rag.py ids`, and see where the main refund rule now ranks
  for "What is the refund window for a missed lunch?". It is the chunk that starts "Report a missed lunch or dinner
  within 6 hours" (`refunds.md#4` in Episode 07's index). A different splitter numbers the chunks differently, so look
  for that text, not the tag. On Episode 08's scoreboard it moved from 8th to 3rd with this splitter; check whether
  your index agrees.
- Print the total tokens of `python desk.py --demo` for both routers and work out what the LLM router costs per
  1,000 questions on your plan.

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is
live, so try it yourself first.
