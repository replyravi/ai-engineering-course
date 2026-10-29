# Your Turn: a quintal converter and 5/5 on a local model

Two changes to the Mandi Tool Belt: one new tool, and one new model.

## The task

1. **A converter tool.** In a new file `tools/units.py`, write a `@tool` called `quintals_to_kg`:

   ```python
   @tool
   def quintals_to_kg(quintals: float) -> str:
       """..."""
   ```

   One quintal is 100 kg (see `site/units.html`). Write the docstring yourself: it is the only thing the
   model reads before it decides to use your tool, so say what goes in, what comes out, and when to use it.
   Return text (for example `"3.5 quintals = 350 kg"`), and return an `"Error: ..."` text for a negative
   number instead of raising.
2. **Put it on the belt.** In `tools/__init__.py`, add a kind called `"units"` to `KINDS` and to
   `tool_belt()`. Run `python ask.py --list-tools`: the agent should now hold 6 tools.
3. **A question that needs it.** Add question 6 to `data/questions.json`, for example *"How many kilograms of
   tomatoes reached Indore on day 5?"* (arrivals are stored in quintals). Write its `truth_sql` by hand
   (hint: SQL can multiply by 100 too) and check the right answer with `python check_answers.py --truths`.
   In `check_answers.py`, give the scoreboard your new tool kind (`tools_for`).
4. **Test it offline.** In a new `tests/test_units.py`, test the tool on its own (3.5 -> 350, a negative
   number -> `Error:`). Then copy the scripted scoreboard test in `tests/test_checker.py` and add the replies
   for question 6: a `tool_call("run_select", ...)`, a `tool_call("quintals_to_kg", ...)` and the answer line.
   `python -m pytest` must pass. (A few old tests count the tools or the questions: update their numbers.)
5. **The local model.** Install Ollama (https://ollama.com), run `ollama pull qwen3.5:4b`, and set ONE line
   in the root `.env`: `LLM_PROVIDER=ollama`. Check the download size on https://ollama.com/library/qwen3.5
   before you pull it. Its model card ([Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)) gives the
   **Apache-2.0** licence (checked 2026-10-23): free to use, also commercially.
   Run `python check_answers.py --fix description`.
6. **Reach 6/6.** A small local model makes different mistakes than Gemini. Read every FAIL: the SQL it ran
   is printed under it. Improve a tool description, the system prompt in `agent.py`, or switch to
   `--fix price-tool`, until the scoreboard says 6/6. Note how many model calls and tokens each run took
   (Ollama reports token usage too; if a row says `n/r`, your Ollama version did not).

Post your final score, the model you used and the one change that helped most in the video comments.

## Done when

- `python ask.py --list-tools` shows 6 tools, and `quintals_to_kg` has a description you wrote.
- `python check_answers.py --truths` shows 6 questions with their right answers.
- `python -m pytest` passes with your new tests.
- `python check_answers.py` (with your best fix) prints `Score: 6/6` with `LLM_PROVIDER=ollama`.

<details>
<summary>Hint</summary>

If the local model never calls your converter, it probably did the multiplication inside SQL
(`arrivals_quintals * 100`). That is not wrong: the grader only checks the number. To see your tool
in the trace, ask a question where the quintals come from the user, not from the database:
*"A trader wants 3.5 quintals of onions. How many 50 kg bags is that?"*

</details>
