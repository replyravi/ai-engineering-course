# Your Turn: give the agent a second tool

Right now the agent has one tool, `days_until_holiday`. Give it a **second** one and make it choose the right
tool on its own.

## The task

1. Pick one:
   - **A timetable lookup.** Write `data/menu.csv` yourself: which sabzi Dabbaloop serves for lunch and for
     dinner on each weekday (invent it). Add a tool `todays_menu(day: str, meal: str = "lunch") -> str`.
   - **A unit converter.** Add a tool `convert(value: float, from_unit: str, to_unit: str) -> str` for one kind
     of unit, for example grams and kilograms for a bulk order.
2. Write a clear docstring and typed arguments. Remember: that is all the model sees.
3. Keep the tool **read-only**, and return problems as text (`"Error: ..."`) instead of raising them.
4. Add it to `create_agent(..., tools=[days_until_holiday, your_tool], ...)` in `rung5_tool.py`, and add one
   sentence about it to `AGENT_RULES`.
5. Write **five** test questions: two for each tool and one that needs no tool at all. Run each one with
   `python rung5_tool.py --trace "your question"` and check which tool was called.
6. Add an offline test in `tests/test_agent.py` with a scripted `tool_call("your_tool", {...})`, like
   `test_the_loop_is_human_ai_tool_ai`. Run `python -m pytest`: everything still passes.

Post your five questions and which tool the agent picked for each in the video comments.

## Done when

- [ ] `python rung5_tool.py --tool-only` (or your own small check) shows your tool's name, description and arguments
- [ ] Your five questions: the agent picks the right tool for all four tool questions, and no tool for the fifth
- [ ] Each trace shows human, AI with a tool call, tool result, final AI (or just human, AI for the fifth)
- [ ] `python -m pytest` passes, with your new test included

<details>
<summary>Hint</summary>

- If the agent picks the wrong tool, the problem is almost always the **descriptions**. Make each one say
  when to use it and when not to ("Use this only for food menus, not for dates").
- Test your tool by hand first, without any model: `your_tool.invoke({"day": "Monday"})`. A tool that is wrong on
  its own will be wrong inside the agent too.
- If the default model struggles to choose, try `LLM_MODEL=gemini-3.8-flash` in `.env` (smarter, slower, uses
  more free quota) and compare. Note the difference in your comment.

</details>

## Stretch

Give `ladder.py` a sixth question that needs your new tool: add it to `QUESTIONS` with `"kind": "fact"` and a
`"must_contain"` word the right answer must include, and add your tool to the agent. Does the ladder stop at
**Tools** for it?

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is
live, so try it yourself first.
