# Your Turn: ask five times, trust the majority

In Step 13 the chain-of-thought row answered every note **once**. But a model's answers may change from run to run,
even with the same prompt: Google does not promise the same answer every time, and Gemini 3.5 Flash-Lite and newer
ignore `temperature`, so there is no dial to turn it up or down.
What if we ask the same note **five times** and take the answer that comes up most often?
This idea has a name: **self-consistency** (Wang and others, 2022, "Self-Consistency Improves Chain of Thought
Reasoning in Language Models").

**First, check that it can help at all.** A majority vote only changes something when the runs disagree:

```bash
python scoreboard.py --repeats 3 --only cot
```

Look at the last line: `chain of thought: varied between runs on notes [...]`. If it says `none`, your model gave the
same verdict every time, and voting cannot raise the score. That is a real finding: post it.
Want to see voting work anyway? Try the same idea on a local model, where temperature still works:
set `LLM_PROVIDER=ollama` in `.env` and build the model with `get_llm(temperature=0.8)` from `llm.py`.
(It is also possible that the verdict stays the same while the transfer lines change. The vote below looks at the answers themselves.)

**The task:** write `self_consistency.py`. For each note, ask `balance_cot_v2` five times, take the majority answer,
and compare three scores on the same ten notes:

- one run (your Step 13 chain-of-thought score)
- the majority of five runs
- how many notes had at least one right answer among the five (the best a vote could ever do)

Post your three numbers in the comments, for example: `CoT once: 7/10, majority of 5: 8/10, any of 5: 9/10 (gemini-3.5-flash-lite)`.

## Steps

1. Create `self_consistency.py` in this folder.
2. Load the notes with `load_notes()` from `common.py` and the model with `make_llm()`.
3. For each note, call `ask_transfers("balance_cot_v2", note, llm)` from `techniques.py` five times.
   That is 50 calls for ten notes (Gemini 3 cannot return several answers in one call, so every sample is a full call).
   Start with the tricky notes 3, 8 and 9 if you want to save quota.
4. Two answers can be equally right with different lines ("Arjun pays Zoya 400" or "Arjun pays Kabir 300 and Zoya 100").
   So vote on what the answer **means**: the money each person gets or pays in total.
5. Take the most common meaning, and check it with `same_result()` from `settle.py`.
6. Print the three scores.

<details>
<summary>Hint (the skeleton, not the full answer)</summary>

```python
from collections import Counter

from common import load_notes, make_llm
from settle import same_result
from techniques import ask_transfers

RUNS = 5


def meaning(transfers):
    """What an answer means: each person's total in (+) or out (-), rounded to whole rupees."""
    net = {}
    for t in transfers:
        net[t.from_person] = net.get(t.from_person, 0) - t.amount
        net[t.to_person] = net.get(t.to_person, 0) + t.amount
    return tuple(sorted((person, round(amount)) for person, amount in net.items() if round(amount)))


llm = make_llm()
for note in load_notes():
    answers = [ask_transfers("balance_cot_v2", note, llm).transfers for _ in range(RUNS)]
    answers = [a for a in answers if a is not None]           # skip runs with no usable answer
    votes = Counter(meaning(a) for a in answers)
    winner = ...                                              # the most common meaning
    majority_right = ...                                      # same_result() on one answer with that meaning
    any_right = ...                                           # was any of the five right?
```

</details>

## Think about it

- Five runs cost five times the quota and five times the waiting. When is a better answer worth that?
- Compare with the structured row of the scoreboard: there the model only reads, and Python does the maths.
  Is asking five times better than not asking the model to do arithmetic at all?
- If the five runs never disagree, what does that tell you about "creativity" settings on 2026 models?

## Stretch goals

- **Write your own ten notes.** Replace `data/trip_notes.jsonl` with notes from a real trip (change the names and
  amounts if you share it). Work out each answer by hand first, then run `python settle.py`: it tells you when your
  hand answer does not add up. Then run the scoreboard. (Some tests check our ten notes by name; update them too.)
- **A better message prompt.** Copy `prompts/message_v1.toml` to `message_v2.toml`, change a rule, and count how
  often `retry.py` needs a second attempt over all ten notes with each version.
- **Write a test:** add a test for your `meaning()` function in `tests/` that uses two different but equally right
  settle-ups for note 3 (Goa) and checks they get the same meaning. No model needed.
