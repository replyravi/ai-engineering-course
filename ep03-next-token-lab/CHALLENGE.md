# Your Turn: break the loop

In step 5, `python greedy_loop.py --steps 80` got stuck. Greedy decoding wrote
`The best way to make kareena is to make it with a little lemon juice and water.` and then the same line again,
and `find_loop()` reported `the last 22 tokens are repeated 2 times in a row`.

**The task:** add a **repetition penalty** to the hand-written loop and show that the loop is gone.

A repetition penalty makes every token that is already in the text a little less likely. The rule most libraries
use comes from the CTRL paper (2019): a positive score is **divided** by the penalty, a negative score is
**multiplied** by it. A penalty of 1.0 changes nothing; the paper suggests about 1.2.

Post your result in the comments, for example: `penalty 1.0: loop of 22 tokens | penalty 1.2: no loop | same as generate: True`.

## Steps

1. Create `challenge.py`. Write `repetition_penalty(logits, ids, penalty)`: it takes the scores for the next token
   and the token IDs written so far (prompt included), and returns new scores where every token that appears in
   `ids` is penalised. Each token is penalised once, however often it appears.
2. You do not need to change `generate()` from `greedy_loop.py`. Wrap the model instead:
   `next_logits = lambda ids: repetition_penalty(next_token_logits(model, ids), ids, 1.2)`.
3. Run 80 greedy steps on `Chai tastes best when` with penalty 1.0 and with 1.2. Print the text and `find_loop()` for both.
4. Check your function against the library: `model.generate(..., do_sample=False, repetition_penalty=1.2)` should
   pick the **same 80 tokens** as your loop. (`library_greedy()` in `greedy_loop.py` shows how to call `generate`.)
5. Add a test to `tests/test_greedy.py` with a scripted fake model (like `scripted()` there): a model that always
   prefers token 3 by a small margin should pick its second choice once 3 is already in the text.

<details>
<summary>Hint (the skeleton, not the full answer)</summary>

```python
import numpy as np

from decoder import load_decoder, next_token_logits, read_prompts
from greedy_loop import find_loop, generate


def repetition_penalty(logits, ids, penalty):
    out = logits.copy()
    seen = np.unique(ids)              # every token already in the text, once
    # positive scores: divide by penalty. negative scores: multiply by penalty. (np.where helps)
    ...
    return out


tokenizer, model = load_decoder()
prompt_ids = tokenizer(read_prompts()["cold_open"][0])["input_ids"]
for penalty in [1.0, 1.2]:
    new_ids, _ = generate(lambda ids: repetition_penalty(next_token_logits(model, ids), ids, penalty),
                          prompt_ids, 80, eos_id=tokenizer.eos_token_id)
    print(penalty, find_loop(new_ids), repr(tokenizer.decode(new_ids)))
```

</details>

## Think about it

- Why divide positive scores but multiply negative ones? What would dividing a negative score do?
- The penalty also hits words a sentence needs again and again, like `the` and `.`. What happens to the text with a penalty of 2.0? Try it.
- Step 7 broke the loop another way: sampling. Which one would you use for a chatbot, and why?

## Stretch goal: beat the model

Write `beat_the_model.py`, a small game. Show a half sentence, let the player type a guess for the next word,
**then** reveal the model's top 5 next tokens with their probabilities (reuse `top_tokens()` from `next_token.py`
and `softmax()` from `attention.py`). Score 3 points for the model's top token, 1 point for anything else in the top 5.
Then append the model's top token and play the next round. Five rounds per game.

Careful with spaces: the model's tokens usually start with a space (`' it'`), and your player types `it`.
Compare `token.strip().lower()` with the guess.
