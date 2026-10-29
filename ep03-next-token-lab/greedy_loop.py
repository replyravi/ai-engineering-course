"""
greedy_loop.py - the generation loop, written by hand (steps 5 and 6).

A decoder writes like this, again and again:
  1. read the whole text so far (one forward pass)
  2. take the scores at the LAST position, one per vocabulary entry
  3. a decoding rule picks ONE token (greedy: the highest score)
  4. stick that token on the end and go back to 1
It stops when it writes the end-of-text token, or when it reaches the max-tokens limit.

    python greedy_loop.py          step 5: 20 greedy steps, one line per step
    python greedy_loop.py stop     step 6: the two stop rules

Try:  python greedy_loop.py --steps 80      (watch it get stuck in a loop)
"""

import argparse

import numpy as np

from attention import softmax
from decoder import load_decoder, next_token_logits, piece, read_prompts

PROMPTS = read_prompts()
END_OF_TEXT = "end-of-text token"
MAX_TOKENS = "max tokens"
SHOW_STEPS = 20          # print one line for each of the first 20 steps


def greedy_pick(logits: np.ndarray) -> int:
    """Greedy decoding: always take the token with the highest score."""
    return int(np.argmax(logits))


def generate(next_logits, prompt_ids: list[int], max_new_tokens: int, eos_id=None, pick=greedy_pick, on_step=None):
    """The generation loop. Returns (new token IDs, which stop rule fired).

    next_logits(ids) -> scores for the next token (one forward pass of the model)
    pick(logits)     -> one token ID (the decoding rule: greedy here, sampling in step 7)
    on_step(step, token, logits) is called after every pick, so we can print what happens.
    """
    ids = list(prompt_ids)
    new_ids = []
    for step in range(1, max_new_tokens + 1):
        logits = next_logits(ids)            # read everything written so far
        token = pick(logits)                 # choose ONE next token
        if on_step is not None:
            on_step(step, token, logits)
        if token == eos_id:                  # stop rule 1: the model wrote "the end"
            return new_ids, END_OF_TEXT
        ids.append(token)                    # write it down, then read again
        new_ids.append(token)
    return new_ids, MAX_TOKENS               # stop rule 2: we hit our limit


def find_loop(ids: list[int], min_size: int = 4):
    """Does the text end by repeating itself? Returns (tokens in the repeating piece, times in a row), or None."""
    for size in range(min_size, len(ids) // 2 + 1):
        if ids[-size:] == ids[-2 * size:-size]:
            times = 2
            while len(ids) >= (times + 1) * size and ids[-(times + 1) * size:-times * size] == ids[-size:]:
                times += 1
            return size, times
    return None


def library_greedy(tokenizer, model, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """The same job done by the library: model.generate() with sampling switched off."""
    import torch

    ids = torch.tensor([prompt_ids])
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return out[0, len(prompt_ids):].tolist()


def tail(text: str, width: int = 46) -> str:
    """The end of a long text, so each printed line stays short."""
    shown = repr(text)
    return shown if len(shown) <= width else "..." + shown[-(width - 3):]


# ---------------------------------------------------------------------------
# Step 5 - greedy loop by hand
# ---------------------------------------------------------------------------

def show_greedy(tokenizer, model, prompt: str, steps: int) -> None:
    prompt_ids = tokenizer(prompt)["input_ids"]
    written = list(prompt_ids)
    print(f"Prompt: {prompt!r}")
    print(f"Greedy: always take the most likely next token. {steps} steps. Each step re-reads the whole text.\n")
    print(f"  {'step':>4}  {'picked':<14}{'prob':>7}   text so far")

    def on_step(step, token, logits):
        written.append(token)
        if step <= SHOW_STEPS:
            prob = softmax(logits)[token]
            print(f"  {step:>4}  {piece(tokenizer, token):<14}{prob:>7.1%}   {tail(tokenizer.decode(written))}")
        elif step == SHOW_STEPS + 1:
            print(f"  ... steps {SHOW_STEPS + 1} to {steps} are not printed")

    new_ids, rule = generate(lambda ids: next_token_logits(model, ids), prompt_ids, steps,
                             eos_id=tokenizer.eos_token_id, on_step=on_step)
    print(f"\nStopped by: {rule} ({len(new_ids)} new tokens)")
    print(f"Final text: {tokenizer.decode(prompt_ids + new_ids)!r}")

    loop = find_loop(new_ids)
    if loop:
        size, times = loop
        print(f"Stuck in a loop: the last {size} tokens are repeated {times} times in a row:")
        print(f"  {tokenizer.decode(new_ids[-size:])!r}")
    else:
        print("No repeating loop at the end (yet).")

    same = library_greedy(tokenizer, model, prompt_ids, steps) == new_ids
    print(f"model.generate(do_sample=False) picks the same {len(new_ids)} tokens: {same}")


# ---------------------------------------------------------------------------
# Step 6 - the two stop rules
# ---------------------------------------------------------------------------

def show_stop(tokenizer, model, prompt: str, steps: int) -> None:
    eos = tokenizer.eos_token_id
    print(f"Stop rule 1: the model writes the end-of-text token {tokenizer.eos_token} (ID {eos}).")
    print("Stop rule 2: we reach max_new_tokens, the limit we set.\n")
    runs = [(PROMPTS["ends_itself"][0], 40), (PROMPTS["runs_on"][0], 15)]
    for text, limit in runs:
        prompt_ids = tokenizer(text)["input_ids"]
        last_step = []
        new_ids, rule = generate(lambda ids: next_token_logits(model, ids), prompt_ids, limit, eos_id=eos,
                                 on_step=lambda step, token, logits: last_step.append(step))
        print(f"Prompt: {text!r}   max_new_tokens={limit}")
        print(f"  wrote {len(new_ids)} tokens: {tokenizer.decode(new_ids)!r}")
        where = f" at step {last_step[-1]}" if rule == END_OF_TEXT else ""
        print(f"  stop rule that fired: {rule}{where}\n")
    print("The first reply ends like a web page ('Leave a Reply'): a base model copies the shape of the")
    print("text it was trained on, including where such text usually ends.")


STEPS = {"greedy": show_greedy, "stop": show_stop}


def main(argv=None, tokenizer=None, model=None) -> int:
    parser = argparse.ArgumentParser(description="Episode 03, steps 5-6: the generation loop by hand")
    parser.add_argument("step", nargs="?", default="greedy", choices=list(STEPS))
    parser.add_argument("--prompt", default=PROMPTS["cold_open"][0])
    parser.add_argument("--steps", type=int, default=20, help="max new tokens (step 5)")
    args = parser.parse_args(argv)
    if model is None:
        tokenizer, model = load_decoder()
    STEPS[args.step](tokenizer, model, args.prompt, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
