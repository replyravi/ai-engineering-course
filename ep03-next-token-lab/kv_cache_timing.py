"""
kv_cache_timing.py - measure what the KV cache saves (step 9).

Our loop so far re-reads the WHOLE text at every step. But the keys and values of the old tokens
never change (the causal mask means old tokens never look at new ones). So a real decoder keeps
them on a shelf, the KV cache, and each step only computes the ONE new token.

We write 100 tokens both ways, greedy, and time them:
    without cache: step n reads all prompt + n tokens again
    with cache:    step 1 reads the prompt, every later step reads just one token

Run:  python kv_cache_timing.py           (your times will differ: they depend on your computer)
Try:  python kv_cache_timing.py --tokens 200
"""

import argparse
import time

import torch

from decoder import load_decoder, read_prompts

NEW_TOKENS = 100


def write_without_cache(model, prompt_ids: list[int], new_tokens: int) -> list[int]:
    """Greedy loop that re-reads everything at every step (what we did in step 5)."""
    ids = torch.tensor([prompt_ids])
    for _ in range(new_tokens):
        with torch.inference_mode():
            logits = model(ids, use_cache=False).logits
        next_id = logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, len(prompt_ids):].tolist()


def write_with_cache(model, prompt_ids: list[int], new_tokens: int):
    """Greedy loop with the KV cache. Returns (new token IDs, the cache)."""
    with torch.inference_mode():
        out = model(torch.tensor([prompt_ids]), use_cache=True)     # read the prompt once
    cache = out.past_key_values                                      # keys and values of every token so far
    next_id = out.logits[0, -1].argmax().view(1, 1)
    new_ids = [int(next_id)]
    for _ in range(new_tokens - 1):
        with torch.inference_mode():
            out = model(next_id, past_key_values=cache, use_cache=True)   # feed ONE token
        cache = out.past_key_values                                   # the shelf grows by one token per layer
        next_id = out.logits[0, -1].argmax().view(1, 1)
        new_ids.append(int(next_id))
    return new_ids, cache


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def positions_read(prompt_len: int, new_tokens: int, cache: bool) -> int:
    """How many token positions go through the model in total."""
    if cache:
        return prompt_len + (new_tokens - 1)                  # the prompt once, then one token per step
    return sum(prompt_len + step for step in range(new_tokens))   # everything, again and again


def main(argv=None, tokenizer=None, model=None) -> int:
    parser = argparse.ArgumentParser(description="Episode 03, step 9: the KV cache, on and off")
    parser.add_argument("--prompt", default=read_prompts()["cold_open"][0])
    parser.add_argument("--tokens", type=int, default=NEW_TOKENS, help="new tokens to write each way")
    args = parser.parse_args(argv)
    if model is None:
        tokenizer, model = load_decoder()
    torch.manual_seed(0)
    prompt_ids = tokenizer(args.prompt)["input_ids"]
    n = args.tokens

    write_with_cache(model, prompt_ids, 5)          # warm-up, so the first timing is not unfair
    write_without_cache(model, prompt_ids, 5)

    plain, slow = timed(write_without_cache, model, prompt_ids, n)
    (cached, cache), fast = timed(write_with_cache, model, prompt_ids, n)

    print(f"Prompt: {args.prompt!r} ({len(prompt_ids)} tokens). Writing {n} new tokens, greedy, both ways.\n")
    print(f"  {'':<15}{'time':>8}{'tokens per second':>20}{'positions read':>17}")
    print(f"  {'without cache':<15}{slow:>7.2f}s{n / slow:>20.1f}{positions_read(len(prompt_ids), n, False):>17,}")
    print(f"  {'with cache':<15}{fast:>7.2f}s{n / fast:>20.1f}{positions_read(len(prompt_ids), n, True):>17,}")
    print(f"\nMeasured speed-up: {slow / fast:.1f}x faster with the cache")
    print(f"Same {n} tokens both ways: {plain == cached}")

    first = cache.layers[0]
    keys, values = first.keys, first.values
    size = sum(layer.keys.numel() + layer.values.numel() for layer in cache.layers) * keys.element_size()
    print(f"\nThe cache after writing: {len(cache.layers)} layers, each with keys {tuple(keys.shape)} "
          f"and values {tuple(values.shape)}")
    print(f"  (batch, key-value heads, tokens so far, numbers per head) -> {size / 1e6:.1f} MB in memory")
    print("Fewer positions to read is not the whole story: every step still runs all 30 layers")
    print("and reads every weight once, so the measured speed-up is smaller than the positions ratio.")
    print("The longer the text, the bigger the saving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
