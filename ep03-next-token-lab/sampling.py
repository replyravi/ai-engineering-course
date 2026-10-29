"""
sampling.py - decoding rules that add a little chance: temperature, top-k and top-p (steps 7 and 8).

Greedy always takes the top token, so the same prompt always gives the same text, and it can loop.
Sampling draws the next token at random, weighted by its probability. Three knobs shape the draw:

    apply_temperature(logits, t)   t < 1 sharpens (safer), t > 1 flattens (wilder)
    top_k_filter(logits, k)        keep only the k best tokens
    top_p_filter(logits, p)        keep the smallest group of best tokens whose probabilities reach p
    sample(logits, rng)            one weighted random draw
    make_picker(...)               all of the above as one decoding rule for greedy_loop.generate()

The top half is pure NumPy (tests/test_sampling.py checks it on fixed numbers). The bottom half runs the demos:

    python sampling.py temperature    step 7: three temperatures, a fixed seed
    python sampling.py cut            step 8: top-k and top-p by hand, checked against transformers
"""

import argparse

import numpy as np

from attention import softmax

SEED = 42


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide every score by the temperature. Small t makes big scores bigger; large t squashes them together."""
    if temperature <= 0:
        raise ValueError("temperature must be above 0. For 'always the top token' use greedy_pick().")
    return logits / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k highest scores. Every other score becomes -inf, so softmax gives it 0."""
    if k < 1:
        raise ValueError("k must be at least 1")
    k = min(k, logits.shape[-1])
    kth_best = np.sort(logits)[-k]                   # the k-th highest score
    return np.where(logits >= kth_best, logits, -np.inf)   # a tie with the k-th score is kept too


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Keep the best tokens until their probabilities add up to p (the 'nucleus'). The rest become -inf."""
    if not 0 < p <= 1:
        raise ValueError("p must be between 0 and 1")
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]                  # token IDs, best first
    before = np.cumsum(probs[order]) - probs[order]  # how much probability the better tokens already hold
    keep = order[before < p]                         # keep a token while the running sum is still below p
    filtered = np.full_like(logits, -np.inf)
    filtered[keep] = logits[keep]
    return filtered


def kept(scores: np.ndarray) -> int:
    """How many tokens are still allowed (not -inf)."""
    return int(np.isfinite(scores).sum())


def sample(logits: np.ndarray, rng) -> int:
    """One random token, drawn with the softmax probabilities as weights."""
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def make_picker(temperature: float = 1.0, top_k=None, top_p=None, rng=None):
    """A decoding rule for greedy_loop.generate(): temperature, then top-k, then top-p, then a draw.

    That is the same order transformers uses inside model.generate().
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)

    def pick(logits: np.ndarray) -> int:
        scores = apply_temperature(logits, temperature)
        if top_k is not None:
            scores = top_k_filter(scores, top_k)
        if top_p is not None and top_p < 1.0:
            scores = top_p_filter(scores, top_p)
        return sample(scores, rng)

    return pick


# ===========================================================================
# Demos (python sampling.py [temperature | cut]) - these load the real decoder
# ===========================================================================

TEMPERATURES = [0.2, 0.8, 1.5]
NEW_TOKENS = 25


def tokens_for(probs: np.ndarray, share: float = 0.9) -> int:
    """How many of the best tokens you need before their probabilities add up to `share`."""
    return int(np.searchsorted(np.cumsum(np.sort(probs)[::-1]), share) + 1)


def continue_with(tokenizer, model, prompt_ids, picker, new_tokens: int) -> str:
    from decoder import next_token_logits
    from greedy_loop import generate

    new_ids, _ = generate(lambda ids: next_token_logits(model, ids), prompt_ids, new_tokens,
                          eos_id=tokenizer.eos_token_id, pick=picker)
    return tokenizer.decode(new_ids)


def demo_temperature(tokenizer, model, prompt: str) -> None:
    from decoder import next_token_logits, piece

    prompt_ids = tokenizer(prompt)["input_ids"]
    logits = next_token_logits(model, prompt_ids)
    columns = TEMPERATURES[:2] + [1.0] + TEMPERATURES[2:]          # 0.2, 0.8, 1.0 (unchanged), 1.5
    probs = {t: softmax(apply_temperature(logits, t)) for t in columns}
    top5 = np.argsort(probs[1.0])[::-1][:5]
    print(f"Prompt: {prompt!r}\n")
    print("Temperature reshapes the next-token probabilities (the 5 best tokens at t=1.0):")
    print(f"  {'token':<12}" + "".join(f"{'t=' + str(t):>9}" for t in columns))
    for token_id in top5:
        print(f"  {piece(tokenizer, int(token_id)):<12}" + "".join(f"{probs[t][token_id]:>9.1%}" for t in columns))
    print(f"  {'tokens for 90%':<12}" + "".join(f"{tokens_for(probs[t]):>9,}" for t in columns))
    print("Low t: one token takes almost everything. High t: the probability spreads over many tokens.\n")

    print(f"Sampling {NEW_TOKENS} new tokens, seed {SEED}, no top-k or top-p:")
    first_run = {}
    for t in TEMPERATURES:
        text = continue_with(tokenizer, model, prompt_ids, make_picker(t, rng=np.random.default_rng(SEED)), NEW_TOKENS)
        first_run[t] = text
        print(f"  t={t:<4} {text!r}")
    again = all(continue_with(tokenizer, model, prompt_ids, make_picker(t, rng=np.random.default_rng(SEED)),
                              NEW_TOKENS) == first_run[t] for t in TEMPERATURES)
    print(f"\nRun all three again with seed {SEED}: the same text: {again}")


def demo_cut(tokenizer, model, prompt: str) -> None:
    import torch
    from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

    from decoder import next_token_logits, piece

    prompt_ids = tokenizer(prompt)["input_ids"]
    logits = next_token_logits(model, prompt_ids)
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    print(f"Prompt: {prompt!r}   (temperature 1.0)\n")

    k = 5
    cut_k = top_k_filter(logits, k)
    new_probs = softmax(cut_k)
    print(f"Top-k, k={k}: keep the {k} best tokens, then share 100% among them")
    for token_id in order[:k]:
        print(f"  {piece(tokenizer, int(token_id)):<12}{probs[token_id]:>7.1%}  ->{new_probs[token_id]:>7.1%}")
    print(f"  tokens kept: {kept(cut_k)} of {len(logits):,}\n")

    p = 0.9
    cut_p = top_p_filter(logits, p)
    n_kept = kept(cut_p)
    running = np.cumsum(probs[order])
    print(f"Top-p, p={p}: keep the best tokens until their running sum reaches {p}")
    print(f"  {'rank':>4}  {'token':<12}{'prob':>7}{'running sum':>13}")
    for rank in list(range(1, 6)) + [n_kept]:
        if rank == n_kept:
            print("   ...")
        token_id = int(order[rank - 1])
        print(f"  {rank:>4}  {piece(tokenizer, token_id):<12}{probs[token_id]:>7.2%}{running[rank - 1]:>12.1%}")
    print(f"  tokens kept: {n_kept}. With p=0.5 only {kept(top_p_filter(logits, 0.5))} would stay.")
    print("  Top-p adapts: when the model is sure, few tokens pass; when it is unsure, many do.\n")

    scores = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)   # transformers works on torch tensors
    ids = torch.tensor([prompt_ids])
    lib_k = TopKLogitsWarper(top_k=k)(ids, scores.clone())[0].isfinite().numpy()
    lib_p = TopPLogitsWarper(top_p=p)(ids, scores.clone())[0].isfinite().numpy()
    print("Checked against transformers' own TopKLogitsWarper and TopPLogitsWarper:")
    print(f"  top-k={k}: the same {lib_k.sum()} tokens kept: {bool((lib_k == np.isfinite(cut_k)).all())}")
    print(f"  top-p={p}: the same {lib_p.sum()} tokens kept: {bool((lib_p == np.isfinite(cut_p)).all())}\n")

    settings = dict(temperature=0.8, top_k=50, top_p=0.9)
    print(f"Same settings, two implementations ({', '.join(f'{n}={v}' for n, v in settings.items())}, 20 new tokens):")
    rng = np.random.default_rng(SEED)
    for i in range(3):
        text = continue_with(tokenizer, model, prompt_ids, make_picker(**settings, rng=rng), 20)
        print(f"  ours, run {i + 1}:      {text!r}")
    torch.manual_seed(SEED)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, max_new_tokens=20,
                         num_return_sequences=3, pad_token_id=tokenizer.eos_token_id, **settings)
    for i, row in enumerate(out):
        print(f"  generate, run {i + 1}:  {tokenizer.decode(row[len(prompt_ids):], skip_special_tokens=True)!r}")
    print("The words differ: each side uses its own random numbers. The rules, and the style, are the same.")


DEMOS = {"temperature": demo_temperature, "cut": demo_cut}


def main(argv=None, tokenizer=None, model=None) -> int:
    from decoder import load_decoder, read_prompts

    parser = argparse.ArgumentParser(description="Episode 03, steps 7-8: temperature, top-k and top-p")
    parser.add_argument("demo", nargs="?", default="temperature", choices=list(DEMOS))
    parser.add_argument("--prompt", default=read_prompts()["cold_open"][0])
    args = parser.parse_args(argv)
    if model is None:
        tokenizer, model = load_decoder()
    DEMOS[args.demo](tokenizer, model, args.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
