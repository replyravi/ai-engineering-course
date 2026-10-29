"""
next_token.py - open a real decoder and watch it choose the next token (steps 2, 3 and 4).

    python next_token.py load     step 2: the model, its vocabulary, and our prompt as tokens
    python next_token.py logits   step 3: one forward pass, one score per vocabulary entry per position
    python next_token.py          step 4: softmax over the LAST position, the top 10 next tokens

Try your own half sentence:  python next_token.py --prompt "The best thing about Mumbai is"
"""

import argparse

import numpy as np

from attention import softmax           # the same softmax you wrote in Episode 02
from decoder import MODEL_NAME, load_decoder, piece, read_prompts

PROMPT = read_prompts()["cold_open"][0]      # "Chai tastes best when"


def top_tokens(probs: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
    """The k most likely token IDs with their probabilities, best first."""
    best = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in best]


# ---------------------------------------------------------------------------
# Step 2 - load the decoder and look at its vocabulary
# ---------------------------------------------------------------------------

def show_load(tokenizer, model, prompt: str) -> None:
    config = model.config
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL_NAME} (decoder-only)")
    print(f"Learned numbers (parameters): {params:,}, stored as {str(model.dtype).replace('torch.', '')}")
    print(f"Blocks (layers): {config.num_hidden_layers}   attention heads: {config.num_attention_heads}   "
          f"numbers per token: {config.hidden_size}")
    print(f"Vocabulary: {len(tokenizer):,} tokens (IDs 0 to {len(tokenizer) - 1:,})")
    print(f"Context window: {config.max_position_embeddings:,} tokens")
    print(f"End-of-text token: {tokenizer.eos_token} (ID {tokenizer.eos_token_id})\n")

    ids = tokenizer(prompt)["input_ids"]
    print(f"Prompt: {prompt!r} -> {len(ids)} tokens")
    print(f"  {'piece':<12}{'in the vocabulary':<20}ID")
    for token_id, raw in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        print(f"  {piece(tokenizer, token_id):<12}{raw:<20}{token_id}")
    print("The vocabulary writes a leading space as 'Ġ'. So ' tastes' (with its space) is ONE token.")


# ---------------------------------------------------------------------------
# Step 3 - one forward pass: logits
# ---------------------------------------------------------------------------

def forward(tokenizer, model, prompt: str):
    """Run the model once. Returns (ids, logits) with logits shaped (1, positions, vocabulary)."""
    import torch

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        logits = model(ids).logits
    return ids[0].tolist(), logits.double().numpy()


def show_logits(tokenizer, model, prompt: str) -> None:
    ids, logits = forward(tokenizer, model, prompt)
    batch, positions, vocab = logits.shape
    print(f"Prompt: {prompt!r} ({positions} tokens)")
    print(f"logits shape: {logits.shape} = (batch {batch}, positions {positions}, vocabulary {vocab:,})")
    print(f"That is {positions * vocab:,} scores: one for every vocabulary entry, at every position.\n")

    print("Every position guesses the token that comes AFTER it, all in one pass:")
    print(f"  {'pos':<5}{'text so far':<26}{'top guess':<12}{'real next token':<17}")
    for pos in range(positions):
        so_far = tokenizer.decode(ids[: pos + 1])
        best = int(np.argmax(logits[0, pos]))
        if pos + 1 < positions:
            real = ids[pos + 1]
            verdict = f"{piece(tokenizer, real):<11}{'right' if best == real else 'wrong'}"
        else:
            verdict = "(not written yet)"
        print(f"  {pos:<5}{so_far!r:<26}{piece(tokenizer, best):<12}{verdict}")
    print("To WRITE, we only need the last row. The other rows are how a decoder trains:")
    print("every guess is checked against the real next token, all positions at once.\n")

    # The causal mask, seen from outside: change the LAST token, the earlier rows do not move.
    import torch

    changed = ids[:-1] + [tokenizer(" if")["input_ids"][0]]
    with torch.inference_mode():
        other = model(torch.tensor([changed])).logits.double().numpy()
    gap = float(np.abs(other[0, :-1] - logits[0, :-1]).max())
    print(f"Swap the last token {piece(tokenizer, ids[-1])} for {piece(tokenizer, changed[-1])}:")
    print(f"  scores at positions 0-{positions - 2} stay the same: {gap < 1e-4} (biggest change {gap:.1e})")
    print(f"  scores at position {positions - 1} change: {not np.allclose(other[0, -1], logits[0, -1])}")


# ---------------------------------------------------------------------------
# Step 4 - softmax over the last position: the top 10 next tokens
# ---------------------------------------------------------------------------

def show_top10(tokenizer, model, prompt: str) -> None:
    ids, logits = forward(tokenizer, model, prompt)
    last = logits[0, -1]                      # (vocabulary,): the scores for the NEXT token
    probs = softmax(last)
    print(f"Prompt: {prompt!r}")
    print(f"Scores (logits) at the last position: {last.shape[0]:,} numbers, "
          f"from {last.min():.1f} to {last.max():.1f}")
    print(f"After softmax: {probs.shape[0]:,} probabilities, all positive, sum = {probs.sum():.4f}\n")

    top = top_tokens(probs, 10)
    print(f"  {'rank':<6}{'token':<14}{'ID':>7}{'logit':>9}{'probability':>13}")
    for rank, (token_id, p) in enumerate(top, start=1):
        print(f"  {rank:<6}{piece(tokenizer, token_id):<14}{token_id:>7}{last[token_id]:>9.2f}{p:>12.1%}")
    covered = sum(p for _, p in top)
    print(f"\nThe top 10 hold {covered:.1%} of the probability. "
          f"The other {probs.shape[0] - 10:,} tokens share {1 - covered:.1%}.")
    print(f"Greedy decoding would pick {piece(tokenizer, top[0][0])}. "
          "The pick comes from these probabilities and a decoding rule, not from attention scores.")


STEPS = {"load": show_load, "logits": show_logits, "top10": show_top10}


def main(argv=None, tokenizer=None, model=None) -> int:
    parser = argparse.ArgumentParser(description="Episode 03, steps 2-4: inside one next-token choice")
    parser.add_argument("step", nargs="?", default="top10", choices=list(STEPS))
    parser.add_argument("--prompt", default=PROMPT, help="the half sentence the model continues")
    args = parser.parse_args(argv)
    if model is None:
        tokenizer, model = load_decoder()
    STEPS[args.step](tokenizer, model, args.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
