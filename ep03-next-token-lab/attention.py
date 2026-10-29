# Copied from Episode 02 (ep02-attention-xray/attention.py) so this folder runs on its own.
# Everything below these comment lines is unchanged. Episode 03 adds the causal mask in its own
# file, causal_mask.py, through the mask= argument that is already here.
# tests/test_mask.py checks that this copy still matches Episode 02's file.

"""
attention.py - self-attention in plain NumPy, written from scratch (steps 9, 10 and 11).

The top half is a small library. Episode 03 imports it and adds a causal mask:

    from attention import scaled_dot_product_attention, multi_head_attention, positional_encoding

    softmax(scores)                                   turn scores into weights that sum to 1
    scaled_dot_product_attention(Q, K, V, mask=None)  -> (output, weights)
    split_heads(x, num_heads), combine_heads(x)       reshape for multi-head attention
    multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None)  -> (output, weights of every head)
    layer_norm(x), encoder_block(X, params, num_heads, mask=None)  one BERT-style encoder block
    random_block_params(d_model, d_ff, rng)           made-up weights for the demos and tests
    positional_encoding(num_positions, d_model)       re-exported from positions.py (step 7)

Shapes: the last two axes are (tokens, numbers). Extra axes in front, like a batch, are carried
along, so the same code works for one sentence (n, d) and for a batch (b, n, d).

Masks follow Hugging Face's attention_mask: True or 1 = "may look at this key", False or 0 = "hidden".
A mask must broadcast to the score matrix (..., n_queries, n_keys).

The bottom half runs the demos:
    python attention.py            step 9:  attention from scratch on 4 tokens
    python attention.py scaling    step 10: why we divide by the square root of d_k
    python attention.py heads      step 11: two heads, then one whole encoder block
"""

import argparse

import numpy as np

from positions import positional_encoding

__all__ = ["softmax", "scaled_dot_product_attention", "split_heads", "combine_heads",
           "multi_head_attention", "layer_norm", "encoder_block", "random_block_params", "positional_encoding"]


# ---------------------------------------------------------------------------
# Step 9 - scaled dot-product attention
# ---------------------------------------------------------------------------

def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Turn any row of scores into positive weights that sum to 1."""
    shifted = scores - np.max(scores, axis=axis, keepdims=True)   # same answer, but no overflow for big scores
    exps = np.exp(shifted)                                        # exp(-inf) = 0, so hidden keys get weight 0
    return exps / np.sum(exps, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None, scale: bool = True):
    """softmax(Q @ K.T / sqrt(d_k)) @ V.  Returns (output, weights).

    Q: (..., n_queries, d_k)   K: (..., n_keys, d_k)   V: (..., n_keys, d_v)
    weights: (..., n_queries, n_keys), every row sums to 1
    output:  (..., n_queries, d_v), each row is a weighted mix of the rows of V
    mask:    optional, True/1 = may look, False/0 = hidden (gets weight exactly 0)
    scale:   False skips the division by sqrt(d_k) (only to see why we need it, step 10)
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2)              # how well each query matches each key
    if scale:
        scores = scores / np.sqrt(d_k)               # keep scores in a range where softmax stays smooth
    if mask is not None:
        allowed = np.broadcast_to(np.asarray(mask, dtype=bool), scores.shape)
        if not allowed.any(axis=-1).all():
            raise ValueError("Every query must be allowed to look at one key at least (a mask row is all False).")
        scores = np.where(allowed, scores, -np.inf)  # hidden keys: minus infinity, so softmax gives them 0
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


# ---------------------------------------------------------------------------
# Step 11 - multi-head attention and one encoder block
# ---------------------------------------------------------------------------

def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """(..., n, d_model) -> (..., num_heads, n, d_model // num_heads): each head gets its own slice of numbers."""
    *front, n, d_model = x.shape
    if d_model % num_heads:
        raise ValueError(f"d_model={d_model} cannot be split evenly into {num_heads} heads")
    x = x.reshape(*front, n, num_heads, d_model // num_heads)
    return np.swapaxes(x, -3, -2)


def combine_heads(x: np.ndarray) -> np.ndarray:
    """(..., num_heads, n, d_head) -> (..., n, num_heads * d_head): glue the heads side by side again."""
    x = np.swapaxes(x, -3, -2)
    *front, n, num_heads, d_head = x.shape
    return x.reshape(*front, n, num_heads * d_head)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads: int, mask=None):
    """Several attention heads side by side. Returns (output, weights).

    X: (..., n, d_model). Wq, Wk, Wv, Wo: (d_model, d_model).
    output: (..., n, d_model), the same width as X.  weights: (..., num_heads, n, n).
    mask: (n_keys,), (n, n), or for a batch (batch, n, n) or (batch, 1, n_keys).
    """
    Q, K, V = (split_heads(X @ W, num_heads) for W in (Wq, Wk, Wv))   # (..., heads, n, d_head)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim >= 2:
            mask = np.expand_dims(mask, axis=-3)    # (..., 1, n, n): the same mask for every head
    heads, weights = scaled_dot_product_attention(Q, K, V, mask)
    return combine_heads(heads) @ Wo, weights                         # concatenate, then project


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Give every token's numbers mean 0 and spread 1 (real models also learn a scale and shift)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def encoder_block(X, params: dict, num_heads: int, mask=None) -> np.ndarray:
    """One encoder block, BERT style: attention -> add & norm -> feed-forward -> add & norm."""
    attended, _ = multi_head_attention(X, params["Wq"], params["Wk"], params["Wv"], params["Wo"], num_heads, mask)
    X = layer_norm(X + attended)                              # add: the skip path keeps the original signal
    hidden = np.maximum(0, X @ params["W1"])                  # feed-forward, each token on its own (ReLU)
    return layer_norm(X + hidden @ params["W2"])              # add & norm again


def random_block_params(d_model: int, d_ff: int, rng) -> dict:
    """Made-up weights for one block (a trained model learns these). Scaled so the numbers stay moderate."""
    def matrix(rows, cols):
        return rng.normal(size=(rows, cols)) / np.sqrt(rows)

    params = {name: matrix(d_model, d_model) for name in ["Wq", "Wk", "Wv", "Wo"]}
    params["W1"], params["W2"] = matrix(d_model, d_ff), matrix(d_ff, d_model)
    return params


# ===========================================================================
# Demos (python attention.py [scratch | scaling | heads])
# ===========================================================================

SEED = 42
TOKENS = ["the", "dosa", "was", "crisp"]


def show(name: str, matrix: np.ndarray, labels=None) -> None:
    """Print a small matrix with 2 decimals, rows labelled by token."""
    print(f"{name} {matrix.shape}")
    if labels is not None and matrix.shape[-1] == len(labels):
        print("        " + "".join(f"{t:>7}" for t in labels))
    for label, row in zip(labels or [""] * len(matrix), matrix):
        print(f"  {label:<6}" + "".join(f"{x:>7.2f}" for x in row))


def demo_scratch() -> None:
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(4, 8))                     # 4 tokens x 8 numbers (pretend embeddings + positions)
    Wq, Wk, Wv = (rng.normal(size=(8, 4)) for _ in range(3))   # learned in a real model, random here
    print(f"Tokens: {TOKENS}   X {X.shape}   Wq, Wk, Wv {Wq.shape} each\n")

    Q, K, V = X @ Wq, X @ Wk, X @ Wv                # every token: a query, a key and a value
    print(f"Q = X @ Wq {Q.shape}   K = X @ Wk {K.shape}   V = X @ Wv {V.shape}\n")

    scores = Q @ K.T                                # 4 x 4: query of row i against key of column j
    scaled = scores / np.sqrt(K.shape[-1])          # divide by sqrt(d_k) = sqrt(4) = 2
    weights = softmax(scaled)                       # each row becomes percentages
    show("scores = Q @ K.T", scores, TOKENS)
    print()
    show("weights = softmax(scores / sqrt(4))", weights, TOKENS)
    print(f"row sums: {weights.sum(axis=1).round(6).tolist()}\n")

    i = TOKENS.index("dosa")                        # follow one query, like the animation
    print(f"The query of {TOKENS[i]!r} meets the four keys:")
    print("  scores:     " + "  ".join(f"{t} {s:.2f}" for t, s in zip(TOKENS, scores[i])))
    print("  / 2:        " + "  ".join(f"{t} {s:.2f}" for t, s in zip(TOKENS, scaled[i])))
    print("  softmax:    " + "  ".join(f"{t} {w:.0%}" for t, w in zip(TOKENS, weights[i]))
          + f"   (total {weights[i].sum():.0%})")

    output = weights @ V                            # 4 x 4: a weighted mix of the value rows
    print(f"\noutput = weights @ V {output.shape}")
    mix = sum(w * v for w, v in zip(weights[i], V))
    print(f"Row {TOKENS[i]!r} of the output equals the weighted mix of the four value rows: {np.allclose(output[i], mix)}")

    out_lib, w_lib = scaled_dot_product_attention(Q, K, V)
    print(f"scaled_dot_product_attention() gives the same answer: {np.allclose(out_lib, output) and np.allclose(w_lib, weights)}")

    # Attention alone is order-blind: reverse the tokens and every token's output stays the same.
    flipped = X[::-1]
    out_flipped, _ = scaled_dot_product_attention(flipped @ Wq, flipped @ Wk, flipped @ Wv)
    print(f"\nReverse the 4 tokens: each token's output is unchanged: {np.allclose(out_flipped[::-1], output)}")
    P = positional_encoding(4, 8)                   # step 7: add a position pattern first
    with_pos = (X + P) @ Wq, (X + P) @ Wk, (X + P) @ Wv
    flipped_pos = (flipped + P) @ Wq, (flipped + P) @ Wk, (flipped + P) @ Wv
    same = np.allclose(scaled_dot_product_attention(*flipped_pos)[0][::-1], scaled_dot_product_attention(*with_pos)[0])
    print(f"Add positions first, then reverse: each token's output is unchanged: {same} (order matters now)")


def demo_scaling() -> None:
    rng = np.random.default_rng(SEED)
    n, sentences = 6, 500
    print(f"{sentences} random sentences of {n} tokens. Q and K numbers are random with spread 1.\n")
    print(f"  {'d_k':>4}  {'spread of Q.K scores':>20}  {'biggest weight in a row (average)':>34}")
    print(f"  {'':>4}  {'':>20}  {'unscaled':>15}  {'divided by sqrt(d_k)':>20}")
    for d_k in [4, 64, 512]:
        Q = rng.normal(size=(sentences, n, d_k))
        K = rng.normal(size=(sentences, n, d_k))
        V = rng.normal(size=(sentences, n, d_k))
        scores = Q @ np.swapaxes(K, -1, -2)
        _, unscaled = scaled_dot_product_attention(Q, K, V, scale=False)
        _, scaled = scaled_dot_product_attention(Q, K, V)
        print(f"  {d_k:>4}  {scores.std():>20.1f}  {unscaled.max(axis=-1).mean():>15.2f}  "
              f"{scaled.max(axis=-1).mean():>20.2f}")
    print(f"\nThe spread grows like sqrt(d_k): sqrt(4) = 2, sqrt(64) = 8, sqrt(512) = {np.sqrt(512):.1f}")

    print("\nOne row, d_k = 512, the same scores two ways:")
    print("  unscaled:          " + "  ".join(f"{w:.2f}" for w in unscaled[0, 0]))
    print("  divided by 22.6:   " + "  ".join(f"{w:.2f}" for w in scaled[0, 0]))
    print("Unscaled, one key takes almost everything (nearly one-hot). Scaled, the weights stay spread out.")


def demo_heads() -> None:
    rng = np.random.default_rng(SEED)
    d_model, num_heads = 8, 2
    X = rng.normal(size=(4, d_model))
    params = random_block_params(d_model, 4 * d_model, rng)
    Wq, Wk, Wv, Wo = params["Wq"], params["Wk"], params["Wv"], params["Wo"]

    print(f"X                         {X.shape}  4 tokens, model width {d_model}")
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    print(f"Q, K, V = X @ W           {Q.shape}  each")
    Qh, Kh, Vh = split_heads(Q, num_heads), split_heads(K, num_heads), split_heads(V, num_heads)
    print(f"split into {num_heads} heads        {Qh.shape}  heads, tokens, numbers per head")
    heads, weights = scaled_dot_product_attention(Qh, Kh, Vh)
    print(f"attention weights         {weights.shape}  one 4 x 4 grid per head")
    print(f"head outputs              {heads.shape}")
    joined = combine_heads(heads)
    print(f"concatenate the heads     {joined.shape}")
    output = joined @ Wo
    print(f"project with Wo           {output.shape}  back to the input width\n")

    lib_output, _ = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads)
    print(f"multi_head_attention() gives the same answer: {np.allclose(lib_output, output)}")
    i = TOKENS.index("dosa")
    for h in range(num_heads):
        j = int(np.argmax(weights[h, i]))
        print(f"  head {h + 1}: {TOKENS[i]!r} looks most at {TOKENS[j]!r} ({weights[h, i, j]:.0%})")
    print("(Random weights, so these choices mean nothing yet. Training gives each head its own job.)\n")

    block_out = encoder_block(X, params, num_heads)
    print(f"One encoder block: in {X.shape} -> attention, add & norm, feed-forward {params['W1'].shape}, "
          f"add & norm -> out {block_out.shape}")
    print(f"After layer norm, each token's numbers have mean {abs(block_out.mean(axis=-1)).max():.2f} "
          f"and spread {block_out.std(axis=-1).mean():.2f}")
    print("DistilBERT does the same with 12 heads x 64 numbers = 768, and stacks 6 such blocks (step 8).")


DEMOS = {"scratch": demo_scratch, "scaling": demo_scaling, "heads": demo_heads}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Episode 02, steps 9-11: attention in NumPy")
    parser.add_argument("demo", nargs="?", default="scratch", choices=list(DEMOS))
    args = parser.parse_args(argv)
    np.set_printoptions(precision=2, suppress=True)
    DEMOS[args.demo]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
