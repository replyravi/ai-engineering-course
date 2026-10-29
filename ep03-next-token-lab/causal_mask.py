"""
causal_mask.py - the one small mask that turns Episode 02's attention into a decoder's (step 1).

Episode 02's attention lets every word look at every other word, also the words AFTER it.
A decoder writes left to right, so word i may only look at words 0, 1, ..., i. We hide the
future with a mask: those scores become minus infinity before softmax, so their weights are
exactly 0, and each row still sums to 1.

    causal_mask(n)      -> (n, n) True on and below the diagonal ("may look"), False above it

It plugs into the mask= argument that attention.py (copied from Episode 02) already has:

    scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=causal_mask(n))

Run:  python causal_mask.py
"""

import numpy as np

from attention import multi_head_attention, random_block_params, scaled_dot_product_attention, softmax

SEED = 42
WORDS = ["chai", "tastes", "best", "when", "it"]


def causal_mask(n: int) -> np.ndarray:
    """(n, n) of True/False. Row i is True for keys 0..i (the past and itself), False for the future."""
    return np.tril(np.ones((n, n), dtype=bool))   # tril = keep the lower triangle


def hide_future(scores: np.ndarray) -> np.ndarray:
    """The scores with every future position replaced by minus infinity (what the mask does inside attention)."""
    n = scores.shape[-1]
    return np.where(causal_mask(n), scores, -np.inf)


def is_lower_triangular(weights: np.ndarray) -> bool:
    """True if every weight above the diagonal is exactly 0.0 (in every head, if there are several)."""
    n = weights.shape[-1]
    above = ~causal_mask(n)
    return bool((weights[..., above] == 0.0).all())


def show(name: str, matrix: np.ndarray, labels: list, number_format: str = "{:>8.2f}") -> None:
    """Print a small grid with the words on both sides."""
    print(name)
    print("         " + "".join(f"{w:>8}" for w in labels))
    for label, row in zip(labels, matrix):
        print(f"  {label:<7}" + "".join(number_format.format(x) for x in row))


def main() -> int:
    rng = np.random.default_rng(SEED)
    n = len(WORDS)
    X = rng.normal(size=(n, 8))                     # 5 words x 8 numbers (pretend embeddings + positions)
    Wq, Wk, Wv = (rng.normal(size=(8, 4)) for _ in range(3))   # learned in a real model, random here
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    print(f"Words: {WORDS}   X {X.shape}   Q, K, V {Q.shape} each\n")

    scores = Q @ K.T / np.sqrt(K.shape[-1])         # Episode 02: every word against every word
    show("scores = Q @ K.T / sqrt(4)", scores, WORDS)

    mask = causal_mask(n)
    print()
    show("causal mask (1 = may look, 0 = the future, hidden)", mask.astype(int), WORDS, "{:>8}")

    masked = hide_future(scores)
    print()
    show("scores with the future set to -inf", masked, WORDS)

    weights = softmax(masked)                       # exp(-inf) = 0, so the future gets weight 0
    print()
    show("weights = softmax(masked scores)", weights, WORDS)
    print(f"row sums: {weights.sum(axis=1).round(6).tolist()}")
    print(f"every weight above the diagonal is exactly 0.0: {is_lower_triangular(weights)}")
    print(f"row '{WORDS[0]}' can only look at itself: weight {weights[0, 0]:.2f}")

    output, lib_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    print(f"\nEpisode 02's scaled_dot_product_attention(Q, K, V, mask=causal_mask(5)) "
          f"gives the same weights: {np.allclose(lib_weights, weights)}")

    # No peeking: change the LAST word, then check whether the first four outputs move.
    X_changed = X.copy()
    X_changed[-1] = rng.normal(size=8)
    Q2, K2, V2 = X_changed @ Wq, X_changed @ Wk, X_changed @ Wv
    open_before, _ = scaled_dot_product_attention(Q, K, V)
    open_after, _ = scaled_dot_product_attention(Q2, K2, V2)
    masked_after, _ = scaled_dot_product_attention(Q2, K2, V2, mask=causal_mask(n))
    print(f"\nChange the vector of the last word '{WORDS[-1]}'. Do the outputs of the first {n - 1} words stay the same?")
    print(f"  without the mask: {np.allclose(open_before[:-1], open_after[:-1])}")
    print(f"  with the causal mask: {np.allclose(output[:-1], masked_after[:-1])}")
    print(f"So in training all {n} positions can guess their next word at the same time, and none can peek at its answer.")

    params = random_block_params(8, 32, rng)
    _, head_weights = multi_head_attention(X, params["Wq"], params["Wk"], params["Wv"], params["Wo"],
                                           num_heads=2, mask=causal_mask(n))
    print(f"\nTwo heads with the same mask: weights {head_weights.shape}, "
          f"every head lower-triangular: {is_lower_triangular(head_weights)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
