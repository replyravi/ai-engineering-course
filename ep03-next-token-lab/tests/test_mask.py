"""
tests/test_mask.py - the causal mask on top of Episode 02's NumPy attention (step 1). Fixed inputs, no model.
"""

from pathlib import Path

import numpy as np
import pytest

from attention import multi_head_attention, random_block_params, scaled_dot_product_attention, softmax
from causal_mask import causal_mask, hide_future, is_lower_triangular, main

HERE = Path(__file__).resolve().parents[1]
EPISODE_02 = HERE.parent / "ep02-attention-xray"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_mask_is_the_lower_triangle():
    assert causal_mask(3).tolist() == [[True, False, False],
                                       [True, True, False],
                                       [True, True, True]]


def test_future_gets_exactly_zero_and_rows_still_sum_to_one(rng):
    Q, K, V = (rng.normal(size=(5, 4)) for _ in range(3))
    _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(5))
    assert is_lower_triangular(weights)                          # exactly 0.0 above the diagonal
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert weights[0, 0] == pytest.approx(1.0)                   # the first token can only look at itself


def test_hide_future_puts_minus_infinity_above_the_diagonal(rng):
    scores = rng.normal(size=(4, 4))
    masked = hide_future(scores)
    above = ~causal_mask(4)
    assert np.isneginf(masked[above]).all()
    assert np.allclose(masked[~above], scores[~above])           # the past is untouched
    assert is_lower_triangular(softmax(masked))


def test_changing_a_later_token_never_changes_earlier_outputs(rng):
    X = rng.normal(size=(5, 8))
    Wq, Wk, Wv = (rng.normal(size=(8, 4)) for _ in range(3))
    X2 = X.copy()
    X2[3:] = rng.normal(size=(2, 8))                             # rewrite the last two tokens
    before, _ = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv, mask=causal_mask(5))
    after, _ = scaled_dot_product_attention(X2 @ Wq, X2 @ Wk, X2 @ Wv, mask=causal_mask(5))
    assert np.allclose(before[:3], after[:3])                   # tokens 0-2 cannot see the change
    assert not np.allclose(before[3:], after[3:])


def test_without_the_mask_the_future_leaks_in(rng):
    X = rng.normal(size=(5, 8))
    Wq, Wk, Wv = (rng.normal(size=(8, 4)) for _ in range(3))
    X2 = X.copy()
    X2[-1] = rng.normal(size=8)
    before, _ = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)
    after, _ = scaled_dot_product_attention(X2 @ Wq, X2 @ Wk, X2 @ Wv)
    assert not np.allclose(before[0], after[0])                 # token 0 peeked at the last token


def test_mask_works_for_every_head_and_a_batch(rng):
    X = rng.normal(size=(2, 4, 8))                               # a batch of 2 sentences
    p = random_block_params(8, 32, rng)
    output, weights = multi_head_attention(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads=2, mask=causal_mask(4))
    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)                         # sentence, head, query, key
    assert is_lower_triangular(weights)


def test_causal_and_padding_masks_combine(rng):
    Q, K, V = (rng.normal(size=(4, 4)) for _ in range(3))
    padding = np.array([True, True, True, False])                # the last token is [PAD]
    _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(4) & padding)
    assert (weights[:, 3] == 0.0).all()                          # nobody looks at the padding
    assert is_lower_triangular(weights)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_demo_runs(capsys):
    assert main() == 0
    out = capsys.readouterr().out
    assert "every weight above the diagonal is exactly 0.0: True" in out
    assert "with the causal mask: True" in out
    assert "without the mask: False" in out


@pytest.mark.parametrize("name", ["attention.py", "positions.py"])
def test_copy_still_matches_episode_02(name):
    """attention.py and positions.py are copies of Episode 02's files, plus a short header comment."""
    original = EPISODE_02 / name
    if not original.exists():
        pytest.skip("Episode 02's folder is not next to this one, nothing to compare with")
    lines = (HERE / name).read_text(encoding="utf-8").splitlines(keepends=True)
    start = 0
    while lines[start].startswith("#"):                          # skip our header comment ...
        start += 1
    assert lines[start].strip() == ""                            # ... and the blank line after it
    assert "".join(lines[start + 1:]) == original.read_text(encoding="utf-8"), (
        f"{name} differs from Episode 02's copy. Copy it again and keep the header comment.")
