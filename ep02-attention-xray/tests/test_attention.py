"""
tests/test_attention.py - the NumPy attention maths (steps 9 to 11). Pure functions, fixed inputs.
"""

import numpy as np
import pytest

import positions
from attention import (combine_heads, encoder_block, layer_norm, main, multi_head_attention,
                       positional_encoding, random_block_params, scaled_dot_product_attention,
                       softmax, split_heads)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_softmax_rows_sum_to_one_even_for_huge_scores():
    weights = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]))
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert np.allclose(weights[0], weights[1])        # shifting all scores changes nothing
    assert not np.isnan(weights).any()                # no overflow


def test_small_example_by_hand():
    Q = K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0]])
    output, weights = scaled_dot_product_attention(Q, K, V)
    # scores = [[1, 0], [0, 1]] / sqrt(2); softmax([0.7071, 0]) = [0.6698, 0.3302]
    assert np.allclose(weights[0], [0.66976155, 0.33023845])
    assert np.allclose(output[0], 0.66976155 * V[0] + 0.33023845 * V[1])


def test_shapes_and_rows_sum_to_one(rng):
    Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(4, 8)), rng.normal(size=(4, 3))
    output, weights = scaled_dot_product_attention(Q, K, V)
    assert weights.shape == (4, 4)
    assert output.shape == (4, 3)                     # one row per query, as wide as V
    assert np.allclose(weights.sum(axis=-1), 1.0)
    assert (weights >= 0).all()


def test_scaling_keeps_softmax_from_going_one_hot(rng):
    Q, K, V = (rng.normal(size=(200, 6, 512)) for _ in range(3))
    _, unscaled = scaled_dot_product_attention(Q, K, V, scale=False)
    _, scaled = scaled_dot_product_attention(Q, K, V)
    assert unscaled.max(axis=-1).mean() > 0.9         # nearly one-hot
    assert scaled.max(axis=-1).mean() < 0.6           # spread out


def test_padding_mask_gives_exactly_zero_weight(rng):
    Q, K, V = (rng.normal(size=(4, 8)) for _ in range(3))
    mask = np.array([1, 1, 1, 0])                     # like attention_mask: the last token is [PAD]
    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert (weights[:, 3] == 0.0).all()               # exactly zero, not just small
    assert np.allclose(weights.sum(axis=-1), 1.0)
    # the padded token's value cannot leak in: change it and nothing changes
    V2 = V.copy()
    V2[3] = 999.0
    assert np.allclose(scaled_dot_product_attention(Q, K, V2, mask=mask)[0], output)


def test_a_query_must_see_at_least_one_key(rng):
    Q, K, V = (rng.normal(size=(3, 4)) for _ in range(3))
    with pytest.raises(ValueError, match="at least|one key"):
        scaled_dot_product_attention(Q, K, V, mask=np.zeros((3, 3), dtype=bool))


def test_batch_axis_gives_the_same_answer_per_sentence(rng):
    Q, K, V = (rng.normal(size=(2, 5, 4)) for _ in range(3))
    output, weights = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (2, 5, 4) and weights.shape == (2, 5, 5)
    for b in range(2):
        alone, _ = scaled_dot_product_attention(Q[b], K[b], V[b])
        assert np.allclose(output[b], alone)


def test_attention_alone_ignores_word_order(rng):
    X = rng.normal(size=(5, 8))
    Wq, Wk, Wv = (rng.normal(size=(8, 8)) for _ in range(3))
    order = [3, 0, 4, 1, 2]
    out, _ = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)
    shuffled = X[order]
    out_shuffled, _ = scaled_dot_product_attention(shuffled @ Wq, shuffled @ Wk, shuffled @ Wv)
    assert np.allclose(out_shuffled, out[order])      # same outputs, just in the new order


def test_split_and_combine_heads_round_trip(rng):
    x = rng.normal(size=(3, 4, 8))                    # batch 3, 4 tokens, width 8
    heads = split_heads(x, 2)
    assert heads.shape == (3, 2, 4, 4)
    assert np.allclose(heads[:, 1], x[..., 4:])       # head 2 gets numbers 4-7 of every token
    assert np.allclose(combine_heads(heads), x)
    with pytest.raises(ValueError):
        split_heads(x, 3)


def test_multi_head_attention_keeps_the_width(rng):
    X = rng.normal(size=(4, 8))
    p = random_block_params(8, 32, rng)
    output, weights = multi_head_attention(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads=2)
    assert output.shape == X.shape
    assert weights.shape == (2, 4, 4)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_one_head_equals_plain_attention(rng):
    X = rng.normal(size=(4, 8))
    p = random_block_params(8, 32, rng)
    output, _ = multi_head_attention(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads=1)
    plain, _ = scaled_dot_product_attention(X @ p["Wq"], X @ p["Wk"], X @ p["Wv"])
    assert np.allclose(output, plain @ p["Wo"])


def test_multi_head_mask_applies_to_every_head(rng):
    X = rng.normal(size=(2, 4, 8))                    # a batch of 2 sentences
    p = random_block_params(8, 32, rng)
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])[:, np.newaxis, :]   # (batch, 1, keys): sentence 2 is padded
    _, weights = multi_head_attention(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads=2, mask=mask)
    assert weights.shape == (2, 2, 4, 4)
    assert (weights[1, :, :, 2:] == 0.0).all()        # both heads ignore the padding
    assert (weights[0] > 0).all()
    _, one_sentence = multi_head_attention(X[1], p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads=2,
                                           mask=np.array([1, 1, 0, 0]))   # a plain row mask works too
    assert np.allclose(one_sentence, weights[1])


def test_encoder_block_keeps_shape_and_normalises(rng):
    X = rng.normal(size=(4, 8))
    out = encoder_block(X, random_block_params(8, 32, rng), num_heads=2)
    assert out.shape == (4, 8)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-7)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)
    assert np.allclose(layer_norm(np.array([[1.0, 2.0, 3.0]])), [[-1.2247, 0.0, 1.2247]], atol=1e-3)


def test_positional_encoding_is_available_from_attention():
    assert positional_encoding is positions.positional_encoding


@pytest.mark.parametrize("demo, expected", [
    ("scratch", "row sums: [1.0, 1.0, 1.0, 1.0]"),
    ("scaling", "nearly one-hot"),
    ("heads", "back to the input width"),
])
def test_demos_run(capsys, demo, expected):
    assert main([demo]) == 0
    assert expected in capsys.readouterr().out
