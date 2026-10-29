"""
tests/test_sampling.py - temperature, top-k, top-p and sampling on fixed logits with fixed seeds (steps 7 and 8).
"""

import numpy as np
import pytest

from attention import softmax
from greedy_loop import greedy_pick
from sampling import apply_temperature, kept, make_picker, sample, top_k_filter, top_p_filter

# Five tokens whose probabilities you can check by hand: 50%, 25%, 15%, 7%, 3%
PROBS = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
LOGITS = np.log(PROBS)


def test_softmax_of_our_logits_gives_the_probabilities_back():
    assert np.allclose(softmax(LOGITS), PROBS)


def test_temperature_one_changes_nothing():
    assert np.allclose(softmax(apply_temperature(LOGITS, 1.0)), PROBS)


def test_low_temperature_sharpens_and_high_temperature_flattens():
    top = {t: softmax(apply_temperature(LOGITS, t))[0] for t in [0.2, 1.0, 1.5]}
    assert top[0.2] > 0.95                      # almost always the best token
    assert top[0.2] > top[1.0] > top[1.5]
    flat = softmax(apply_temperature(LOGITS, 100.0))
    assert np.allclose(flat, 0.2, atol=0.01)    # very hot: nearly a fair five-way draw


@pytest.mark.parametrize("bad", [0, -1.0])
def test_temperature_must_be_above_zero(bad):
    with pytest.raises(ValueError, match="greedy"):
        apply_temperature(LOGITS, bad)


def test_top_k_keeps_the_k_best():
    cut = top_k_filter(LOGITS, 2)
    assert kept(cut) == 2
    assert np.isfinite(cut[:2]).all() and np.isneginf(cut[2:]).all()
    assert np.allclose(softmax(cut)[:2], [0.5 / 0.75, 0.25 / 0.75])   # the two share 100% again


def test_top_k_bigger_than_the_vocabulary_keeps_everything():
    assert kept(top_k_filter(LOGITS, 50)) == 5
    with pytest.raises(ValueError):
        top_k_filter(LOGITS, 0)


@pytest.mark.parametrize("p, expected", [(0.4, 1), (0.7, 2), (0.8, 3), (0.95, 4), (1.0, 5)])
def test_top_p_keeps_the_smallest_group_that_reaches_p(p, expected):
    # running sums: 0.50, 0.75, 0.90, 0.97, 1.00
    assert kept(top_p_filter(LOGITS, p)) == expected


def test_top_p_always_keeps_the_best_token():
    cut = top_p_filter(LOGITS, 0.01)
    assert kept(cut) == 1 and np.isfinite(cut[0])


@pytest.mark.parametrize("seed", range(5))
def test_filters_keep_the_same_tokens_as_transformers(seed):
    import torch
    from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

    logits = np.random.default_rng(seed).normal(scale=3.0, size=2000)
    scores = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
    ids = torch.zeros((1, 1), dtype=torch.long)
    for k in [1, 5, 50]:
        theirs = TopKLogitsWarper(top_k=k)(ids, scores.clone())[0].isfinite().numpy()
        assert (theirs == np.isfinite(top_k_filter(logits, k))).all()
    for p in [0.5, 0.9, 0.95]:
        theirs = TopPLogitsWarper(top_p=p)(ids, scores.clone())[0].isfinite().numpy()
        assert (theirs == np.isfinite(top_p_filter(logits, p))).all()


def test_sampling_is_repeatable_with_a_seed():
    def twenty_draws(seed):
        rng = np.random.default_rng(seed)
        return [sample(LOGITS, rng) for _ in range(20)]

    assert twenty_draws(42) == twenty_draws(42)       # same seed, same "random" tokens
    assert twenty_draws(42) != twenty_draws(43)       # another seed, other tokens


def test_samples_follow_the_probabilities():
    rng = np.random.default_rng(0)
    draws = np.array([sample(LOGITS, rng) for _ in range(20_000)])
    share = np.bincount(draws, minlength=5) / len(draws)
    assert np.allclose(share, PROBS, atol=0.015)


def test_samples_never_leave_the_kept_tokens():
    pick = make_picker(temperature=1.5, top_k=2, rng=np.random.default_rng(1))
    assert {pick(LOGITS) for _ in range(500)} == {0, 1}
    pick = make_picker(top_p=0.8, rng=np.random.default_rng(1))
    assert {pick(LOGITS) for _ in range(500)} == {0, 1, 2}


def test_top_k_of_one_is_greedy():
    pick = make_picker(top_k=1, rng=np.random.default_rng(3))
    assert all(pick(LOGITS) == greedy_pick(LOGITS) == 0 for _ in range(50))


def test_temperature_comes_before_top_p():
    # At t=0.2 the best token alone holds more than 95%, so top-p 0.9 keeps only that one.
    pick = make_picker(temperature=0.2, top_p=0.9, rng=np.random.default_rng(7))
    assert {pick(LOGITS) for _ in range(200)} == {0}
