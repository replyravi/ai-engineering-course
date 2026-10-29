"""
tests/test_greedy.py - the generation loop and its stop rules, on a scripted token stream (steps 5 and 6).

No model needed: a tiny fake "model" is completely sure of the next token at every step.
"""

import numpy as np

from greedy_loop import END_OF_TEXT, MAX_TOKENS, find_loop, generate, greedy_pick
from sampling import make_picker

EOS = 0          # our pretend end-of-text token
PROMPT = [7, 8]


def scripted(tokens, vocab: int = 10):
    """A fake next_logits(): at step n it gives token tokens[n] a high score. Also records what it read."""
    reads = []

    def next_logits(ids):
        reads.append(list(ids))
        logits = np.full(vocab, -5.0)
        logits[tokens[len(reads) - 1]] = 5.0
        return logits

    return next_logits, reads


def test_greedy_pick_takes_the_highest_score():
    assert greedy_pick(np.array([0.1, 3.0, -2.0, 2.9])) == 1


def test_stops_at_the_end_of_text_token():
    next_logits, reads = scripted([3, 4, EOS, 9, 9])
    new_ids, rule = generate(next_logits, PROMPT, max_new_tokens=10, eos_id=EOS)
    assert rule == END_OF_TEXT
    assert new_ids == [3, 4]                     # the end-of-text token itself is not part of the text
    assert len(reads) == 3                       # it stopped right away, no extra model calls


def test_stops_at_max_tokens():
    next_logits, _ = scripted([3, 4, 5, 6, 1, 2])
    new_ids, rule = generate(next_logits, PROMPT, max_new_tokens=4, eos_id=EOS)
    assert rule == MAX_TOKENS
    assert new_ids == [3, 4, 5, 6]


def test_every_step_reads_everything_written_so_far():
    next_logits, reads = scripted([3, 4, 5])
    generate(next_logits, PROMPT, max_new_tokens=3, eos_id=EOS)
    assert reads == [[7, 8], [7, 8, 3], [7, 8, 3, 4]]   # no KV cache: the whole text, every time


def test_without_an_end_token_only_the_limit_stops_it():
    next_logits, _ = scripted([EOS, EOS, EOS])
    new_ids, rule = generate(next_logits, PROMPT, max_new_tokens=3, eos_id=None)
    assert rule == MAX_TOKENS and new_ids == [EOS, EOS, EOS]


def test_on_step_sees_every_pick():
    seen = []
    next_logits, _ = scripted([3, 4, EOS])
    generate(next_logits, PROMPT, max_new_tokens=5, eos_id=EOS,
             on_step=lambda step, token, logits: seen.append((step, token)))
    assert seen == [(1, 3), (2, 4), (3, EOS)]


def test_a_sampling_rule_plugs_into_the_same_loop():
    next_logits, _ = scripted([3, 4, 5, 6])
    picker = make_picker(top_k=1, rng=np.random.default_rng(42))   # top-k=1 always takes the best token
    new_ids, _ = generate(next_logits, PROMPT, max_new_tokens=4, eos_id=EOS, pick=picker)
    assert new_ids == [3, 4, 5, 6]


def test_find_loop_spots_a_repeating_ending():
    assert find_loop([1, 2] + [5, 6, 7, 8] * 3) == (4, 3)
    assert find_loop([9] + [5, 6, 7, 8] * 2 + [5, 6]) == (4, 2)    # stopped in the middle of a round
    assert find_loop([1, 2, 3, 4, 5, 6, 7, 8, 9]) is None
