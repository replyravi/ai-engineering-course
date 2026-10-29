"""
tests/test_real_models.py - checks on the real downloaded models (steps 5, 6, 8, 12 and 13).

They run offline once the models are downloaded, and SKIP with the fix if they are not.
"""

import numpy as np
import pytest

import embeddings_peek
import meaning_map
import real_attention
import two_meanings
from encoder import read_sentences
from model_fixtures import encoder, minilm  # noqa: F401
from testing.fakes import fake_embeddings


def test_token_table_is_vocabulary_by_768(encoder):
    _, model = encoder
    assert tuple(model.embeddings.word_embeddings.weight.shape) == (30522, 768)


def test_bert_style_positions_are_learned_not_sine(encoder):
    _, model = encoder
    learned = model.embeddings.position_embeddings.weight.detach().numpy()
    assert learned.shape == (512, 768)
    assert model.config.sinusoidal_pos_embds is False
    assert not np.allclose(learned, embeddings_peek.positional_encoding(512, 768), atol=1e-3)


def test_real_attention_rows_sum_to_one(encoder):
    tokenizer, model = encoder
    tokens, weights = real_attention.attention_maps(read_sentences()["pronoun"][0], tokenizer, model)
    assert weights.shape == (6, 12, len(tokens), len(tokens))
    assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
    layer, head, weight = real_attention.find_head(weights, tokens, "they", "students")
    assert (layer, head) == (3, 11) and weight > 0.5


def test_fast_attention_without_weights_gives_a_clear_error():
    class FastModel:                       # behaves like a model loaded with attn_implementation="sdpa"
        def __call__(self, **inputs):
            return type("Output", (), {"attentions": ()})()

    class Tokenizer:
        def __call__(self, text, return_tensors=None):
            import torch
            return {"input_ids": torch.tensor([[101, 102]])}

    with pytest.raises(RuntimeError, match="eager"):
        real_attention.attention_maps("hello", Tokenizer(), FastModel())


def test_same_word_same_input_vector_different_context(encoder):
    tokenizer, model = encoder
    river, money, river_2, _ = read_sentences()["two_meanings"]
    a, b, c = (two_meanings.word_vectors(s, "bank", tokenizer, model) for s in (river, money, river_2))
    assert a["id"] == b["id"] == 2924
    assert two_meanings.compare(a, b)["table_row"] == pytest.approx(1.0)
    final = two_meanings.compare(a, b)["layers"][-1]
    assert final < 0.8                                            # the context pulled them apart
    assert two_meanings.compare(a, c)["layers"][-1] > final       # river ~ river beats river ~ money


def test_meaning_map_groups_words(minilm):
    words = [w for group in meaning_map.GROUPS.values() for w in group]
    vectors = meaning_map.embed(words + [meaning_map.QUERY], minilm)
    neighbours = meaning_map.nearest(vectors[-1], vectors[:-1], words, k=3)
    assert {w for w, _ in neighbours} <= set(meaning_map.GROUPS["food"])   # pakora lands among the foods


def test_meaning_map_draws_with_fake_embeddings(tmp_path, capsys):
    picture = tmp_path / "map.png"
    assert meaning_map.main(embeddings=fake_embeddings(), picture=picture) == 0   # offline, no model needed
    assert picture.stat().st_size > 10_000
    assert "Nearest neighbour in the same group" in capsys.readouterr().out
