"""
tests/test_local_models.py - checks on the real downloaded models (steps 2-6, 8-11).

They run offline once the models are downloaded, and SKIP with the fix if they are not.
"""

import numpy as np
import pytest

import cross_attention
import kv_cache_timing
import next_token
from attention import softmax
from decoder import next_token_logits, read_prompts
from greedy_loop import END_OF_TEXT, MAX_TOKENS, generate, library_greedy
from model_fixtures import classifier, decoder, t5  # noqa: F401
from sampling import top_k_filter, top_p_filter
from three_families import FAMILIES, classify, family_table, summarise

PROMPTS = read_prompts()
COLD_OPEN = PROMPTS["cold_open"][0]


def test_vocabulary_context_and_end_token(decoder):
    tokenizer, model = decoder
    assert len(tokenizer) == model.config.vocab_size == 49152
    assert model.config.max_position_embeddings == 8192
    assert tokenizer.eos_token == "<|endoftext|>" and tokenizer.eos_token_id == 0
    assert str(model.dtype) == "torch.float32"


def test_logits_have_one_score_per_vocabulary_entry_per_position(decoder):
    tokenizer, model = decoder
    ids, logits = next_token.forward(tokenizer, model, COLD_OPEN)
    assert len(ids) == 5
    assert logits.shape == (1, 5, 49152)


def test_top_next_token_for_the_cold_open(decoder):
    tokenizer, model = decoder
    probs = softmax(next_token_logits(model, tokenizer(COLD_OPEN)["input_ids"]))
    (best, p), *_ = next_token.top_tokens(probs, 10)
    assert tokenizer.decode([best]) == " it"
    assert p == pytest.approx(0.451, abs=0.01)


def test_the_real_decoder_cannot_see_the_future(decoder):
    import torch

    tokenizer, model = decoder
    ids = tokenizer(COLD_OPEN)["input_ids"]
    with torch.inference_mode():
        a = model(torch.tensor([ids])).logits[0]
        b = model(torch.tensor([ids[:-1] + [tokenizer(" if")["input_ids"][0]]])).logits[0]
    assert torch.allclose(a[:-1], b[:-1], atol=1e-4)     # earlier positions do not move
    assert not torch.allclose(a[-1], b[-1])


def test_hand_written_loop_matches_model_generate(decoder):
    tokenizer, model = decoder
    prompt_ids = tokenizer(COLD_OPEN)["input_ids"]
    new_ids, rule = generate(lambda ids: next_token_logits(model, ids), prompt_ids, 12, eos_id=tokenizer.eos_token_id)
    assert rule == MAX_TOKENS
    assert new_ids == library_greedy(tokenizer, model, prompt_ids, 12)
    assert tokenizer.decode(new_ids).startswith(" it is cooked in a sauce")


def test_the_model_can_write_end_of_text_by_itself(decoder):
    tokenizer, model = decoder
    prompt_ids = tokenizer(PROMPTS["ends_itself"][0])["input_ids"]
    new_ids, rule = generate(lambda ids: next_token_logits(model, ids), prompt_ids, 40, eos_id=tokenizer.eos_token_id)
    assert rule == END_OF_TEXT
    assert len(new_ids) < 40


def test_kv_cache_gives_the_same_tokens(decoder):
    tokenizer, model = decoder
    prompt_ids = tokenizer(COLD_OPEN)["input_ids"]
    with_cache, cache = kv_cache_timing.write_with_cache(model, prompt_ids, 12)
    assert with_cache == kv_cache_timing.write_without_cache(model, prompt_ids, 12)
    assert len(cache.layers) == 30
    assert tuple(cache.layers[0].keys.shape) == (1, 3, 5 + 11, 64)   # prompt + 11 fed tokens
    assert kv_cache_timing.positions_read(5, 100, cache=False) == 5450
    assert kv_cache_timing.positions_read(5, 100, cache=True) == 104


def test_our_cuts_match_transformers_on_real_logits(decoder):
    import torch
    from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

    tokenizer, model = decoder
    prompt_ids = tokenizer(COLD_OPEN)["input_ids"]
    logits = next_token_logits(model, prompt_ids)
    scores = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
    ids = torch.tensor([prompt_ids])
    theirs_k = TopKLogitsWarper(top_k=5)(ids, scores.clone())[0].isfinite().numpy()
    theirs_p = TopPLogitsWarper(top_p=0.9)(ids, scores.clone())[0].isfinite().numpy()
    assert (theirs_k == np.isfinite(top_k_filter(logits, 5))).all()
    assert (theirs_p == np.isfinite(top_p_filter(logits, 0.9))).all()
    assert theirs_p.sum() == 167


@pytest.mark.parametrize("step", ["load", "logits", "top10"])
def test_next_token_steps_run(decoder, capsys, step):
    tokenizer, model = decoder
    assert next_token.main([step], tokenizer=tokenizer, model=model) == 0
    assert "Chai tastes best when" in capsys.readouterr().out


def test_kv_cache_script_runs(decoder, capsys):
    tokenizer, model = decoder
    assert kv_cache_timing.main(["--tokens", "8"], tokenizer=tokenizer, model=model) == 0
    assert "Same 8 tokens both ways: True" in capsys.readouterr().out


def test_t5_translates_and_returns_cross_attention(t5):
    tokenizer, model = t5
    input_ids, output_ids = cross_attention.translate(tokenizer, model, PROMPTS["translate"][0])
    assert tokenizer.decode(output_ids, skip_special_tokens=True) == "Die Kantine ist am Sonntag geschlossen."
    cross, decoder_self = cross_attention.attention_maps(model, input_ids, output_ids)
    assert cross.shape == (6, 8, len(output_ids) - 1, len(input_ids))
    assert np.allclose(cross.sum(axis=-1), 1.0, atol=1e-5)
    above = ~np.tril(np.ones(decoder_self.shape[-2:], dtype=bool))
    assert (decoder_self[..., above] < 1e-6).all()                    # T5's decoder is causal too


def test_cross_attention_script_saves_a_picture(t5, tmp_path, capsys):
    picture = tmp_path / "cross.png"
    assert cross_attention.main([], picture=picture) == 0
    assert picture.stat().st_size > 10_000
    assert "geschlossen   -> ▁closed" in capsys.readouterr().out


def test_fast_attention_without_weights_gives_a_clear_error():
    class FastModel:                       # behaves like T5 loaded with the default attn_implementation="sdpa"
        def __call__(self, **inputs):
            return type("Output", (), {"cross_attentions": (), "decoder_attentions": ()})()

    with pytest.raises(RuntimeError, match="eager"):
        cross_attention.attention_maps(FastModel(), [1, 2], [0, 3, 1])


def test_t5_summary_is_generated_text(t5):
    tokenizer, model = t5
    summary = summarise(tokenizer, model, PROMPTS["notice"][0])
    assert 0 < len(summary.split()) < len(PROMPTS["notice"][0].split())


def test_encoder_gives_one_of_two_labels(classifier):
    labels = classify(classifier, PROMPTS["feedback"])
    assert [label for label, _ in labels] == ["NEGATIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]


def test_only_the_encoder_decoder_has_cross_attention():
    assert [(family, cross) for family, _, _, cross, _ in FAMILIES] == [
        ("Encoder-only", "no"), ("Encoder-decoder", "yes"), ("Decoder-only", "no")]
    table = family_table()
    assert all(family in table for family, *_ in FAMILIES)
