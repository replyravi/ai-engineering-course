"""
tests/test_local_models.py - the three REAL local models, offline.

Each test is skipped (an "s" in the pytest output) until its model is downloaded:
run encoder.py, decoder_local.py and multimodal_clip.py once with internet, then test again.

They use the SAMPLE reviews and pictures (see zoo_fixtures.py), so they still pass after you
put your own reviews in data/reviews.txt and your own photos in photos/.
"""

import numpy as np
import pytest

from common import count_parameters, find_photos
from zoo_fixtures import draw_sample_photos, needs_model, sample_reviews


@needs_model("sentence-transformers/all-MiniLM-L6-v2")
def test_real_encoder_finds_the_stale_snack_review():
    import encoder
    from llm import get_embeddings

    embeddings = get_embeddings()
    reviews = sample_reviews()
    vectors = encoder.embed(reviews, embeddings)
    assert vectors.shape == (10, 384)
    assert encoder.parameters_of(embeddings) == 22_713_216
    best = encoder.rank(np.array(embeddings.embed_query("the snacks were stale")), vectors, reviews)[0]
    assert best[1] == 3 and "samosa" in best[2]                   # no shared words, same meaning


@needs_model("Qwen/Qwen2.5-0.5B-Instruct")
def test_real_local_decoder_writes_and_its_weights_stay_the_same():
    import decoder_local
    from common import OWNER_PROMPT, OWNER_SYSTEM

    chat = decoder_local.load_decoder()
    assert count_parameters(chat.model) == 494_032_768
    before = decoder_local.weights_fingerprint(chat.model)
    reply = decoder_local.generate(chat, OWNER_SYSTEM, OWNER_PROMPT, max_new_tokens=20)
    assert isinstance(reply, str) and reply.strip()
    assert decoder_local.weights_fingerprint(chat.model) == before   # tokens flow, weights stay


@needs_model("openai/clip-vit-base-patch32")
def test_real_clip_picks_the_right_label_for_each_sample_photo(tmp_path):
    import multimodal_clip as mc

    clip = mc.load_clip()
    assert count_parameters(clip.model) == 151_277_313
    photos = [p for p in find_photos(draw_sample_photos(tmp_path)) if mc.expected_label(p, mc.LABELS)]
    assert len(photos) == 4
    for path in photos:
        scores = mc.score_photo(clip, mc.open_photo(path), mc.LABELS)
        assert scores[0][0] == mc.expected_label(path, mc.LABELS)
        assert sum(score for _, score in scores) == pytest.approx(1.0, abs=1e-3)

    captions = [mc.TEMPLATE.format(mc.expected_label(p, mc.LABELS)) for p in photos]
    grid = mc.similarity_grid(clip, [mc.open_photo(p) for p in photos], captions)
    assert grid.argmax(dim=1).tolist() == list(range(len(photos)))   # each photo matches its own caption best
