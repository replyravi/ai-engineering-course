"""
tests/test_ranking.py - the encoder steps (3 and 4): cosine ranking and the parameter counter.

Everything here runs offline with small hand-made numbers or fake embeddings,
and with the sample reviews (zoo_fixtures.py), not your own data/reviews.txt.
"""

import math

import numpy as np
import pytest
import torch

import encoder
from common import count_parameters, in_millions, load_reviews
from testing.fakes import fake_embeddings
from zoo_fixtures import SAMPLE_REVIEWS_FILE, sample_reviews


def test_cosine_similarity_basics():
    assert encoder.cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)     # same direction, any length
    assert encoder.cosine_similarity([1, 0], [0, 3]) == pytest.approx(0.0)     # unrelated
    assert encoder.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)   # opposite
    assert encoder.cosine_similarity([1, 1], [1, 0]) == pytest.approx(1 / math.sqrt(2))


def test_rank_puts_the_closest_text_first():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    texts = ["first", "second", "third"]
    result = encoder.rank(np.array([1.0, 0.0]), vectors, texts, top=2)
    assert [number for _, number, _ in result] == [1, 3]          # review numbers start at 1
    assert [text for _, _, text in result] == ["first", "third"]
    assert result[0][0] >= result[1][0]                           # best score first


def test_count_parameters_on_a_tiny_model():
    assert count_parameters(torch.nn.Linear(3, 2)) == 8           # 3 x 2 weights + 2 biases
    tiny = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    assert count_parameters(tiny) == 15 + 4                       # ReLU has no parameters


def test_in_millions():
    assert in_millions(22_713_216) == "22.7 million"
    assert in_millions(494_032_768) == "494.0 million"


def test_the_sample_reviews_file_has_ten_reviews():
    reviews = load_reviews(SAMPLE_REVIEWS_FILE)
    assert len(reviews) == 10
    assert all(len(review) > 10 for review in reviews)


def test_load_reviews_skips_empty_lines(tmp_path):
    path = tmp_path / "reviews.txt"
    path.write_text("Good tea.\n\n   \nCold coffee.\n", encoding="utf-8")
    assert load_reviews(path) == ["Good tea.", "Cold coffee."]


def test_shared_words_ignores_filler_words():
    assert encoder.shared_words("the snacks were stale", "The samosa was fried yesterday") == set()
    assert encoder.shared_words("cold coffee", "My coffee was cold") == {"cold", "coffee"}


def test_encoder_script_runs_offline_with_fake_embeddings(capsys, monkeypatch):
    monkeypatch.setattr(encoder, "load_reviews", sample_reviews)   # the ten sample reviews
    assert encoder.main([], embeddings=fake_embeddings()) == 0
    out = capsys.readouterr().out
    assert "10 vectors, 384 numbers each" in out
    assert "Parameters: not published" in out                    # a fake (or cloud) model has no local weights
    assert 'Closest reviews to "the snacks were stale"' in out
