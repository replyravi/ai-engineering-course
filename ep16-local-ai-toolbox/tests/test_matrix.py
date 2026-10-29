"""
tests/test_matrix.py - the similarity matrix: its maths, its shape and how it is printed (step 8).

The first tests use small hand-made vectors, so they need no model at all.
The last one uses the real all-MiniLM-L6-v2 (skipped until it is downloaded).
"""

import numpy as np
import pytest

from common import EMBEDDINGS_MODEL
from lab_fixtures import FIXED_NOTES, needs_model
from skills.search import format_matrix, least_similar_pair, most_similar_pair, similarity_matrix


def unit(*numbers):
    vector = np.array(numbers, dtype=float)
    return vector / np.linalg.norm(vector)


# three arrows in 2D: a and b point almost the same way, c points away from a
VECTORS = np.array([unit(1, 0.1), unit(1, 0.3), unit(-1, 0.2)])
LABELS = ["a", "b", "c"]


def test_matrix_is_square_symmetric_with_ones_on_the_diagonal():
    matrix = similarity_matrix(VECTORS)

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert matrix.min() >= -1.0 and matrix.max() <= 1.0 + 1e-9


def test_cosine_can_be_negative():
    matrix = similarity_matrix(VECTORS)

    assert matrix[0, 2] < 0          # a and c point away from each other


def test_most_similar_pair_skips_the_diagonal():
    i, j, score = most_similar_pair(similarity_matrix(VECTORS))

    assert (i, j) == (0, 1)          # a + b, not a + a
    assert score == pytest.approx(float(VECTORS[0] @ VECTORS[1]))


def test_least_similar_pair_finds_the_negative_cell():
    i, j, score = least_similar_pair(similarity_matrix(VECTORS))

    assert (i, j) == (0, 2) and score < 0


def test_format_matrix_shows_two_decimals_signs_and_labels():
    text = format_matrix(similarity_matrix(VECTORS), LABELS)
    lines = text.splitlines()

    assert len(lines) == 4                               # header + one row per label
    assert lines[0].split() == ["1", "2", "3"]           # numbered columns
    assert lines[1].split()[:3] == ["1.", "a", "1.00"]
    assert "-0.9" in lines[1]                            # the sign is kept
    assert all(len(cell.split(".")[-1]) == 2 for cell in lines[2].split()[2:])   # two decimals


@needs_model(EMBEDDINGS_MODEL)
def test_real_model_gives_384_numbers_per_note():
    from skills.search import encode

    vectors = encode([note.text for note in FIXED_NOTES])

    assert vectors.shape == (len(FIXED_NOTES), 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)   # normalised
    assert similarity_matrix(vectors).shape == (4, 4)
