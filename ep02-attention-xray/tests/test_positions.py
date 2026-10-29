"""
tests/test_positions.py - the sine/cosine position table (step 7). Pure NumPy, no model.
"""

import numpy as np
import pytest

from positions import main, positional_encoding, same_bag


def test_shape_and_range():
    table = positional_encoding(50, 64)
    assert table.shape == (50, 64)
    assert table.min() >= -1.0 and table.max() <= 1.0


def test_position_zero_is_sin0_cos0():
    assert np.allclose(positional_encoding(1, 8)[0], [0, 1, 0, 1, 0, 1, 0, 1])


def test_one_cell_matches_the_formula():
    table = positional_encoding(10, 64)
    angle = 3 / 10000 ** (4 / 64)                     # position 3, pair starting at column 4
    assert table[3, 4] == pytest.approx(np.sin(angle))
    assert table[3, 5] == pytest.approx(np.cos(angle))


def test_every_position_is_unique():
    table = positional_encoding(512, 768)
    assert len(np.unique(table.round(6), axis=0)) == 512


def test_odd_width_is_refused():
    with pytest.raises(ValueError):
        positional_encoding(5, 7)


def test_positions_turn_a_bag_into_an_order():
    rng = np.random.default_rng(1)
    words = rng.normal(size=(3, 16))
    reordered = words[[2, 1, 0]]
    table = positional_encoding(3, 16)
    assert same_bag(words, reordered)                 # without positions: the same bag
    assert not same_bag(words + table, reordered + table)


def test_main_saves_the_picture(tmp_path, capsys):
    picture = tmp_path / "positions.png"
    assert main(picture=picture) == 0
    assert picture.stat().st_size > 10_000
    out = capsys.readouterr().out
    assert "all rows unique: True" in out
    assert "words + positions: False" in out
