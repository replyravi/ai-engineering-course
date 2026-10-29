"""
tests/test_search.py - ranking notes by meaning (steps 7 and 9), and reading the notes folder.

rank() is tested with hand-made vectors (no model needed). The real-model test uses fixed
sentences, so it still passes after you rewrite the notes in my_notes/.
"""

import numpy as np

from common import EMBEDDINGS_MODEL, load_notes
from lab_fixtures import FIXED_NOTES, needs_model
from skills.search import rank
from skills.summarise import split_sentences


def test_rank_puts_the_closest_vector_first():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    query = np.array([0.0, 1.0])

    ranked = rank(query, vectors, ["east", "north", "north-east"])

    assert [name for name, _ in ranked] == ["north", "north-east", "east"]
    assert [round(score, 2) for _, score in ranked] == [1.0, 0.8, 0.0]


def test_load_notes_reads_txt_files_sorted_and_skips_empty_ones(tmp_path):
    (tmp_path / "b_second.txt").write_text("Second note,\n  over two lines.", encoding="utf-8")
    (tmp_path / "a_first.txt").write_text("First note.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "picture.jpg").write_bytes(b"not a note")

    notes = load_notes(tmp_path)

    assert [note.name for note in notes] == ["a_first", "b_second"]
    assert notes[1].text == "Second note, over two lines."       # one line, single spaces


def test_split_sentences_cuts_after_full_stops():
    assert split_sentences("Leave at 9:40 pm. Reach by 9! Is the PNR printed?") == \
        ["Leave at 9:40 pm.", "Reach by 9!", "Is the PNR printed?"]


@needs_model(EMBEDDINGS_MODEL)
def test_train_ticket_finds_the_rail_journey_note_first():
    from skills.search import search

    results = search("train ticket", FIXED_NOTES, k=4)

    assert results[0][0] == "rail_journey"
    assert results[0][1] > results[1][1]                 # clearly ahead of the next note
    assert all(-1.0 <= score <= 1.0 for _, score in results)


@needs_model(EMBEDDINGS_MODEL)
def test_meaning_beats_shared_words():
    from skills.search import search

    # "groceries" and "vegetables" appear in no note, but the shopping note means it
    assert search("groceries and vegetables to buy", FIXED_NOTES, k=1)[0][0] == "shopping"
