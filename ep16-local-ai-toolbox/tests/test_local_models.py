"""
tests/test_local_models.py - the REAL local models, offline (steps 4, 6, 10 and 13).

Each test is skipped (an "s" in the pytest output) until its model is downloaded:
run steps 4 and 10 once with internet, then test again. The answers come from greedy
decoding (do_sample=False), so the same model gives the same answer on every run.
"""

import re

from common import CHAT_MODEL, EMBEDDINGS_MODEL, VISION_MODEL
from lab_fixtures import draw_sample_photo, needs_model
from skills.lazy import lazy_load, loaded_names, release

NOTE = ("Jaipur trip for the wedding. Book two sleeper berths on the overnight train on Thursday. "
        "The train leaves at 9:40 pm, so reach the station by 9.")


@needs_model(CHAT_MODEL)
def test_local_summary_is_short_and_the_same_every_time():
    from skills.summarise import MAX_NEW_TOKENS, summarise

    first, second = summarise(NOTE), summarise(NOTE)

    assert first.text and first.text == second.text          # greedy: same input, same answer
    assert 0 < first.tokens_out <= MAX_NEW_TOKENS
    assert first.tokens_in > first.tokens_out


@needs_model(CHAT_MODEL)
def test_local_translation_writes_devanagari():
    from skills.translate import translate

    reply = translate("Please water the plants.", "Hindi")

    assert re.search(r"[ऀ-ॿ]", reply.text)          # Hindi letters (right or wrong, see step 6)


@needs_model(EMBEDDINGS_MODEL)
def test_highlighter_copies_one_sentence_of_the_note():
    from skills.summarise import highlight, split_sentences

    assert highlight(NOTE) in split_sentences(NOTE)


@needs_model(VISION_MODEL)
def test_vision_model_sees_the_red_mug(tmp_path):
    from skills.ask_photo import ask, open_photo

    answer = ask(open_photo(draw_sample_photo(tmp_path)), "What colour is the mug?")

    assert "red" in answer.text.lower()
    assert answer.seconds > 0


def test_lazy_load_loads_once_and_release_forgets():
    release()
    calls = []

    first = lazy_load("toy", lambda: calls.append(1) or object())
    second = lazy_load("toy", lambda: calls.append(1) or object())

    assert first is second and calls == [1]                   # loaded only once
    assert "toy" in loaded_names()
    assert release("toy") == ["toy"] and "toy" not in loaded_names()


def test_the_sample_photo_is_drawn_the_same_every_time(tmp_path):
    a = draw_sample_photo(tmp_path / "a").read_bytes()
    b = draw_sample_photo(tmp_path / "b").read_bytes()

    assert a == b
