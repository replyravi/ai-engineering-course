"""
tests/zoo_fixtures.py - shared test helpers for this episode.

    from zoo_fixtures import draw_sample_photos, needs_model, sample_reviews

needs_model("Qwen/Qwen2.5-0.5B-Instruct") skips a test when that model was never downloaded.
Tests run offline (tests/conftest.py), so they can only use models already on your computer.
Run the matching script once with internet (it downloads the model), then the test runs too.

The tests check the CODE, not your data. So they use their own copy of the sample data:
  sample_reviews()            the ten Kettle Point reviews (tests/sample_reviews.txt)
  draw_sample_photos(folder)  the four sample drawings, drawn fresh into a temporary folder
Your own data/reviews.txt and photos/ can hold anything you like: no test reads them.
"""

import importlib.util
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache

from common import load_reviews

TESTS = Path(__file__).resolve().parent
EPISODE = TESTS.parent
SAMPLE_REVIEWS_FILE = TESTS / "sample_reviews.txt"   # a copy of the data/reviews.txt that ships with the course
PHOTO_TOOL = EPISODE / "tools" / "make_sample_photos.py"


def is_downloaded(model_id: str) -> bool:
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def needs_model(model_id: str):
    return pytest.mark.skipif(not is_downloaded(model_id),
                              reason=f"{model_id} is not downloaded yet (run its script once with internet)")


def sample_reviews() -> list[str]:
    """The ten sample reviews, whatever you wrote in data/reviews.txt."""
    return load_reviews(SAMPLE_REVIEWS_FILE)


def draw_sample_photos(folder) -> Path:
    """Draw cup.jpg, plant.jpg, shoe.jpg and book.jpg into folder and return the folder.

    Uses main() from tools/make_sample_photos.py, loaded from its file path
    (tools/ is a plain folder, not a Python package).
    """
    spec = importlib.util.spec_from_file_location("make_sample_photos", PHOTO_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    tool.main(folder)
    return Path(folder)
