"""
tests/lab_fixtures.py - shared test helpers for Tiffin Toolbox.

    from lab_fixtures import FIXED_NOTES, draw_sample_photo, needs_model

needs_model("Qwen/Qwen2.5-0.5B-Instruct") skips a test when that model was never downloaded.
Tests run offline (tests/conftest.py sets HF_HUB_OFFLINE=1 and blocks the internet), so they can
only use models already on your computer. Run the matching step once with internet, then test again.

The tests check the CODE, not your data. So they use their own fixed sentences and draw their
own copy of the desk picture: you can put anything in my_notes/ and photos/.
"""

import importlib.util
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache

from common import Note

TESTS = Path(__file__).resolve().parent
EPISODE = TESTS.parent
SAMPLE_CARDS_FILE = TESTS / "sample_model_cards.json"   # a frozen copy of model_cards.json (2026-10-26)
PHOTO_TOOL = EPISODE / "tools" / "make_desk_photo.py"

# Fixed sentences for the ranking tests (not your my_notes/ files).
FIXED_NOTES = [
    Note("rail_journey", "Booked two sleeper berths on the overnight train to Jaipur; reach the station by 9 pm."),
    Note("shopping", "Buy rice, lentils, onions, tomatoes, curd and a dozen eggs from the corner shop."),
    Note("database_class", "Third normal form removes columns that depend on another non-key column."),
    Note("plants", "Water the tulsi every morning and the money plant once a week."),
]


def is_downloaded(model_id: str) -> bool:
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def needs_model(model_id: str):
    return pytest.mark.skipif(not is_downloaded(model_id),
                              reason=f"{model_id} is not downloaded yet (run its step once with internet)")


def draw_sample_photo(folder) -> Path:
    """Draw desk.jpg into folder with tools/make_desk_photo.py and return its path.

    The tool is loaded from its file path (tools/ is a plain folder, not a Python package).
    """
    spec = importlib.util.spec_from_file_location("make_desk_photo", PHOTO_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool.main(folder)
