"""
tests/model_fixtures.py - the real downloaded models, for the tests that need them.

Tests run with the internet blocked (tests/conftest.py). So a model must already be on your
computer. If it is not, these fixtures SKIP the test and tell you the fix, instead of failing.

Test files import what they need, for example:
    from model_fixtures import decoder, t5  # noqa: F401
"""

import pytest

FIX = "run the step once with internet (it downloads the model), then run the tests again"


@pytest.fixture(scope="session")
def decoder():
    """(tokenizer, model) for SmolLM2-135M."""
    from decoder import load_decoder

    try:
        return load_decoder()
    except Exception as error:   # not downloaded yet, and tests run offline
        pytest.skip(f"SmolLM2-135M not downloaded yet: {FIX} (python next_token.py load) [{type(error).__name__}]")


@pytest.fixture(scope="session")
def t5():
    """(tokenizer, model) for T5-small, loaded with eager attention so it returns its weights."""
    from three_families import load_t5

    try:
        return load_t5()
    except Exception as error:
        pytest.skip(f"T5-small not downloaded yet: {FIX} (python three_families.py) [{type(error).__name__}]")


@pytest.fixture(scope="session")
def classifier():
    """The small sentiment encoder (DistilBERT fine-tuned on SST-2)."""
    from three_families import load_classifier

    try:
        return load_classifier()
    except Exception as error:
        pytest.skip(f"The sentiment encoder is not downloaded yet: {FIX} (python three_families.py) "
                    f"[{type(error).__name__}]")
