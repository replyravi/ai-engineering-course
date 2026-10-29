"""
tests/model_fixtures.py - the real downloaded models, for the tests that need them.

Tests run with the internet blocked (tests/conftest.py). So a model must already be on your
computer. If it is not, these fixtures SKIP the test and tell you the fix, instead of failing.

Test files import what they need, for example:
    from model_fixtures import tokenizer, encoder  # noqa: F401
"""

import pytest

FIX = "run the step once with internet (it downloads the model), then run the tests again"


@pytest.fixture(scope="session")
def tokenizer():
    from encoder import load_tokenizer

    try:
        return load_tokenizer()
    except Exception as error:   # not downloaded yet, and tests run offline
        pytest.skip(f"DistilBERT tokenizer not downloaded yet: {FIX} (python tokens.py) [{type(error).__name__}]")


@pytest.fixture(scope="session")
def encoder():
    """(tokenizer, model) for DistilBERT."""
    from encoder import load_encoder

    try:
        return load_encoder()
    except Exception as error:
        pytest.skip(f"DistilBERT not downloaded yet: {FIX} (python embeddings_peek.py) [{type(error).__name__}]")


@pytest.fixture(scope="session")
def tiktoken_ready():
    """True once both tiktoken files are on your computer."""
    import tiktoken

    try:
        for name in ["cl100k_base", "o200k_base"]:
            tiktoken.get_encoding(name)
    except Exception as error:
        pytest.skip(f"tiktoken files not downloaded yet: {FIX} (python tokens.py compare) [{type(error).__name__}]")
    return True


@pytest.fixture(scope="session")
def minilm():
    """The local embeddings model from llm.py (all-MiniLM-L6-v2)."""
    from llm import get_embeddings

    try:
        embeddings = get_embeddings()
        embeddings.embed_query("warm-up")
        return embeddings
    except Exception as error:
        pytest.skip(f"MiniLM not downloaded yet: {FIX} (python meaning_map.py) [{type(error).__name__}]")
