"""
llm.py - the ONE place that decides which AI model this project uses.

Every episode project copies this file. Your code just calls:

    from llm import get_llm, get_embeddings
    llm = get_llm()                # a LangChain chat model
    embeddings = get_embeddings()  # turns text into vectors (for RAG)

    from llm import get_crewai_llm # only in the CrewAI episodes
    crew_llm = get_crewai_llm()

To switch the AI provider you change ONE line in your .env file:

    LLM_PROVIDER=gemini      (default, FREE key from Google AI Studio)
    LLM_PROVIDER=openai      (paid key)
    LLM_PROVIDER=anthropic   (paid key, Claude models)
    LLM_PROVIDER=ollama      (free, runs on your own computer, no key)

Optional: LLM_MODEL=<model name> picks a different model of that provider.

There is ONE .env file for the whole course, at the repo root (ai-engineering-course/.env).
This file finds it from any episode folder, so you never copy your key around.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
Model names checked on 2026-10-23.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# 0. Find and read the ONE .env file of the course
# ---------------------------------------------------------------------------

def find_env_file(filename: str = ".env") -> str:
    """Return the path of the nearest .env file, or "" if there is none.

    We look in the folder you run Python from, then in each parent folder.
    If that finds nothing, we do the same starting from the folder of this file.
    So "python first_call.py" inside ep01-four-kinds-of-ai/ finds ai-engineering-course/.env.

    (Plain load_dotenv() starts its search in different places depending on how
    Python was started - a script, the VS Code debugger, a notebook - so we search ourselves.)
    """
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for folder in (start, *start.parents):
            candidate = folder / filename
            if candidate.is_file():
                return str(candidate)
    return ""


# Read the .env file into os.environ. Settings already in your environment win.
ENV_FILE = find_env_file()
if ENV_FILE:
    load_dotenv(ENV_FILE)


# ---------------------------------------------------------------------------
# Settings (checked against the providers' docs on 2026-10-23)
# ---------------------------------------------------------------------------

# The model each provider uses when LLM_MODEL is empty.
DEFAULT_MODELS = {
    # Free on the Google AI Studio free tier. Fast, and recommended by Google for new projects.
    # Smarter (but slower and uses more quota): gemini-3.8-flash
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-6-luna",          # OpenAI's lowest-cost model
    "anthropic": "claude-haiku-5-5",  # Anthropic's fastest, lowest-cost Claude model
    "ollama": "qwen3.5:4b",          # small local model with tool calling (run: ollama pull qwen3.5:4b)
}

# Which environment variable holds the key for each provider.
# (Gemini accepts GEMINI_API_KEY or GOOGLE_API_KEY. If both are set, GOOGLE_API_KEY wins.)
API_KEY_NAMES = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "ollama": [],  # Ollama runs on your computer, no key needed
}

# Step-by-step help shown when a key is missing.
WHERE_TO_GET_A_KEY = {
    "gemini": (
        "Get a FREE Gemini key (no credit card needed):\n"
        "  1. Open https://aistudio.google.com/apikey\n"
        "  2. Sign in with your Google account and accept the terms\n"
        "  3. Click 'Create API key' and copy it\n"
        "  4. Put it in your .env file like this:  GEMINI_API_KEY=paste-your-key-here"
    ),
    "openai": (
        "Get an OpenAI key (paid, needs billing):\n"
        "  1. Open https://platform.openai.com/api-keys\n"
        "  2. Click 'Create new secret key' and copy it\n"
        "  3. Put it in your .env file like this:  OPENAI_API_KEY=paste-your-key-here"
    ),
    "anthropic": (
        "Get an Anthropic (Claude) key (paid, needs billing):\n"
        "  1. Open https://platform.claude.com/settings/keys\n"
        "  2. Click 'Create Key' and copy it\n"
        "  3. Put it in your .env file like this:  ANTHROPIC_API_KEY=paste-your-key-here"
    ),
}

SUPPORTED_PROVIDERS = list(DEFAULT_MODELS)

# The default embeddings model runs on YOUR computer, for free (downloads ~90 MB once).
DEFAULT_LOCAL_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_EMBEDDINGS_MODEL = "gemini-embedding-2"

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class MissingAPIKeyError(RuntimeError):
    """Raised when the .env file has no API key for the chosen provider."""


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def get_provider() -> str:
    """Return the provider name from .env (gemini, openai, anthropic or ollama)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={provider!r} is not supported. "
            f"Use one of: {', '.join(SUPPORTED_PROVIDERS)} (in your .env file)."
        )
    return provider


def get_model_name(provider: str | None = None) -> str:
    """Return LLM_MODEL from .env, or the default model of the provider."""
    provider = provider or get_provider()
    model = os.getenv("LLM_MODEL", "").strip()
    return model or DEFAULT_MODELS[provider]


def check_api_key(provider: str) -> None:
    """Stop with a friendly message if the provider's API key is missing."""
    key_names = API_KEY_NAMES[provider]
    if not key_names:
        return  # Ollama: no key needed
    for name in key_names:
        value = os.getenv(name, "").strip()
        if value and not value.startswith("paste-your"):
            return  # found a key
    raise MissingAPIKeyError(
        f"\n\nNo API key found for LLM_PROVIDER={provider}.\n"
        f"Looked for: {' or '.join(key_names)} in your .env file.\n\n"
        f"{WHERE_TO_GET_A_KEY[provider]}\n\n"
        "No .env file yet? In the ai-engineering-course folder (the repo root), copy .env.example to .env:\n"
        "  macOS / Linux:       cp .env.example .env\n"
        "  Windows PowerShell:  Copy-Item .env.example .env\n"
        f"(.env file in use right now: {ENV_FILE or 'none found'})\n"
    )


# ---------------------------------------------------------------------------
# 1. Chat model for LangChain
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0, **kwargs):
    """Return a LangChain chat model for the provider chosen in .env.

    temperature: how "creative" the answers are (0 = focused, 1 = creative).
        Note (2026): most cloud models now choose their own sampling.
        - Gemini 3.5 Flash-Lite, 3.6 Flash and newer ignore temperature
          (Google deprecated it on 2026-07-21).
        - Claude Sonnet 5.5, Claude Fable 5.1 and OpenAI reasoning models reject values other than 1.
        So we only send temperature to local Ollama models, where it still works.
    **kwargs: extra settings passed straight to the model class
        (for example max_tokens=500, or thinking_level="low" for Gemini).
    """
    provider = get_provider()
    model = get_model_name(provider)
    check_api_key(provider)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, **kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, **kwargs)

    # provider == "ollama"
    from langchain_ollama import ChatOllama

    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    return ChatOllama(model=model, temperature=temperature, base_url=base_url, **kwargs)


# ---------------------------------------------------------------------------
# 2. Embeddings (text -> numbers) for RAG and vector databases
# ---------------------------------------------------------------------------

def get_embeddings():
    """Return an embeddings model.

    Default: a free LOCAL model (sentence-transformers) - no key, no internet after
    the first download. Set EMBEDDINGS_PROVIDER=gemini in .env to use Google's
    gemini-embedding-2 instead (free tier, needs your Gemini key).

    Important: if you change the embeddings model, rebuild your vector database.
    Vectors from different models cannot be mixed.
    """
    provider = os.getenv("EMBEDDINGS_PROVIDER", "local").strip().lower()
    model = os.getenv("EMBEDDINGS_MODEL", "").strip()

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model or DEFAULT_LOCAL_EMBEDDINGS_MODEL)

    if provider == "gemini":
        check_api_key("gemini")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model or DEFAULT_GEMINI_EMBEDDINGS_MODEL)

    raise ValueError(f"EMBEDDINGS_PROVIDER={provider!r} is not supported. Use: local or gemini.")


# ---------------------------------------------------------------------------
# 3. LLM for CrewAI (CrewAI episodes only - CrewAI has its own LLM class)
# ---------------------------------------------------------------------------

def get_crewai_llm(**kwargs):
    """Return a crewai.LLM for the provider chosen in .env.

    CrewAI names models as "<provider>/<model>", for example "gemini/gemini-3.5-flash-lite".
    Needs the CrewAI requirements (requirements-crewai.txt).
    """
    from crewai import LLM

    provider = get_provider()
    model = get_model_name(provider)
    check_api_key(provider)

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        return LLM(model=f"ollama/{model}", base_url=base_url, **kwargs)

    # gemini -> "gemini/...", openai -> "openai/...", anthropic -> "anthropic/..."
    # CrewAI reads the API key from the same environment variables as LangChain.
    return LLM(model=f"{provider}/{model}", **kwargs)


if __name__ == "__main__":
    # Quick check:  python llm.py
    p = get_provider()
    print(f"Settings file: {ENV_FILE or 'no .env found'}")
    print(f"Provider: {p}   Model: {get_model_name(p)}")
    try:
        check_api_key(p)
    except MissingAPIKeyError as error:
        print(error)  # friendly help instead of a long traceback
        raise SystemExit(1)
    print("API key found. Asking the model a tiny question...")
    try:
        chat_model = get_llm()
    except ImportError:
        # The CrewAI environment (episodes 13-14) has no LangChain model packages,
        # so we check CrewAI's own LLM instead.
        print(get_crewai_llm().call("Say hello in one short sentence."))
    else:
        # .text (not .content): Gemini 3+ returns content as a list of blocks.
        print(chat_model.invoke("Say hello in one short sentence.").text)
