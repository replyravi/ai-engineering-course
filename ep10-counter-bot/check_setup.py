"""
check_setup.py - Step 1: is everything ready for Episode 10? No API key needed to run it.

It prints the AI provider and model from your .env, whether a key was found (never the key itself),
and the versions of the libraries this episode uses, including the one NEW package:
langgraph-checkpoint-sqlite (Step 15 saves chats with it).

Run:  python check_setup.py

Author: Ravi Shanker Singh (TechWithRavi)
"""

import sys
from importlib.metadata import PackageNotFoundError, version

from llm import ENV_FILE, MissingAPIKeyError, check_api_key, get_model_name, get_provider

# package -> the version this episode was tested with
PACKAGES = {
    "langchain": "1.4.7",
    "langgraph": "1.2.16",
    "langchain-community": "0.4.2",      # the loaders (archived in 2026, pinned on purpose)
    "pypdf": "6.21.2",                   # used by PyPDFLoader
    "beautifulsoup4": "4.15.0",          # used by BSHTMLLoader
    "lxml": "6.1.3",                     # used by BSHTMLLoader
    "langchain-chroma": "1.1.0",         # the index (Step 14)
    "langgraph-checkpoint-sqlite": "3.1.2",   # NEW: saved chats (Step 15), from this folder's requirements.txt
}


def main():
    print(f"Python {sys.version.split()[0]}")
    print(f"Settings file: {ENV_FILE or 'no .env found'}")
    try:
        provider = get_provider()
    except ValueError as error:
        print(error)
        return 1
    print(f"Provider: {provider}   Model: {get_model_name(provider)}")
    try:
        check_api_key(provider)
        print("API key: found (Steps 9 to 15 can call the model)")
    except MissingAPIKeyError:
        print("API key: NOT found. Steps 2 to 8 work without it; Steps 9 to 15 need it (see Step 0.3).")

    missing = 0
    print("\nLibraries:")
    for package, tested in PACKAGES.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "MISSING"
            missing += 1
        mark = "ok" if installed == tested else ("<- install it" if installed == "MISSING" else f"(tested with {tested})")
        print(f"  {package:<28} {installed:<9} {mark}")
    if missing:
        print("\nFix: run  pip install -r requirements.txt  in this folder (with the .venv active).")
        return 1
    print("\nReady for Episode 10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
