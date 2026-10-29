"""
check_setup.py - check your workbench before Episode 4 (step 1).

Same shared .venv and root .env as Episode 1. This also tells you which of today's
three local models are already on your computer (the rest download on first use).
Nothing is sent over the internet, and your key is never printed.

Run:  python check_setup.py
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

import llm

PACKAGES = ["langchain", "langchain-google-genai", "transformers", "torch",
            "sentence-transformers", "pillow", "rich"]

# (model id, download size, the step that uses it)
LOCAL_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "about 91 MB", "step 3"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "about 1 GB", "step 5"),
    ("openai/clip-vit-base-patch32", "about 605 MB", "step 8"),
]


def line(name: str, value: str) -> None:
    print(f"{name + ' ':.<40} {value}")


def is_downloaded(model_id: str) -> bool:
    """True if the model is already in your Hugging Face cache (checked without the internet)."""
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def main() -> int:
    problems = 0
    print("Episode 4 - setup check\n")

    v = sys.version_info
    ok_python = (3, 11) <= (v.major, v.minor) <= (3, 13)
    line("Python", f"{v.major}.{v.minor}.{v.micro} " + ("OK" if ok_python else "NOT SUPPORTED - use Python 3.13"))
    problems += not ok_python

    in_venv = sys.prefix != sys.base_prefix
    line("Virtual env", f"active ({Path(sys.prefix).name})" if in_venv else "NOT active - activate .venv first")
    problems += not in_venv

    for package in PACKAGES:
        try:
            line(package, version(package))
        except PackageNotFoundError:
            line(package, "MISSING - run: pip install -r requirements.txt")
            problems += 1

    line(".env file", f"found ({Path(llm.ENV_FILE).parent.name}/.env)" if llm.ENV_FILE
         else "NOT FOUND - copy .env.example to .env in the repo root")
    try:
        provider = llm.get_provider()
    except ValueError as error:          # a typo in LLM_PROVIDER
        line("Provider", f"WRONG - {error}")
        return 1
    line("Provider", provider)
    line("Model", llm.get_model_name(provider))

    names = llm.API_KEY_NAMES[provider]
    found = [n for n in names if os.getenv(n, "").strip() and not os.getenv(n, "").startswith("paste-your")]
    if not names:
        line("API key", "not needed (local model)")
    elif found:
        line("API key", f"{found[0]} found (hidden)")
    else:
        line("API key", f"MISSING - put {names[0]}=... in the root .env file")
        problems += 1

    print()
    for model_id, size, step in LOCAL_MODELS:
        line(model_id, "downloaded" if is_downloaded(model_id) else f"not yet ({size}, downloads in {step})")

    print()
    if problems:
        print(f"{problems} thing(s) to fix, see above.")
        if provider in llm.WHERE_TO_GET_A_KEY and not found:
            print("\n" + llm.WHERE_TO_GET_A_KEY[provider])
        return 1
    print(f"Provider {provider}, key found. OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
