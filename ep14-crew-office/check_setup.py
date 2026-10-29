"""
check_setup.py - Step 1: the same CrewAI environment as Episode 13, plus this episode's packages.

    python check_setup.py             (all checks, then ONE tiny question to the model)
    python check_setup.py --offline   (the checks only, nothing is sent over the internet)

Your key is never printed in full.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import llm

# What this episode imports, and the versions it was tested with (requirements.txt).
PACKAGES = ["crewai", "crewai-tools", "pandas", "numexpr", "onnxruntime", "tokenizers",
            "huggingface_hub", "chromadb", "lancedb", "numpy", "python-dotenv", "pytest"]


def mask(key: str) -> str:
    """Show only the first and last 4 characters of a key."""
    return "*" * len(key) if len(key) < 12 else f"{key[:4]}...{key[-4:]}"


def line(name: str, value: str) -> None:
    print(f"{name + ' ':.<25} {value}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the CrewAI environment and the model key.")
    parser.add_argument("--offline", action="store_true", help="do not send the tiny test question")
    args = parser.parse_args(argv)

    problems = 0
    print("AI Engineering From Zero - Episode 14 setup check (the .venv-crewai of Episode 13)\n")

    v = sys.version_info
    if (3, 11) <= (v.major, v.minor) <= (3, 13):
        line("Python", f"{v.major}.{v.minor}.{v.micro} OK")
    else:
        line("Python", f"{v.major}.{v.minor}.{v.micro} NOT SUPPORTED - make .venv-crewai with Python 3.13")
        problems += 1

    if sys.prefix == sys.base_prefix:
        line("Virtual env", "NOT active - activate .venv-crewai first (Step 0)")
        problems += 1
    else:
        name = Path(sys.prefix).name
        line("Virtual env", f"active ({name})" + ("" if "crewai" in name else " - this episode needs .venv-crewai"))

    for package in PACKAGES:
        try:
            line(package, version(package))
        except PackageNotFoundError:
            line(package, "MISSING - run: pip install -r requirements.txt")
            problems += 1

    from tools.tone import model_is_downloaded     # imported here: it needs the packages above
    line("Tone model", "downloaded" if model_is_downloaded() else "not yet (Step 3 downloads it once)")
    telemetry_off = os.getenv("CREWAI_DISABLE_TELEMETRY", "").lower() == "true"
    tracing_off = os.getenv("CREWAI_TRACING_ENABLED", "").lower() == "false"
    line("CrewAI telemetry", "off" if telemetry_off else "ON - set CREWAI_DISABLE_TELEMETRY=true in .env")
    line("CrewAI tracing", "off" if tracing_off else "ON - set CREWAI_TRACING_ENABLED=false in .env")
    ort_off = os.getenv("ORT_DISABLE_TELEMETRY", "") == "1"
    line("ONNX Runtime telemetry", "off" if ort_off else "ON - set ORT_DISABLE_TELEMETRY=1 in .env")

    if llm.ENV_FILE:
        line(".env file", f"found ({Path(llm.ENV_FILE).parent.name}/.env)")
    else:
        line(".env file", "NOT FOUND - copy .env.example to .env in the repo root")
        problems += 1
    try:
        provider = llm.get_provider()
    except ValueError as error:   # a typo in LLM_PROVIDER
        line("Provider", f"WRONG - {error}")
        return 1
    model = llm.get_model_name(provider)
    line("Provider", provider)
    line("Model", f"{model}   (CrewAI name: {provider}/{model})")

    key_names = llm.API_KEY_NAMES[provider]
    found = {n: os.getenv(n, "").strip() for n in key_names}
    found = {n: key for n, key in found.items() if key and not key.startswith("paste-your")}
    if not key_names:
        line("API key", "not needed (local model)")
    elif found:
        for name, key in found.items():
            line("API key", f"{name} found ({mask(key)})")
    else:
        line("API key", f"MISSING - put {key_names[0]}=... in the .env file")
        problems += 1

    print()
    if problems:
        print(f"{problems} thing(s) to fix, see above.")
        if provider in llm.WHERE_TO_GET_A_KEY and not found:
            print("\n" + llm.WHERE_TO_GET_A_KEY[provider])
        print("\nNo key yet? The crew steps also run offline with a scripted model: add --replay.")
        return 1
    if args.offline:
        print("All checks passed (offline: no question was sent).")
        return 0

    print(f"Asking {provider}/{model} one tiny question through crewai.LLM ...")
    try:
        reply = llm.get_crewai_llm().call("Reply with exactly one word: OK")
    except Exception as error:  # a wrong key, no internet, the free-tier limit ...
        print(f"The model call failed: {type(error).__name__}: {error}")
        print("(429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and run again)")
        return 1
    print(f"Reply: {str(reply).strip()}")
    print(f"\nprovider={provider}, model={model}, reply OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
