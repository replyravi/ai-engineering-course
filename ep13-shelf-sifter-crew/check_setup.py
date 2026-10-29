"""
check_setup.py - Step 1: check the CrewAI environment (.venv-crewai) and your key.

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

# What this episode imports, and the version tested with it (requirements-crewai.txt).
PACKAGES = ["crewai", "crewai-tools", "langchain-core", "beautifulsoup4", "requests",
            "chromadb", "onnxruntime", "numpy", "python-dotenv", "pytest"]


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
    print("AI Engineering From Zero - CrewAI setup check (Episodes 13-14)\n")

    # 1. Python: CrewAI 1.15 needs Python below 3.14 (we use 3.11 to 3.13)
    v = sys.version_info
    if (3, 11) <= (v.major, v.minor) <= (3, 13):
        line("Python", f"{v.major}.{v.minor}.{v.micro} OK")
    else:
        line("Python", f"{v.major}.{v.minor}.{v.micro} NOT SUPPORTED - make .venv-crewai with Python 3.13")
        problems += 1

    # 2. The CrewAI environment, not the LangChain one
    if sys.prefix == sys.base_prefix:
        line("Virtual env", "NOT active - activate .venv-crewai first (Step 0)")
        problems += 1
    elif "crewai" in Path(sys.prefix).name:
        line("Virtual env", f"active ({Path(sys.prefix).name})")
    else:
        line("Virtual env", f"active ({Path(sys.prefix).name}) - this episode needs .venv-crewai")

    # 3. Libraries
    for package in PACKAGES:
        try:
            line(package, version(package))
        except PackageNotFoundError:
            line(package, "MISSING - run: pip install -r requirements.txt")
            problems += 1

    # 4. Privacy switches (privacy.py sets them unless your .env says otherwise)
    telemetry_off = os.getenv("CREWAI_DISABLE_TELEMETRY", "").lower() == "true"
    tracing_off = os.getenv("CREWAI_TRACING_ENABLED", "").lower() == "false"
    line("CrewAI telemetry", "off" if telemetry_off else "ON - set CREWAI_DISABLE_TELEMETRY=true in .env")
    line("CrewAI tracing", "off" if tracing_off else "ON - set CREWAI_TRACING_ENABLED=false in .env")

    # 5. The .env file, the provider and the key
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
    found = {name: os.getenv(name, "").strip() for name in key_names}
    found = {name: key for name, key in found.items() if key and not key.startswith("paste-your")}
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
        return 1
    if args.offline:
        print("All checks passed (offline: no question was sent).")
        return 0

    # 6. One tiny question through CrewAI's own LLM class, the one our agents use
    print(f"Asking {provider}/{model} one tiny question through crewai.LLM ...")
    try:
        reply = llm.get_crewai_llm().call("Reply with one short sentence: are you ready to help?")
    except Exception as error:  # a wrong key, no internet, the free-tier limit ...
        print(f"The model call failed: {type(error).__name__}: {error}")
        print("(429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and run again)")
        return 1
    print(f"Reply: {str(reply).strip()}")
    print(f"\n{provider.capitalize()} ready. Next: python robots_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
