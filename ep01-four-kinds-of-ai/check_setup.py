"""
check_setup.py - check your AI workbench once, before the first call (step 4).

    python check_setup.py                              (inside ep01-four-kinds-of-ai)
    python ep01-four-kinds-of-ai/check_setup.py        (from the repo root)

Nothing is sent over the internet, and your key is never printed in full.
"""

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import llm

HERE = Path(__file__).resolve().parent
PACKAGES = ["langchain", "langchain-google-genai", "scikit-learn", "pandas",
            "matplotlib", "sentence-transformers", "torch", "python-dotenv"]


def mask(key: str) -> str:
    """Show only the first and last 4 characters of an API key.

    Why this exists: check_setup must prove a key is present without printing
    the whole secret on screen or into a screenshot.

    What it does: "AIza...wxyz" for a long key. A short string becomes stars.

    Called by: main (below) and tests/test_pipeline.py.
    """
    if len(key) < 12:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def repo_root() -> Path:
    """Find the course folder, the one that contains .env.example.

    Why this exists: this script can be started inside the episode folder
    or from the repo root. The .env file lives at the root, not here.

    What it does: walk from this file upward until a folder contains
    .env.example. If none does, return this folder's parent.

    Called by: main (below).
    """
    for folder in (HERE, *HERE.parents):
        if (folder / ".env.example").is_file():
            return folder
    return HERE.parent


def ignored_by_git(path: Path) -> str:
    """Say whether .env is kept out of git.

    Why this exists: a key inside a commit is a leaked key. This check
    tells you to add ".env" to .gitignore before you continue.

    What it does: ask `git check-ignore`. If git is missing (a ZIP download),
    read .gitignore and look for a line that is exactly ".env".

    Called by: main (below) and tests/test_pipeline.py.
    """
    try:
        result = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=path.parent,
                                capture_output=True, timeout=10)
        if result.returncode == 0:
            return "yes"
        if result.returncode == 1:
            return "NO - add a line .env to .gitignore"
    except (OSError, subprocess.TimeoutExpired):
        pass   # git is not installed
    # Not a git folder (for example a ZIP download): read .gitignore ourselves.
    for folder in (path.parent, *path.parent.parents):
        gitignore = folder / ".gitignore"
        if gitignore.is_file():
            lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
            return "yes (.gitignore lists .env)" if ".env" in lines else "NO - add a line .env to .gitignore"
    return "unknown (no .gitignore found)"


def line(name: str, value: str) -> None:
    """Print one setup row: the name, dots, then the value.

    Why this exists: the report is easier to scan when every row lines up.

    Called by: main (below) only.
    """
    print(f"{name + ' ':.<25} {value}")


def main() -> int:
    """Check Python, the virtual env, libraries, .env, and the key. Send nothing.

    Why this exists: step 4, before the first model call. A missing library
    or a missing key should be a short report, not a traceback.

    What it does: does not call get_llm(), so no request is sent. It calls
    llm.get_provider() and llm.get_model_name(), then mask() on the key and
    ignored_by_git() on the .env path. Returns 0 when nothing is wrong, 1
    when something must be fixed.

    Called by: the bottom of this file, and tests/test_pipeline.py.
    """
    problems = 0
    print("AI Engineering From Zero - setup check\n")

    # 1. Python version: 3.11 to 3.13 (3.13 recommended)
    v = sys.version_info
    if (3, 11) <= (v.major, v.minor) <= (3, 13):
        line("Python", f"{v.major}.{v.minor}.{v.micro} OK")
    else:
        line("Python", f"{v.major}.{v.minor}.{v.micro} NOT SUPPORTED - install Python 3.13")
        problems += 1

    # 2. Virtual environment
    if sys.prefix != sys.base_prefix:
        line("Virtual env", f"active ({Path(sys.prefix).name})")
    else:
        line("Virtual env", "NOT active - activate .venv first (see SETUP.md)")
        problems += 1

    # 3. Libraries
    for package in PACKAGES:
        try:
            line(package, version(package))
        except PackageNotFoundError:
            line(package, "MISSING - run: pip install -r requirements-base.txt")
            problems += 1

    # 4. The .env file and the key
    root = repo_root()
    env_file = Path(llm.ENV_FILE) if llm.ENV_FILE else root / ".env"
    if llm.ENV_FILE:
        line(".env file", f"found ({env_file.parent.name}/.env)")
    else:
        line(".env file", "NOT FOUND - copy .env.example to .env in the repo root")
        problems += 1

    try:
        provider = llm.get_provider()
    except ValueError as error:   # a typo in LLM_PROVIDER
        line("Provider", f"WRONG - {error}")
        print(f"\n{problems + 1} thing(s) to fix, see above.")
        return 1
    line("Provider", provider)
    line("Model", llm.get_model_name(provider))

    key_names = llm.API_KEY_NAMES[provider]
    found = {name: os.getenv(name, "").strip() for name in key_names}
    found = {name: key for name, key in found.items() if key and not key.startswith("paste-your")}
    if not key_names:
        line("API key", "not needed (local model)")
    elif found:
        for name, key in found.items():
            line("API key", f"{name} found ({mask(key)})")
        if len(found) > 1:
            print("  Note: both are set. GOOGLE_API_KEY wins.")
    else:
        line("API key", f"MISSING - put {key_names[0]}=... in the .env file")
        problems += 1

    line(".env ignored by git", ignored_by_git(env_file))

    print()
    if problems == 0:
        print("All good. Next: python first_call.py")
        return 0
    print(f"{problems} thing(s) to fix, see above.")
    if provider in llm.WHERE_TO_GET_A_KEY and not found:
        print("\n" + llm.WHERE_TO_GET_A_KEY[provider])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
