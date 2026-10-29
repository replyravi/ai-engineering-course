"""
sync_template.py - keep the shared helper files of every episode identical to _template/.

Each episode folder carries its own copy of the shared files (so you can open one
folder and run it on its own). This tool copies them from _template/ and tells you
what had drifted.

    python tools/sync_template.py ep01-four-kinds-of-ai          copy into one episode
    python tools/sync_template.py --all                          copy into every epNN-* folder
    python tools/sync_template.py --check --all                  only REPORT drift, change nothing

Exit code 1 in --check mode when something drifted (handy before a commit).

Synced files: llm.py, testing/ and the two template-owned test files
(tests/conftest.py blocks the internet in tests, tests/test_llm.py checks llm.py).
Put episode-only fixtures in the episode's own test files, not in conftest.py.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import argparse
import difflib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "_template"

# What gets copied from _template/ into each episode folder.
SYNCED = ["llm.py", "testing", "tests/conftest.py", "tests/test_llm.py"]

# Root requirement files whose pinned lines must match the _template copies.
PINNED = ["requirements-base.txt", "requirements-crewai.txt"]


def template_files() -> list[Path]:
    """Every file to sync, as paths relative to _template/ (folders are expanded)."""
    files = []
    for name in SYNCED:
        path = TEMPLATE / name
        if path.is_dir():
            files += [p for p in sorted(path.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
        else:
            files.append(path)
    return [p.relative_to(TEMPLATE) for p in files]


def changed_lines(old: Path, new: Path) -> int:
    """How many lines differ between two text files."""
    a = old.read_text(encoding="utf-8").splitlines()
    b = new.read_text(encoding="utf-8").splitlines()
    diff = difflib.unified_diff(a, b, lineterm="", n=0)
    return sum(1 for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---"))


def sync_episode(episode: Path, check_only: bool) -> int:
    """Sync (or just check) one episode folder. Returns the number of drifted files."""
    print(episode.name)
    drifted = 0
    for rel in template_files():
        source, target = TEMPLATE / rel, episode / rel
        if not target.exists():
            status = "missing"
        elif source.read_bytes() == target.read_bytes():
            print(f"  {str(rel):<28} same")
            continue
        else:
            status = f"DIFFERENT ({changed_lines(target, source)} lines)"
        drifted += 1
        if check_only:
            print(f"  {str(rel):<28} {status}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"  {str(rel):<28} {status} -> copied from _template")

    # Files in the episode's testing/ folder that the template does not have (kept, just reported).
    known = {str(r) for r in template_files()}
    for extra in sorted((episode / "testing").rglob("*.py")) if (episode / "testing").exists() else []:
        rel = str(extra.relative_to(episode))
        if rel not in known:
            print(f"  {rel:<28} only in this episode (kept)")
    return drifted


def pinned_lines(path: Path) -> list[str]:
    """The package lines of a requirements file, without comments."""
    lines = [line.split("#")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def check_requirements() -> int:
    """Report whether the root requirement pins match the _template copies."""
    print("repo root")
    drifted = 0
    for name in PINNED:
        root_file, template_file = REPO / name, TEMPLATE / name
        if not root_file.exists() or not template_file.exists():
            print(f"  {name:<28} missing (root or _template)")
            drifted += 1
        elif pinned_lines(root_file) == pinned_lines(template_file):
            print(f"  {name:<28} pins same as _template")
        else:
            print(f"  {name:<28} PINS DIFFER from _template (fix by hand)")
            drifted += 1
    return drifted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Copy _template helpers into episode folders.")
    parser.add_argument("episodes", nargs="*", help="episode folder names, e.g. ep01-four-kinds-of-ai")
    parser.add_argument("--all", action="store_true", help="every epNN-* folder in the repo")
    parser.add_argument("--check", action="store_true", help="only report drift, change nothing")
    args = parser.parse_args(argv)

    episodes = sorted(REPO.glob("ep[0-9][0-9]-*")) if args.all else [REPO / name for name in args.episodes]
    episodes = [e for e in episodes if e.is_dir()] if args.all else episodes
    if not episodes:
        parser.error("name at least one episode folder, or use --all")
    for episode in episodes:
        if not episode.is_dir():
            parser.error(f"no such episode folder: {episode.name}")

    pin_drift = check_requirements()
    drifted = sum(sync_episode(episode, check_only=args.check) for episode in episodes)

    if args.check:
        total = drifted + pin_drift
        print(f"\n{total} file(s) drifted." if total else "\nEverything matches _template.")
        return 1 if total else 0
    print(f"\nDone. {drifted} file(s) updated." if drifted else "\nDone. Nothing needed updating.")
    if pin_drift:
        print("Note: the root requirement pins differ from _template. Make them match by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
