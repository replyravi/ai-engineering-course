"""
chunklab/__main__.py - the command line: python -m chunklab <command> [options]

    show       see how each splitter cuts the handbook            (Steps 2-6)
    measure    chunk counts and sizes in MiniLM tokens            (Step 7)
    index      store every strategy in its own Chroma collection  (Steps 8-9)
    score      the hit@3 scoreboard                                (Steps 10-12)
    search     search one collection, with a metadata filter      (Step 13)
    faiss      FAISS next to Chroma: same hits, times, sizes      (Step 14)
    ask        answer with Gemini, citing the section breadcrumb  (Step 15)

Each command lives in its own file (chunklab/show.py, chunklab/measure.py, ...).
"-m" runs the chunklab folder as a package, so its files can import llm.py from this folder.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import importlib
import sys

COMMANDS = {
    "show": "chunklab.show",
    "measure": "chunklab.measure",
    "index": "chunklab.index",
    "score": "chunklab.score",
    "search": "chunklab.search",
    "faiss": "chunklab.faiss_compare",
    "ask": "chunklab.ask",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        commands = __doc__.split("Each command")[0].strip().split("\n", 1)[1]
        print("Usage: python -m chunklab <command> [options]   (add --help after a command for its options)")
        print(f"\nCommands:\n{commands}")
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    module = importlib.import_module(COMMANDS[argv[0]])
    return module.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
