"""
ask.py - Step 12: chat with the Dabbaloop handbook in your terminal.

    python ask.py                                   # keeps answering until you type exit
    python ask.py "Do you deliver on Sunday?"       # one question, then stop

Every answer ends with [file#chunk] citations, and the bot says
"I don't know based on the handbook." when the handbook has no answer.
Build the index first:  python ingest.py

Author: Ravi Shanker Singh (TechWithRavi)
"""

import sys

from ingest import IndexNotBuiltError
from llm import MissingAPIKeyError
from rag_chain import build_chain, cited_tags

EXIT_WORDS = {"exit", "quit", "bye"}


def format_reply(result):
    """Turn {"answer", "sources"} into the text we print."""
    lines = [f"Bot: {result['answer']}"]
    used = set(cited_tags(result["answer"]))
    for source in result["sources"]:
        mark = "cited" if source["tag"] in used else "     "
        lines.append(f"     {mark}  {source['tag']:<18} {source['where']}  (distance {source['distance']:.3f})")
    return "\n".join(lines)


def chat(chain, read=input, write=print):
    """The question loop. `read` and `write` can be replaced in tests."""
    write("Ask the Dabbaloop handbook a question. Type exit to quit.")
    while True:
        try:
            question = read("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in EXIT_WORDS:
            break
        if not question:
            continue
        try:
            write(format_reply(chain.invoke(question)))
        except Exception as error:  # e.g. no internet, or the free-tier limit
            write(f"Bot: the model call failed: {error}")
            write("     (429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and ask again)")
    write("Bye!")


def main():
    try:
        chain = build_chain()
    except (MissingAPIKeyError, IndexNotBuiltError) as error:
        print(str(error).strip())
        return 1

    if len(sys.argv) > 1:  # one question from the command line
        print(format_reply(chain.invoke(" ".join(sys.argv[1:]))))
        return 0
    chat(chain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
