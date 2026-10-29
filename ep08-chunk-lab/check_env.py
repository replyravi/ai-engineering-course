"""
check_env.py - Step 1: are the four splitters, the stores and the local embedder ready?

    python check_env.py

It needs no API key. The first run on a new computer downloads the MiniLM model (about 90 MB),
unless you already have it from an earlier episode.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import sys
from importlib.metadata import version

from chunklab.strategies import EMBEDDER, HANDBOOK_DIR, load_handbook, load_questions

PACKAGES = ["langchain-text-splitters", "langchain-chroma", "chromadb", "faiss-cpu", "tiktoken",
            "sentence-transformers", "rich"]


def main():
    print(f"Python {sys.version.split()[0]}")
    print("Packages: " + ", ".join(f"{name} {version(name)}" for name in PACKAGES))

    from langchain_text_splitters import (  # noqa: F401  (the import itself is the test)
        CharacterTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter,
    )
    print("Splitters: CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, "
          "MarkdownHeaderTextSplitter imported")

    docs = load_handbook()
    characters = sum(len(doc.page_content) for doc in docs)
    print(f"Handbook: {len(docs)} Markdown files in {HANDBOOK_DIR.parent.name}/{HANDBOOK_DIR.name}/ "
          f"({characters:,} characters), {len(load_questions())} test questions")

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model = SentenceTransformer(EMBEDDER, device="cpu")
    dim = len(model.encode("Is the kitchen nut-free?"))
    raw_limit = AutoTokenizer.from_pretrained(EMBEDDER).model_max_length
    print(f"Embedder: {EMBEDDER} -> {dim} numbers per text")
    print(f"  max length: {model.max_seq_length} tokens. Longer text is cut off, with no warning.")
    print(f"  (the tokenizer file on its own says {raw_limit}; the embedder uses {model.max_seq_length})")
    print(f"splitters OK, embedder OK (dim {dim}, max length {model.max_seq_length})")


if __name__ == "__main__":
    main()
