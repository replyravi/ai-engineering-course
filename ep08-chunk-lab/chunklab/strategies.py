"""
chunklab/strategies.py - the handbook, and the four ways we cut it into chunks.

    character   CharacterTextSplitter: cut at blank lines only, then glue pieces up to 400 characters
    recursive   RecursiveCharacterTextSplitter: blank line, then line break, then space (Episode 07's setting)
    tokens      TokenTextSplitter: exactly 100 tiktoken tokens per chunk (encoding cl100k_base)
    headers     MarkdownHeaderTextSplitter first, then a recursive split of long sections,
                measured in the embedder's own tokens

Every chunk is a LangChain Document. Its metadata always has "source" (the file name) and
"chunk" (its number inside that file). The header strategy adds "h1" and "h2": the breadcrumb.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import json
import os
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # the episode folder (ep08-chunk-lab)

# tiktoken downloads its encoding file (about 1.7 MB) on first use. Keep it in this folder
# (git-ignored) instead of your temp folder, so the offline tests can find it later.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(HERE / "tiktoken_cache"))

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from llm import DEFAULT_LOCAL_EMBEDDINGS_MODEL  # noqa: E402

HANDBOOK_DIR = HERE / "data" / "handbook"
QUESTIONS_FILE = HERE / "data" / "questions.json"
EMBEDDER = DEFAULT_LOCAL_EMBEDDINGS_MODEL               # sentence-transformers/all-MiniLM-L6-v2

# --- The settings of each strategy (about the same size, so the race is fair) ---
CHUNK_SIZE = 400            # characters (character and recursive), Episode 07's setting
CHUNK_OVERLAP = 60
TOKEN_ENCODING = "cl100k_base"   # TokenTextSplitter's default is "gpt2", so we name it
TOKEN_CHUNK_SIZE = 100      # tiktoken tokens, roughly 400 characters of English
TOKEN_OVERLAP = 15
HEADERS = [("#", "h1"), ("##", "h2")]   # (Markdown marker, the metadata key it becomes)
SECTION_MAX_TOKENS = 200    # MiniLM tokens: a longer section is split again (MiniLM reads at most 256)
SECTION_OVERLAP = 30

STRATEGIES = {
    "character": f"CharacterTextSplitter(separator='\\n\\n', chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})",
    "recursive": f"RecursiveCharacterTextSplitter(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})",
    "tokens": f"TokenTextSplitter(encoding_name='{TOKEN_ENCODING}', chunk_size={TOKEN_CHUNK_SIZE}, "
              f"chunk_overlap={TOKEN_OVERLAP})",
    "headers": f"MarkdownHeaderTextSplitter(h1, h2), then RecursiveCharacterTextSplitter"
               f".from_huggingface_tokenizer(MiniLM, chunk_size={SECTION_MAX_TOKENS}, chunk_overlap={SECTION_OVERLAP})",
}
LABELS = {"character": "character", "recursive": "recursive", "tokens": "tokens", "headers": "headers + recursive"}


# ---------------------------------------------------------------------------
# The handbook and the questions
# ---------------------------------------------------------------------------

def load_handbook(folder=HANDBOOK_DIR):
    """One Document per Markdown file, labelled with its file name."""
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(Path(folder).glob("*.md"))
    ]


def load_questions(path=QUESTIONS_FILE):
    """The ten test questions: question, sources, section, answer_contains."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


# ---------------------------------------------------------------------------
# The embedder's own tokenizer: how MiniLM measures text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def tokenizer():
    """MiniLM's own tokenizer (WordPiece), loaded on its own: counting tokens needs no model weights."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBEDDER)


@lru_cache(maxsize=1)
def max_length():
    """How many tokens MiniLM reads per text: 256. sentence-transformers cuts off everything after that.

    (The tokenizer file on its own says 512. sentence-transformers uses 256 for this model.)
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDER, device="cpu").max_seq_length


def text_limit():
    """Room for YOUR text: the limit minus the 2 marker tokens MiniLM adds ([CLS] and [SEP])."""
    return max_length() - tokenizer().num_special_tokens_to_add()


def count_tokens(text):
    """The length of `text` in MiniLM tokens."""
    return len(tokenizer().tokenize(text))


# ---------------------------------------------------------------------------
# The four splitters
# ---------------------------------------------------------------------------

def character_splitter(size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cuts ONLY at the separator (a blank line), then glues pieces together up to chunk_size."""
    return CharacterTextSplitter(separator="\n\n", chunk_size=size, chunk_overlap=overlap, add_start_index=True)


def recursive_splitter(size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Tries "\\n\\n", then "\\n", then " ", then "" - a smaller separator only for pieces still too big."""
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)


def token_splitter():
    """Cuts every 100 tiktoken tokens, wherever that lands (even in the middle of a word)."""
    return TokenTextSplitter(encoding_name=TOKEN_ENCODING, chunk_size=TOKEN_CHUNK_SIZE,
                             chunk_overlap=TOKEN_OVERLAP, add_start_index=True)


def header_splitter():
    """Cuts at # and ## headings and puts the heading text into metadata["h1"] and ["h2"]."""
    # strip_headers=False keeps the heading line in the chunk text too (the default, True, removes it).
    return MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)


def section_splitter():
    """The recursive splitter, but chunk_size is counted in MiniLM tokens instead of characters."""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer(), chunk_size=SECTION_MAX_TOKENS, chunk_overlap=SECTION_OVERLAP,
    )


def split_by_headers(docs):
    """Stage 1: every file becomes sections. Each section keeps the file name and gets its headings."""
    sections = []
    for doc in docs:
        for section in header_splitter().split_text(doc.page_content):
            section.metadata = {"source": doc.metadata["source"], **section.metadata}
            sections.append(section)
    return sections


def split_headers_then_recursive(docs):
    """Stage 2: sections longer than SECTION_MAX_TOKENS are split again. Metadata is copied to every part."""
    chunks = []
    splitter = section_splitter()
    for section in split_by_headers(docs):
        parts = splitter.split_documents([section])
        for number, part in enumerate(parts, start=1):
            if len(parts) > 1:
                part.metadata["part"] = f"{number} of {len(parts)}"
            chunks.append(part)
    return chunks


def hide_size_warnings():
    """Step 2 explains the "Created a chunk of size ..." warnings. The other commands hide them."""
    import logging

    logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)


def number_chunks(chunks):
    """Give every chunk its number inside its file: refunds.md#1, refunds.md#2, ..."""
    counts = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counts[source] = counts.get(source, 0) + 1
        chunk.metadata["chunk"] = counts[source]
    return chunks


def make_chunks(strategy, docs=None):
    """Split the handbook (or `docs`) with one of the four strategies."""
    docs = load_handbook() if docs is None else docs
    if strategy == "character":
        chunks = character_splitter().split_documents(docs)
    elif strategy == "recursive":
        chunks = recursive_splitter().split_documents(docs)
    elif strategy == "tokens":
        chunks = token_splitter().split_documents(docs)
    elif strategy == "headers":
        chunks = split_headers_then_recursive(docs)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Use one of: {', '.join(STRATEGIES)}")
    return number_chunks(chunks)


# ---------------------------------------------------------------------------
# Labels for printing and citing
# ---------------------------------------------------------------------------

def tag(doc):
    """Short name of a chunk: file + chunk number, e.g. refunds.md#4."""
    return f"{doc.metadata['source']}#{doc.metadata['chunk']}"


def breadcrumb(doc):
    """The heading path of a chunk, e.g. 'Dabbaloop Refunds and Credits Policy > Missed or late meal'."""
    return " > ".join(doc.metadata[key] for _, key in HEADERS if key in doc.metadata)


def citation(doc):
    """What the bot cites: file > section (+ part), e.g. 'refunds.md > Missed or late meal'."""
    section = doc.metadata.get("h2") or doc.metadata.get("h1") or f"chunk {doc.metadata['chunk']}"
    label = f"{doc.metadata['source']} > {section}"
    if "part" in doc.metadata:
        label += f", part {doc.metadata['part'].split(' of ')[0]}"
    return label


def shared_text(first, second, minimum=10):
    """The overlap: the longest end of `first` that is also the start of `second` (at least `minimum` long)."""
    for size in range(min(len(first), len(second)), minimum - 1, -1):
        if first[-size:] == second[:size]:
            return second[:size]
    return ""
