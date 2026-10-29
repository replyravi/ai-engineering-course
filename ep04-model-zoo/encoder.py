"""
encoder.py - an ENCODER foundation model reads your reviews (steps 3 and 4).

all-MiniLM-L6-v2 was pre-trained by other people on a huge amount of text. It has never
seen YOUR reviews, yet it turns each one into 384 numbers (a vector) that capture its
meaning. Reviews that mean similar things get vectors that point the same way.

It writes nothing. It only reads, and turns text into numbers.
So a foundation model does not have to be generative.

Run:  python encoder.py                               (step 3, then a search for the default query)
      python encoder.py "can I pay with my phone?"      (step 4 with your own query)
"""

import re
import sys
import time

import numpy as np

from common import REVIEWS_FILE, console, count_parameters, in_millions, load_reviews, make_table
from llm import get_embeddings

DEFAULT_QUERY = "the snacks were stale"
TOP = 3   # how many reviews to show

# Little words that say nothing about meaning (used only to compare words, step 4).
FILLER = {"a", "an", "the", "and", "or", "is", "are", "was", "were", "it", "i", "my", "me", "they",
          "to", "of", "for", "with", "in", "on", "at", "so", "can", "do", "did", "be", "when", "this",
          "that", "there", "here", "too", "very"}


# ---------------------------------------------------------------------------
# Step 3 - embed every review and count the parameters
# ---------------------------------------------------------------------------

def embed(texts: list[str], embeddings) -> np.ndarray:
    """One vector per text: a table with one row per text and 384 columns."""
    return np.array(embeddings.embed_documents(texts))


def parameters_of(embeddings) -> int | None:
    """Parameters of the local model inside get_embeddings(), or None for a cloud model."""
    model = getattr(embeddings, "_client", None)   # the sentence-transformers model inside the LangChain wrapper
    return count_parameters(model) if model is not None else None


# ---------------------------------------------------------------------------
# Step 4 - rank the reviews by cosine similarity to a query
# ---------------------------------------------------------------------------

def cosine_similarity(a, b) -> float:
    """How closely two vectors point the same way: 1.0 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank(query_vector, vectors, texts: list[str], top: int = TOP) -> list[tuple[float, int, str]]:
    """The `top` texts closest to the query, best first, as (score, review number, text)."""
    scored = [(cosine_similarity(query_vector, v), number, text)
              for number, (v, text) in enumerate(zip(vectors, texts), start=1)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top]


def shared_words(a: str, b: str) -> set[str]:
    """Meaningful words that appear in both texts (filler words removed)."""
    def words(text):
        return set(re.findall(r"[a-z]+", text.lower())) - FILLER
    return words(a) & words(b)


def main(argv=None, embeddings=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    query = " ".join(argv).strip() or DEFAULT_QUERY

    reviews = load_reviews()
    print(f"Read {len(reviews)} reviews from data/{REVIEWS_FILE.name}")

    # Step 3: the encoder turns each review into a vector
    start = time.perf_counter()
    if embeddings is None:
        embeddings = get_embeddings()          # local all-MiniLM-L6-v2 (no key, no cost)
    vectors = embed(reviews, embeddings)
    seconds = time.perf_counter() - start

    name = getattr(embeddings, "model_name", None) or type(embeddings).__name__
    print(f"Encoder: {name}")
    print(f"{vectors.shape[0]} vectors, {vectors.shape[1]} numbers each ({seconds:.1f} s, including loading the model)")
    print("Review 1 starts with:", ", ".join(f"{x:.3f}" for x in vectors[0][:5]), "...")
    params = parameters_of(embeddings)
    if params:
        print(f"Parameters: {params:,} (about {in_millions(params)})")
    else:
        print("Parameters: not published (a cloud model)")

    # Step 4: embed the query the same way, then find the closest reviews
    query_vector = np.array(embeddings.embed_query(query))
    results = rank(query_vector, vectors, reviews)
    rows = [[f"{score:.2f}", f"#{number}", text] for score, number, text in results]
    console.print()
    console.print(make_table(["score", "review", "text"], rows, title=f'Closest reviews to "{query}"'))

    common = shared_words(query, results[0][2])
    print("Words shared by the query and the top review:",
          ", ".join(sorted(common)) if common else "none, it matched on meaning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
