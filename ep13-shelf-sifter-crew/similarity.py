"""
similarity.py - Step 9: find book titles that MEAN something close to what the reader wants.

Same idea as Episodes 07 and 08: an embeddings model turns each text into 384 numbers,
and texts with similar meaning get similar numbers. Everything runs on YOUR computer, for free.

The model is all-MiniLM-L6-v2 again (Apache-2.0 licence). The CrewAI environment has no
PyTorch, so we run the model's ONNX copy through Chroma, which CrewAI already installs.
Same model, same numbers as the sentence-transformers version of Episodes 07 and 08.
The first run downloads it once (about 83 MB) into ~/.cache/chroma/ and then works offline.

    python similarity.py "a long journey by train"

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import sys
from pathlib import Path

import numpy as np

# Where Chroma keeps the downloaded model (the tests check this folder before using it).
MODEL_FOLDER = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx"


def model_is_downloaded() -> bool:
    """True when the local model is already on this computer (so no download is needed)."""
    return (MODEL_FOLDER / "model.onnx").is_file()


def load_embedder():
    """Return the local embeddings model: a function that turns a list of texts into vectors."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2   # imported here: it loads slowly

    return ONNXMiniLM_L6_V2()


def top_matches(query: str, books: list[dict], embed, k: int = 3) -> list[dict]:
    """The k books whose titles are closest in meaning to `query`, best first.

    score = cosine similarity between the query and the title: 1.0 means the same meaning,
    around 0 means unrelated. Compare scores with each other, not with a fixed number.
    """
    vectors = np.array(embed([query] + [book["title"] for book in books]), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)   # length 1, so a dot product = cosine
    scores = vectors[1:] @ vectors[0]
    best_first = np.argsort(-scores, kind="stable")[:k]
    return [
        {"title": books[i]["title"], "score": round(float(scores[i]), 3),
         "price": books[i]["price"], "rating": books[i]["rating"], "url": books[i]["url"]}
        for i in best_first
    ]


if __name__ == "__main__":
    from scraper import FIXTURES, category_url, parse_books

    query = " ".join(sys.argv[1:]) or "a long journey by train"
    saved = FIXTURES / "travel.html"
    if not saved.exists():
        raise SystemExit("No saved page yet. Run Step 3 first:  python save_fixture.py")
    books = parse_books(saved.read_text(encoding="utf-8"), category_url("travel"))
    print(f"Query: {query!r}   ({len(books)} titles on the saved Travel shelf)\n")
    for match in top_matches(query, books, load_embedder()):
        print(f"  {match['score']:.3f}  {match['title']}  (£{match['price']:.2f}, rating {match['rating']}/5)")
