"""
step05_embed_three.py - Step 5: embeddings are for SEARCH.

We turn three sentences into vectors with the free local model and compare them.
Two of them mean the same thing with almost no words in common. Watch their score.

    python step05_embed_three.py

No API key needed. The first run downloads the model (about 90 MB) once.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import math

from llm import get_embeddings

SENTENCES = [
    "My lunch box never arrived today.",
    "The rider did not deliver my midday meal.",
    "Fridges must stay at 5 degrees C or colder.",
]


def cosine_similarity(a, b):
    """1.0 = pointing the same way (same meaning), around 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(SENTENCES)
    print(f"dim {len(vectors[0])}  (every sentence becomes {len(vectors[0])} numbers)")
    print(f"first 5 numbers of sentence 1: {[round(x, 4) for x in vectors[0][:5]]}\n")

    pairs = [(0, 1), (0, 2), (1, 2)]
    scored = sorted(pairs, key=lambda p: cosine_similarity(vectors[p[0]], vectors[p[1]]), reverse=True)
    for i, j in scored:
        score = cosine_similarity(vectors[i], vectors[j])
        print(f"{score:.3f}  \"{SENTENCES[i]}\"  vs  \"{SENTENCES[j]}\"")


if __name__ == "__main__":
    main()
