"""
meaning_map.py - words with similar meanings get similar vectors (step 6).

get_embeddings() from llm.py is the free local model all-MiniLM-L6-v2 (a small encoder, like
DistilBERT's little cousin). It turns each word into 384 numbers. We then:
  1. find each word's nearest neighbour by cosine similarity (1.0 = same direction)
  2. squeeze 384 numbers down to 2 with PCA, and draw the map
  3. drop in a new word and find its three nearest neighbours

Run:  python meaning_map.py      (saves plots/meaning_map.png; the model is about 90 MB, cached since Episode 01)
"""

import numpy as np

from encoder import PLOTS, SEED, save_note

PICTURE = PLOTS / "meaning_map.png"

GROUPS = {
    "food": ["dosa", "biryani", "samosa", "idli", "paneer"],
    "vehicle": ["bus", "train", "bicycle", "scooter", "rickshaw"],
    "feeling": ["happy", "angry", "sad", "excited", "nervous"],
}
QUERY = "pakora"   # a new word that is not on the map yet

# Colours and shapes for the three groups (colour-blind safe; the shapes help too)
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
SHAPES = ["o", "s", "^"]


def embed(words, embeddings=None) -> np.ndarray:
    """One row of numbers per word, each row scaled to length 1 (so a dot product = cosine similarity)."""
    if embeddings is None:
        from llm import get_embeddings
        embeddings = get_embeddings()
    vectors = np.array(embeddings.embed_documents(list(words)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def nearest(vector, vectors, words, k=3, skip=None) -> list:
    """The k words whose vectors point most in the same direction as `vector`: [(word, similarity), ...]."""
    similarities = vectors @ vector
    order = [i for i in np.argsort(similarities)[::-1] if words[i] != skip]
    return [(words[i], float(similarities[i])) for i in order[:k]]


def save_picture(points, words, groups, query_point, neighbours, path=PICTURE):
    import matplotlib

    matplotlib.use("Agg")   # draw straight into a PNG file, no window needed
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for i, name in enumerate(GROUPS):
        mine = [j for j, g in enumerate(groups) if g == name]
        ax.scatter(points[mine, 0], points[mine, 1], s=80, c=COLOURS[i], marker=SHAPES[i],
                   edgecolors="#fcfcfb", linewidths=1.5, label=name, zorder=3)
    for (x, y), word in zip(points, words):
        ax.annotate(word, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10, color="#0b0b0b")

    # the new word, with lines to its three nearest neighbours
    for word, _ in neighbours:
        x, y = points[words.index(word)]
        ax.plot([query_point[0], x], [query_point[1], y], color="#898781", linewidth=1, linestyle="--", zorder=2)
    ax.scatter([query_point[0]], [query_point[1]], s=170, c="#0b0b0b", marker="*", label=f"new word: {QUERY}", zorder=4)
    ax.annotate(QUERY, query_point, xytext=(-10, 10), textcoords="offset points", ha="right", fontsize=11,
                fontweight="bold", color="#0b0b0b")

    ax.set_title("15 words in meaning space (384 numbers squeezed to 2 with PCA)", loc="left", fontsize=13, color="#0b0b0b")
    ax.set_xlabel("PCA direction 1", color="#52514e")
    ax.set_ylabel("PCA direction 2", color="#52514e")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(color="#e1e0d9", linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, labelcolor="#0b0b0b", loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main(embeddings=None, picture=PICTURE) -> int:
    from sklearn.decomposition import PCA

    words = [w for group in GROUPS.values() for w in group]
    groups = [name for name, group in GROUPS.items() for _ in group]
    vectors = embed(words + [QUERY], embeddings)
    vectors, query = vectors[:-1], vectors[-1]
    print(f"Embedded {len(words)} words (+1 new word), {vectors.shape[1]} numbers each\n")

    print("Nearest neighbour of each word (cosine similarity):")
    same_group = 0
    for word, group in zip(words, groups):
        (other, similarity), = nearest(vectors[words.index(word)], vectors, words, k=1, skip=word)
        same = groups[words.index(other)] == group
        same_group += same
        print(f"  {word:<9} -> {other:<9} {similarity:.2f}   {'same group' if same else 'OTHER group'}")
    print(f"Nearest neighbour in the same group: {same_group} of {len(words)}\n")

    neighbours = nearest(query, vectors, words, k=3)
    print(f"New word {QUERY!r}, three nearest: " + ", ".join(f"{w} {s:.2f}" for w, s in neighbours))

    pca = PCA(n_components=2, random_state=SEED).fit(vectors)    # learn the 2 best directions from the 15 words
    points, query_point = pca.transform(vectors), pca.transform(query[np.newaxis])[0]
    print(f"The 2D map keeps {pca.explained_variance_ratio_.sum():.0%} of the spread of the 384 numbers.")

    path = save_picture(points, words, groups, query_point, neighbours, picture)
    print(f"\nSaved the picture: {save_note(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
