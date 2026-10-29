"""
c_clusters.py - Approach C: unsupervised learning (clustering). The labels stay HIDDEN.

  1. A small pre-trained neural network (all-MiniLM-L6-v2, that's deep learning) turns each
     sentence into 384 numbers, an "embedding". Sentences with similar meaning get similar numbers.
  2. k-means puts the 40 sentences into 3 groups, using only those numbers.
  3. PCA squeezes 384 numbers down to 2, so we can draw the groups as a picture.

Run:  python c_clusters.py
(The first run downloads the model, about 90 MB. After that it works offline.)
"""

import time

import matplotlib

matplotlib.use("Agg")  # draw straight into a PNG file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from common import HERE, SEED, load_feedback  # noqa: E402

PICTURE = HERE / "outputs" / "clusters.png"
GROUPS = 3

# Colours and shapes for groups 1, 2 and 3 (colour-blind safe; the shapes help too).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
SHAPES = ["o", "s", "^"]


def embed(texts, embeddings=None) -> np.ndarray:
    """Turn sentences into rows of numbers (embeddings). Similar meaning, similar numbers.

    Why this exists: k-means cannot read words. It only groups points.
    This function is the deep-learning step: a small neural network
    (all-MiniLM-L6-v2) already knows language. We do not train it here.

    What it does: if embeddings is None, call get_embeddings() from llm.py
    (downloads about 90 MB the first time). Then embed_documents, then
    normalize so every vector has length 1. k-means then compares direction
    (meaning), not size.
    Tests pass embeddings=fake_embeddings() so this never downloads a model.

    Called by: main (below, all 40 sentences), compare.fit_named_groups,
    compare.score_clusters, and compare.four_answers.
    """
    if embeddings is None:
        from llm import get_embeddings   # the free local model from llm.py
        embeddings = get_embeddings()
    vectors = np.array(embeddings.embed_documents(list(texts)))
    return normalize(vectors)   # length 1 each, so k-means compares meaning (direction), not size


def find_groups(vectors: np.ndarray, k: int = GROUPS) -> KMeans:
    """Group the embedding points into k clusters. Labels are not used.

    Why this exists: approach C is unsupervised. The model is not told
    which sentences are praise, complaint, or suggestion. It only sees numbers.

    What it does: KMeans with k=3 and SEED from common.py, so the groups
    are the same on every run. Returns the fitted KMeans object.
    .labels_ are 0, 1, 2. Callers add 1 when they want groups 1, 2, 3.

    Called by: main (below), compare.fit_named_groups, and tests/test_pipeline.py.
    """
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(vectors)


def save_picture(vectors: np.ndarray, groups: np.ndarray, path=PICTURE):
    """Draw the groups as a PNG. 384 numbers cannot be plotted, so PCA keeps 2.

    Why this exists: a person needs a picture to see that similar comments
    landed near each other. The picture is not used by compare.py.

    What it does: PCA to 2 axes, one scatter series per group, save
    outputs/clusters.png. Returns that path.

    Called by: main (below). Tests call main with a temp path.
    """
    points = PCA(n_components=2, random_state=SEED).fit_transform(vectors)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for g in sorted(set(groups)):
        mine = points[groups == g]
        ax.scatter(mine[:, 0], mine[:, 1], s=70, c=COLOURS[g - 1], marker=SHAPES[g - 1],
                   edgecolors="#fcfcfb", linewidths=1.5, label=f"group {g} ({len(mine)})", zorder=3)
        centre = mine.mean(axis=0)
        ax.annotate(f"group {g}", centre, xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold", color="#0b0b0b", zorder=4,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fcfcfb", "edgecolor": "none", "alpha": 0.85})

    ax.set_title("40 feedback sentences, grouped by k-means (labels hidden)", loc="left",
                 fontsize=13, color="#0b0b0b")
    ax.set_xlabel("PCA direction 1", color="#52514e")
    ax.set_ylabel("PCA direction 2", color="#52514e")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(color="#e4e3df", linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main(embeddings=None, picture=PICTURE) -> pd.DataFrame:
    """Embed all 40 comments, group them, print a peek at the hidden labels, save the PNG.

    Why this exists: this is the script entry for `python c_clusters.py`.
    compare.py does not call main. It calls embed and find_groups on the
    training rows only, so its group numbers will not match this picture.

    What it does: load_feedback(), embed(), find_groups(), print 5 examples
    per group, then a crosstab of group vs human label (k-means never saw
    those labels), then save_picture().

    Called by: the bottom of this file, and tests/test_pipeline.py
    (with fake embeddings and a temp picture path).
    """
    df = load_feedback()

    start = time.perf_counter()
    vectors = embed(df["text"], embeddings)
    seconds = time.perf_counter() - start
    print(f"Embedded {len(df)} sentences, {vectors.shape[1]} numbers each, in {seconds:.1f} s")

    df["group"] = find_groups(vectors).labels_ + 1   # groups 1, 2, 3 instead of 0, 1, 2

    for g in range(1, GROUPS + 1):
        mine = df[df["group"] == g]
        print(f"\nGroup {g} ({len(mine)} sentences). What do these have in common?")
        for text in mine["text"].head(5):
            print(f"  - {text}")

    print("\nNow peek at the hidden labels (k-means never saw them):")
    print(pd.crosstab(df["group"], df["label"]).to_string())

    path = save_picture(vectors, df["group"].to_numpy(), picture)
    print(f"\nSaved the picture: {path.relative_to(HERE) if path.is_relative_to(HERE) else path}")
    return df


if __name__ == "__main__":
    main()
