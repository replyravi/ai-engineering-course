"""
positions.py - give every position in a sentence its own pattern of numbers (step 7).

Attention on its own sees a sentence as a BAG of word vectors: shuffle the words and the bag
is the same. So we add a position pattern to each word vector. This file writes the sine and
cosine patterns from the 2017 Transformer paper, "Attention Is All You Need":

    PE[pos, 2i]   = sin(pos / 10000 ** (2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000 ** (2i / d_model))

Each pair of columns is a clock hand turning at its own speed. Together they give every
position a unique reading.

Run:  python positions.py      (saves plots/positions.png)
"""

from pathlib import Path

import numpy as np

# Only NumPy at the top: attention.py (and Episode 03) import this file without loading any model.
PICTURE = Path(__file__).resolve().parent / "plots" / "positions.png"
SEED = 42


def positional_encoding(num_positions: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """The sine/cosine position table, shape (num_positions, d_model). Row p is added to the word at position p."""
    if d_model % 2:
        raise ValueError("d_model must be even: sine and cosine columns come in pairs")
    positions = np.arange(num_positions)[:, np.newaxis]        # column: 0, 1, 2, ...
    pair_starts = np.arange(0, d_model, 2)[np.newaxis, :]       # row: 0, 2, 4, ... (one per clock hand)
    angles = positions / base ** (pair_starts / d_model)        # fast hands first, slow hands last
    table = np.zeros((num_positions, d_model))
    table[:, 0::2] = np.sin(angles)                             # even columns
    table[:, 1::2] = np.cos(angles)                             # odd columns
    return table


def same_bag(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a and b hold the same row vectors, in any order (a 'bag' of vectors)."""
    if a.shape != b.shape:
        return False
    return all(any(np.allclose(row, other) for other in b) for row in a)


def save_picture(table: np.ndarray, path=PICTURE):
    """Heatmap: one column per position, one row per number. Every column is a different barcode."""
    import matplotlib

    matplotlib.use("Agg")   # draw straight into a PNG file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blue_grey_red = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#a32f2f"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    image = ax.imshow(table.T, aspect="auto", cmap=blue_grey_red, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"Sine and cosine positions: {table.shape[0]} positions x {table.shape[1]} numbers",
                 loc="left", fontsize=13, color="#0b0b0b")
    ax.set_xlabel("position in the sentence", color="#52514e")
    ax.set_ylabel("which of the 64 numbers", color="#52514e")
    ax.tick_params(colors="#52514e", labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.ax.tick_params(colors="#52514e", labelsize=9)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main(picture=PICTURE) -> int:
    from encoder import read_sentences, save_note

    table = positional_encoding(50, 64)
    print(f"Position table: {table.shape} (50 positions x 64 numbers)\n")
    for pos in [0, 1, 2, 49]:
        print(f"  position {pos:>2}, first six numbers: {np.round(table[pos, :6], 3).tolist()}")

    unique_rows = len(np.unique(table.round(6), axis=0))
    print(f"\nall rows unique: {unique_rows == len(table)}  ({unique_rows} different rows)")

    similarity = table @ table[10]          # dot product of every position with position 10
    similarity[10] = -np.inf                # skip position 10 itself
    closest = np.argsort(similarity)[::-1][:2]
    print(f"positions most like position 10: {sorted(closest.tolist())} (neighbours look alike)")

    # Same words, different order: a bag of word vectors cannot tell them apart.
    first, second = read_sentences()["order"]
    words = sorted(set(first.lower().split()))
    rng = np.random.default_rng(SEED)
    vectors = {word: rng.normal(size=64) for word in words}   # pretend word vectors
    a = np.array([vectors[w] for w in first.lower().split()])
    b = np.array([vectors[w] for w in second.lower().split()])
    print(f"\n'{first}' vs '{second}'")
    print(f"  same bag of vectors? words only:       {same_bag(a, b)}")
    print(f"  same bag of vectors? words + positions: {same_bag(a + table[:3], b + table[:3])}")

    path = save_picture(table, picture)
    print(f"\nSaved the picture: {save_note(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
