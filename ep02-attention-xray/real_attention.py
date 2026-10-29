"""
real_attention.py - look at the attention weights inside a trained model (step 12).

DistilBERT has 6 layers x 12 heads = 72 attention heads. We read our pronoun sentence, ask the
model for every head's weights (output_attentions=True), and search for the head where "they"
looks most at "students". Then we draw that head as a heatmap.

Be honest about it: we picked the best head out of 72. Most heads do other jobs.

Run:  python real_attention.py      (saves plots/real_attention.png)
Try:  python real_attention.py "The students thanked the cook because they loved his samosas." they students
      (your own sentence, the word that asks, the word it should find; both must be single tokens)
"""

import argparse
from collections import Counter

import numpy as np

from encoder import PLOTS, load_encoder, read_sentences, save_note

PICTURE = PLOTS / "real_attention.png"
FROM_WORD, TO_WORD = "they", "students"


def attention_maps(sentence: str, tokenizer, model):
    """Return (tokens, weights) with weights shaped (layers, heads, tokens, tokens)."""
    import torch

    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True)
    if not outputs.attentions:   # an empty tuple: the model runs the fast "sdpa" attention
        raise RuntimeError("No attention weights came back. Load the model with attn_implementation='eager'.")
    weights = torch.stack(outputs.attentions)[:, 0].numpy()   # one sentence, so drop the batch axis
    return tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), weights


def find_head(weights, tokens, from_word, to_word):
    """The (layer, head) where from_word gives the most weight to to_word. Numbers start at 1, like the plot."""
    i, j = tokens.index(from_word), tokens.index(to_word)
    layer, head = np.unravel_index(np.argmax(weights[:, :, i, j]), weights.shape[:2])
    return int(layer) + 1, int(head) + 1, float(weights[layer, head, i, j])


def top_attended(weights, tokens, from_word, layer, head, k=3):
    """The k tokens that from_word looks at most in one head: [(token, weight), ...]."""
    row = weights[layer - 1, head - 1, tokens.index(from_word)]
    return [(tokens[t], float(row[t])) for t in np.argsort(row)[::-1][:k]]


def favourite_targets(weights, tokens, from_word) -> Counter:
    """For every head: which token does from_word look at most? Counted over all heads."""
    row_index = tokens.index(from_word)
    picks = weights[:, :, row_index].argmax(axis=-1).ravel()
    return Counter(tokens[p] for p in picks)


def save_picture(grid, tokens, layer, head, highlight, path=PICTURE):
    import matplotlib

    matplotlib.use("Agg")   # draw straight into a PNG file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    blues = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    image = ax.imshow(grid, cmap=blues, vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)), tokens, rotation=60, ha="right", fontsize=10, color="#0b0b0b")
    ax.set_yticks(range(len(tokens)), tokens, fontsize=10, color="#0b0b0b")
    ax.set_xlabel("looks at (keys)", color="#52514e")
    ax.set_ylabel("token asking (queries)", color="#52514e")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    row = tokens.index(highlight)
    ax.add_patch(Rectangle((-0.5, row - 0.5), len(tokens), 1, fill=False, edgecolor="#eb6834", linewidth=2))
    for col, value in enumerate(grid[row]):
        if value >= 0.05:   # label only the cells that matter in the highlighted row
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="#fcfcfb" if value > 0.5 else "#0b0b0b")

    ax.set_title(f"DistilBERT layer {layer}, head {head}: where each token looks\n"
                 f"(every row sums to 1; the '{highlight}' row is outlined)", loc="left", fontsize=12, color="#0b0b0b")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(colors="#52514e", labelsize=9)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv=None, picture=PICTURE) -> int:
    parser = argparse.ArgumentParser(description="Episode 02, step 12: real attention heads")
    parser.add_argument("sentence", nargs="?", default=read_sentences()["pronoun"][0])
    parser.add_argument("from_word", nargs="?", default=FROM_WORD, help="the token that asks, e.g. they")
    parser.add_argument("to_word", nargs="?", default=TO_WORD, help="the token it should find, e.g. students")
    args = parser.parse_args(argv)
    from_word, to_word = args.from_word, args.to_word

    tokenizer, model = load_encoder()
    tokens, weights = attention_maps(args.sentence, tokenizer, model)
    layers, heads, n, _ = weights.shape
    print(f"Sentence: {args.sentence}")
    print(f"Tokens ({n}): {' '.join(tokens)}")
    for word in (from_word, to_word):
        if word not in tokens:
            print(f"{word!r} is not one of these tokens. Pick two words from the list above.")
            return 1
    print(f"Attention weights: {weights.shape} = {layers} layers x {heads} heads, each a {n} x {n} grid")
    print(f"Every row sums to 1: {bool(np.allclose(weights.sum(axis=-1), 1, atol=1e-5))}\n")

    layer, head, weight = find_head(weights, tokens, from_word, to_word)
    print(f"Searched all {layers * heads} heads for the one where {from_word!r} looks most at {to_word!r}:")
    print(f"  layer {layer}, head {head}: {from_word!r} gives {weight:.2f} of its attention to {to_word!r}")
    top = top_attended(weights, tokens, from_word, layer, head)
    print(f"  top 3 for {from_word!r} in that head: " + ", ".join(f"{t} {w:.2f}" for t, w in top))

    counts = favourite_targets(weights, tokens, from_word)
    print(f"\nHonest check: in how many of the {layers * heads} heads is {to_word!r} the top choice of {from_word!r}? "
          f"{counts[to_word]}")
    print(f"What {from_word!r} looks at most, counted over all {layers * heads} heads:")
    for token, count in counts.most_common(5):
        print(f"  {token:<10} {count:>2} heads")

    path = save_picture(weights[layer - 1, head - 1], tokens, layer, head, from_word, picture)
    print(f"\nSaved the picture: {save_note(path)} (layer {layer}, head {head})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
