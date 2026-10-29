"""
cross_attention.py - watch a decoder look back at the encoder while it translates (step 11).

T5-small (an encoder-decoder) translates one English line into German. Inside every decoder layer,
CROSS-attention lets each German token being written look at the English tokens:
    queries          come from the decoder (the German written so far)
    keys and values  come from the encoder (the English it read)
We ask the model for those weights (output_attentions=True), average them, and draw a heatmap.

Run:  python cross_attention.py                    (saves plots/cross_attention.png)
Try:  python cross_attention.py "The tea is hot."   or   python cross_attention.py --layer 5
"""

import argparse

import numpy as np

from decoder import PLOTS, read_prompts, save_note
from three_families import load_t5

PICTURE = PLOTS / "cross_attention.png"
PREFIX = "translate English to German: "      # T5 knows its jobs by a text prefix


def translate(tokenizer, model, sentence: str):
    """Returns (input IDs, output IDs). The output starts with T5's start token, <pad>, and ends with </s>."""
    import torch

    inputs = tokenizer(PREFIX + sentence, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)   # greedy
    return inputs["input_ids"][0].tolist(), out[0].tolist()


def attention_maps(model, input_ids: list[int], output_ids: list[int]):
    """Run the model once on the finished translation and return (cross, decoder_self) weights.

    cross:        (layers, heads, output tokens, input tokens)
    decoder_self: (layers, heads, output tokens, output tokens)
    Row i is the decoder at the moment it chose output token i + 1.
    """
    import torch

    with torch.inference_mode():
        out = model(input_ids=torch.tensor([input_ids]), decoder_input_ids=torch.tensor([output_ids[:-1]]),
                    output_attentions=True)
    if not out.cross_attentions:
        raise RuntimeError("No attention weights came back. Load T5 with attn_implementation='eager'.")
    cross = torch.stack(out.cross_attentions)[:, 0].numpy()          # drop the batch axis
    decoder_self = torch.stack(out.decoder_attentions)[:, 0].numpy()
    return cross, decoder_self


def strongest_links(grid: np.ndarray, out_tokens: list[str], in_tokens: list[str]) -> list[tuple[str, str, float]]:
    """For every output token: the input token it looks at most, and how much."""
    return [(o, in_tokens[int(np.argmax(row))], float(row.max())) for o, row in zip(out_tokens, grid)]


def save_picture(grid, in_tokens, out_tokens, title: str, path=PICTURE):
    import matplotlib

    matplotlib.use("Agg")   # draw straight into a PNG file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    image = ax.imshow(grid, cmap=blues, vmin=0, vmax=float(grid.max()))
    ax.set_xticks(range(len(in_tokens)), in_tokens, rotation=60, ha="right", fontsize=10, color="#0b0b0b")
    ax.set_yticks(range(len(out_tokens)), out_tokens, fontsize=10, color="#0b0b0b")
    ax.set_xlabel("English input tokens (keys and values, from the encoder)", color="#52514e")
    ax.set_ylabel("German token being written (queries, from the decoder)", color="#52514e")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for row, values in enumerate(grid):
        col = int(np.argmax(values))   # label the strongest cell of every row
        ax.text(col, row, f"{values[col]:.2f}", ha="center", va="center", fontsize=8,
                color="#fcfcfb" if values[col] > 0.6 * grid.max() else "#0b0b0b")
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(colors="#52514e", labelsize=9)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv=None, picture=PICTURE) -> int:
    parser = argparse.ArgumentParser(description="Episode 03, step 11: a cross-attention heatmap")
    parser.add_argument("sentence", nargs="?", default=read_prompts()["translate"][0])
    parser.add_argument("--layer", type=int, default=None, help="one decoder layer (1-6) instead of the average")
    args = parser.parse_args(argv)

    tokenizer, model = load_t5()
    input_ids, output_ids = translate(tokenizer, model, args.sentence)
    in_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    out_tokens = tokenizer.convert_ids_to_tokens(output_ids[1:])        # skip the start token
    cross, decoder_self = attention_maps(model, input_ids, output_ids)
    layers, heads, n_out, n_in = cross.shape

    print(f"English: {args.sentence}")
    print(f"German (T5-small, greedy): {tokenizer.decode(output_ids, skip_special_tokens=True)}\n")
    print(f"Input tokens ({n_in}, the encoder side: keys and values):  {' '.join(in_tokens)}")
    print(f"Output tokens ({n_out}, the decoder side: queries):  {' '.join(out_tokens)}")
    print("('▁' marks the start of a word in T5's vocabulary.)\n")
    print(f"Cross-attention weights: {cross.shape} = {layers} decoder layers x {heads} heads, "
          f"each a {n_out} x {n_in} grid")
    print(f"Every row sums to 1: {bool(np.allclose(cross.sum(axis=-1), 1, atol=1e-5))}")
    above = ~np.tril(np.ones((n_out, n_out), dtype=bool))
    print(f"The decoder's own self-attention {decoder_self.shape} is causal "
          f"(zero above the diagonal): {bool((decoder_self[..., above] < 1e-6).all())}\n")

    if args.layer:
        grid = cross[args.layer - 1].mean(axis=0)
        which = f"decoder layer {args.layer}, average of its {heads} heads"
    else:
        grid = cross.mean(axis=(0, 1))
        which = f"average of all {layers} layers x {heads} heads"
    print(f"Where each German token looks most ({which}):")
    for out_token, in_token, weight in strongest_links(grid, out_tokens, in_tokens):
        print(f"  {out_token:<14}-> {in_token:<12}{weight:.2f}")

    n_prefix = len(tokenizer(PREFIX.strip(), add_special_tokens=False)["input_ids"])
    print(f"\nShare of attention on the {n_prefix} task-prefix tokens ('{PREFIX.strip()}'): "
          f"{grid[:, :n_prefix].sum(axis=-1).mean():.0%}")
    focus = cross.mean(axis=1).max(axis=-1).mean(axis=-1)               # per layer: average biggest weight per row
    sharpest = int(np.argmax(focus)) + 1
    print("How focused each layer is (average biggest weight per row): "
          + ", ".join(f"layer {i + 1} {f:.2f}" for i, f in enumerate(focus)))
    print(f"Sharpest single layer: layer {sharpest} (try --layer {sharpest})")

    title = f"T5-small cross-attention: {which}\n(each row sums to 1; the strongest cell of every row is labelled)"
    path = save_picture(grid, in_tokens, out_tokens, title, picture)
    print(f"\nSaved the picture: {save_note(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
