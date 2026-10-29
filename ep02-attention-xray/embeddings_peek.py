"""
embeddings_peek.py - look at two tables inside the real encoder (steps 5 and 8).

    python embeddings_peek.py              step 5: the token embedding table (one row per vocabulary entry)
    python embeddings_peek.py positions    step 8: the position table (BERT-style models LEARN positions)

Both tables are plain grids of numbers that were learned during training, like every other weight.
"""

import argparse

import numpy as np

from encoder import MODEL_NAME, load_encoder
from positions import positional_encoding

BERT_BASE = "google-bert/bert-base-uncased"   # step 8 reads only its small config.json, not the weights


def first_five(row) -> list:
    """The first five numbers of a row, rounded for printing."""
    return [round(float(x), 4) for x in row[:5]]


def cell(value) -> str:
    """30522 -> '30,522' for the table; text stays as it is."""
    return f"{value:,}" if isinstance(value, int) else str(value)


def cosine(a, b) -> float:
    """Cosine similarity: 1.0 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Step 5 - the token embedding table
# ---------------------------------------------------------------------------

def show_token_table(tokenizer, model) -> None:
    import torch

    table = model.embeddings.word_embeddings.weight.detach().numpy()   # rows = vocabulary, columns = numbers
    rows, width = table.shape
    total = sum(p.numel() for p in model.parameters())
    print(f"Token embedding table: {table.shape} -> {rows:,} rows (one per vocabulary entry) x {width} numbers")
    print(f"That is {table.size:,} learned numbers, {table.size / total:.0%} of all {total:,} in DistilBERT.\n")

    word = "coffee"
    token_id = tokenizer.convert_tokens_to_ids(word)
    print(f"{word!r} has ID {token_id}, so its embedding is row {token_id}.")
    print(f"First five of its {width} numbers: {first_five(table[token_id])}")

    with torch.inference_mode():   # the layer itself, given the ID, returns exactly that row
        from_layer = model.embeddings.word_embeddings(torch.tensor([token_id]))[0].numpy()
    print(f"The lookup layer returns exactly that row: {np.array_equal(from_layer, table[token_id])}\n")

    print("Cosine similarity between rows (input vectors, before any attention):")
    for other in ["tea", "bus"]:
        similarity = cosine(table[token_id], table[tokenizer.convert_tokens_to_ids(other)])
        print(f"  {word} ~ {other:<4} {similarity:.2f}")


# ---------------------------------------------------------------------------
# Step 8 - the position table: learned, not the sine formula
# ---------------------------------------------------------------------------

def show_position_table(tokenizer, model) -> None:
    from transformers import AutoConfig

    learned = model.embeddings.position_embeddings.weight.detach().numpy()
    positions, width = learned.shape
    print(f"DistilBERT position table: {learned.shape} -> {positions} positions x {width} numbers")
    print(f"config.sinusoidal_pos_embds = {model.config.sinusoidal_pos_embds}\n")

    sine = positional_encoding(positions, width)        # our step 7 formula, same size
    print("Position 1, first five numbers:")
    print(f"  learned by DistilBERT:  {first_five(learned[1])}")
    print(f"  sine formula (step 7):  {first_five(sine[1])}")
    print(f"Same table? {bool(np.allclose(learned, sine, atol=1e-3))} -> BERT-style models learn their positions.")
    print(f"Longest input: {positions} tokens (the tokenizer knows it too: {tokenizer.model_max_length})\n")

    has_segments = hasattr(model.embeddings, "token_type_embeddings")
    print(f"DistilBERT has segment (sentence A / B) embeddings: {has_segments}")

    bert = AutoConfig.from_pretrained(BERT_BASE)        # downloads config.json only (570 bytes), no weights
    print(f"\n  {'':<22} {'BERT-base':>10} {'DistilBERT':>11}")
    distil = model.config
    for label, b, d in [("layers (blocks)", bert.num_hidden_layers, distil.n_layers),
                        ("attention heads", bert.num_attention_heads, distil.n_heads),
                        ("numbers per token", bert.hidden_size, distil.dim),
                        ("positions", bert.max_position_embeddings, distil.max_position_embeddings),
                        ("segment types", bert.type_vocab_size, "none"),
                        ("vocabulary", bert.vocab_size, distil.vocab_size)]:
        print(f"  {label:<22} {cell(b):>10} {cell(d):>11}")
    print(f"\nBERT-base position type: {bert.position_embedding_type} (a learned table, like DistilBERT's)")
    print(f"(model: {MODEL_NAME}; BERT-base numbers from {BERT_BASE}/config.json)")


STEPS = {"tokens": show_token_table, "positions": show_position_table}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Episode 02, steps 5 and 8: tables inside the encoder")
    parser.add_argument("table", nargs="?", default="tokens", choices=list(STEPS))
    args = parser.parse_args(argv)
    tokenizer, model = load_encoder()
    STEPS[args.table](tokenizer, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
