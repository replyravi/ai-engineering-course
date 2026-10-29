"""
tokens.py - how a Transformer turns text into token IDs (steps 1 to 4).

    python tokens.py pieces     step 1: split three Episode 01 feedback lines into tokens
    python tokens.py ids        step 2: token IDs, [CLS], [SEP] and the other special tokens
    python tokens.py compare    step 3: same sentence, English vs Hindi, three tokenizers
    python tokens.py pad        step 4: pad two sentences into one batch, with an attention mask

The first run downloads the DistilBERT tokenizer (and, in step 3, two small tiktoken files).
"""

import argparse

from encoder import load_tokenizer, read_sentences

# tiktoken encodings used by OpenAI's GPT models (step 3)
GPT_ENCODINGS = {"cl100k_base": "gpt-4", "o200k_base": "gpt-4o"}


# ---------------------------------------------------------------------------
# Step 1 - text -> tokens (subword pieces)
# ---------------------------------------------------------------------------

def split_words(text: str, tokenizer) -> list:
    """Return [(word, [its pieces]), ...] for every word the tokenizer cut into pieces."""
    encoding = tokenizer(text, add_special_tokens=False)
    pieces = encoding.tokens()          # the token strings
    word_of_piece = encoding.word_ids()  # which word each piece came from: 0, 1, 1, 1, 2, ...
    words = {}
    for piece, word in zip(pieces, word_of_piece):
        words.setdefault(word, []).append(piece)
    return [("".join(p.removeprefix("##") for p in group), group) for group in words.values() if len(group) > 1]


def show_pieces(tokenizer) -> None:
    for number, line in enumerate(read_sentences()["feedback"], start=1):
        pieces = tokenizer.tokenize(line)
        print(f"Sentence {number}: {line}")
        print(f"  {len(pieces)} tokens: {' | '.join(pieces)}")
        splits = [f"{word} = {' + '.join(group)}" for word, group in split_words(line, tokenizer)]
        print(f"  cut into pieces: {', '.join(splits) if splits else 'none'}\n")
    print("'##' means: this piece glues onto the piece before it, with no space.")


# ---------------------------------------------------------------------------
# Step 2 - tokens -> IDs, and the special tokens
# ---------------------------------------------------------------------------

def show_ids(tokenizer) -> None:
    line = read_sentences()["feedback"][1]
    ids = tokenizer(line)["input_ids"]                  # adds [CLS] and [SEP] by itself
    print(f"Sentence: {line}\n")
    print(f"  {'token':<10} {'ID':>6}")
    for token, token_id in zip(tokenizer.convert_ids_to_tokens(ids), ids):
        print(f"  {token:<10} {token_id:>6}")

    print(f"\nVocabulary size: {len(tokenizer):,} entries (IDs 0 to {len(tokenizer) - 1:,})")
    print(f"An ID is just a row number: entry {ids[1]} of the vocabulary is {tokenizer.convert_ids_to_tokens(ids[1])!r}")

    print("\nSpecial tokens:")
    for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        print(f"  {token:<7} ID {tokenizer.convert_tokens_to_ids(token)}")

    tricky = "🍛 biryani"
    print(f"\nA character the vocabulary has never seen: {tricky!r} -> {tokenizer.tokenize(tricky)}")


# ---------------------------------------------------------------------------
# Step 3 - token counts: English vs Hindi, three tokenizers
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer) -> dict:
    """Tokens for one text under WordPiece and two GPT tokenizers (special tokens not counted)."""
    import tiktoken

    counts = {"WordPiece (DistilBERT, BERT)": len(tokenizer(text, add_special_tokens=False)["input_ids"])}
    for name, model in GPT_ENCODINGS.items():
        counts[f"{name} ({model})"] = len(tiktoken.get_encoding(name).encode(text))
    return counts


def show_compare(tokenizer) -> None:
    import tiktoken

    english, hindi = read_sentences()["english_hindi"]
    print(f"English: {english}\nHindi:   {hindi}\n")

    for name, model in GPT_ENCODINGS.items():   # proof that the names match the models
        print(f"tiktoken says {model} uses {tiktoken.encoding_for_model(model).name}")

    english_counts, hindi_counts = count_tokens(english, tokenizer), count_tokens(hindi, tokenizer)
    print(f"\n  {'tokenizer':<30} {'English':>7} {'Hindi':>6}")
    for name in english_counts:
        print(f"  {name:<30} {english_counts[name]:>7} {hindi_counts[name]:>6}")

    ids = tokenizer(hindi, add_special_tokens=False)["input_ids"]
    print(f"\nWordPiece pieces for the Hindi line: {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"WordPiece turns them back into text: {tokenizer.decode(ids)}")
    o200k = tiktoken.get_encoding("o200k_base")
    print(f"o200k_base pieces for the Hindi line: {[o200k.decode([i]) for i in o200k.encode(hindi)]}")


# ---------------------------------------------------------------------------
# Step 4 - padding and the attention mask
# ---------------------------------------------------------------------------

def show_pad(tokenizer) -> None:
    long_line, short_line = read_sentences()["short"]
    batch = tokenizer([long_line, short_line], padding=True)   # pad the shorter one up to the longer one
    width = 8
    for row, (ids, mask) in enumerate(zip(batch["input_ids"], batch["attention_mask"]), start=1):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"row {row} tokens: " + "".join(f"{t:<{width}}" for t in tokens).rstrip())
        print(f"row {row} IDs:    " + "".join(f"{i:<{width}}" for i in ids).rstrip())
        print(f"row {row} mask:   " + "".join(f"{m:<{width}}" for m in mask).rstrip() + "\n")
    lengths = {len(ids) for ids in batch["input_ids"]}
    print(f"Both rows now have {lengths.pop()} tokens. Mask 1 = a real token, 0 = [PAD] (ignore it).")


STEPS = {"pieces": show_pieces, "ids": show_ids, "compare": show_compare, "pad": show_pad}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Episode 02, steps 1-4: tokens and token IDs")
    parser.add_argument("step", nargs="?", default="pieces", choices=list(STEPS))
    args = parser.parse_args(argv)
    STEPS[args.step](load_tokenizer())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
