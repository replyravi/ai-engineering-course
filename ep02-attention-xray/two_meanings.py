"""
two_meanings.py - the same word, two meanings, measured (step 13).

"bank" has ONE token ID, so it starts as ONE row of the embedding table in both sentences.
After the encoder's 6 blocks of attention, each "bank" has its own contextual vector, coloured
by the words around it. Cosine similarity (1.00 = same direction) shows how far they moved.

Run:  python two_meanings.py
"""

import numpy as np

from encoder import load_encoder, read_sentences

WORD = "bank"


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def word_vectors(sentence: str, word: str, tokenizer, model) -> dict:
    """The word's vector at every stage: its table row, then the output of each layer."""
    import torch

    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    position = tokens.index(word)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    token_id = int(inputs["input_ids"][0, position])
    return {
        "id": token_id,
        "position": position,
        "table_row": model.embeddings.word_embeddings.weight[token_id].detach().numpy(),   # before anything
        "layers": [h[0, position].numpy() for h in outputs.hidden_states],   # 0 = embeddings + position, 1-6 = blocks
    }


def compare(first: dict, second: dict) -> dict:
    """Cosine similarity of the two words at every stage."""
    return {
        "table_row": cosine(first["table_row"], second["table_row"]),
        "layers": [cosine(a, b) for a, b in zip(first["layers"], second["layers"])],
    }


def main() -> int:
    tokenizer, model = load_encoder()
    river, money, river_2, money_2 = read_sentences()["two_meanings"]
    vectors = {s: word_vectors(s, WORD, tokenizer, model) for s in (river, money, river_2, money_2)}

    a, b = vectors[river], vectors[money]
    print(f"A: {river}\nB: {money}\n")
    print(f"Token ID of {WORD!r}: {a['id']} in A, {b['id']} in B. Position: {a['position']} in A, {b['position']} in B.\n")

    result = compare(a, b)
    stages = [(f"input embedding (table row {a['id']})", result["table_row"]),
              ("+ position, after the embedding layer", result["layers"][0])]
    stages += [(f"after block {n}", similarity) for n, similarity in enumerate(result["layers"][1:], start=1)]
    print(f"Similarity of the two {WORD!r} vectors, stage by stage:")
    for label, similarity in stages:
        print(f"  {label:<40} {similarity:.2f}")
    print(f"\nSame word, same ID, same input vector. After reading the sentence: {result['layers'][-1]:.2f}")

    print("\nFlukes? Two more sentences, final vectors only:")
    for label, x, y in [("river ~ river", river, river_2), ("money ~ money", money, money_2),
                        ("river ~ money", river, money)]:
        print(f"  {label}   {compare(vectors[x], vectors[y])['layers'][-1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
