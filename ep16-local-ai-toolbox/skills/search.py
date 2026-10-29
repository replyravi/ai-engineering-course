"""
skills/search.py - find notes by MEANING, not by matching words (steps 7, 8 and 9).

An embeddings model turns a whole sentence into ONE list of 384 numbers: an arrow in a
384-dimensional space. Notes about the same topic point in a similar direction.
Cosine similarity measures the angle between two arrows:
   1.00 = same direction (same meaning), 0 = unrelated, below 0 = pointing away.

This is not how a chat model reads. A chat model reads token IDs (Episode 02).
Embeddings are a separate, small model used for search, clustering and RAG (Episode 07).

Run (inside ep16-local-ai-toolbox):
    python -m skills.search --shape            step 7: encode every note
    python -m skills.search --matrix           step 8: every note against every note
    python -m skills.search "train ticket"     step 9: rank the notes for a question
"""

import argparse
import sys

import numpy as np

from common import EMBEDDINGS_MODEL, NOTES_DIR, Stopwatch, device, is_downloaded, load_notes, \
    not_downloaded_help, offline_mode, shorten
from skills.lazy import lazy_load


def load_encoder():
    """The sentence-transformers model (about 90 MB, downloaded once)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDINGS_MODEL, device=device())


def encode(texts: list[str]) -> np.ndarray:
    """One row of 384 numbers per text. Rows have length 1, so a dot product IS the cosine."""
    encoder = lazy_load("embeddings", load_encoder)
    return encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row with every row: an N x N grid, 1.00 on the diagonal."""
    return vectors @ vectors.T


def most_similar_pair(matrix: np.ndarray) -> tuple[int, int, float]:
    """The two DIFFERENT items with the highest score (the diagonal, an item with itself, is skipped)."""
    grid = matrix.astype(float).copy()
    np.fill_diagonal(grid, -np.inf)
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    return int(min(i, j)), int(max(i, j)), float(matrix[i, j])


def least_similar_pair(matrix: np.ndarray) -> tuple[int, int, float]:
    """The two different items with the lowest score."""
    grid = matrix.astype(float).copy()
    np.fill_diagonal(grid, np.inf)
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    return int(min(i, j)), int(max(i, j)), float(matrix[i, j])


def format_matrix(matrix: np.ndarray, labels: list[str]) -> str:
    """The grid as text: numbered columns, one row per label, two decimals with their sign."""
    width = max(len(label) for label in labels) + 4
    header = " " * width + "".join(f"{n:>7}" for n in range(1, len(labels) + 1))
    rows = [header]
    for n, (label, row) in enumerate(zip(labels, matrix), start=1):
        cells = "".join(f"{value:>7.2f}" for value in row)
        rows.append(f"{n:>2}. {label:<{width - 4}}{cells}")
    return "\n".join(rows)


def rank(query_vector: np.ndarray, vectors: np.ndarray, labels: list[str]) -> list[tuple[str, float]]:
    """Every label with its cosine score for the query, best first."""
    scores = vectors @ query_vector
    order = np.argsort(-scores, kind="stable")
    return [(labels[i], float(scores[i])) for i in order]


def search(query: str, notes, k: int = 3) -> list[tuple[str, float]]:
    """The k notes closest in meaning to the query, as (note name, score)."""
    vectors = encode([note.text for note in notes])
    query_vector = encode([query])[0]
    return rank(query_vector, vectors, [note.name for note in notes])[:k]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m skills.search",
                                     description="Search your notes by meaning.")
    parser.add_argument("query", nargs="?", help='what you are looking for, e.g. "train ticket"')
    parser.add_argument("-k", type=int, default=3, help="how many notes to show (default 3)")
    parser.add_argument("--shape", action="store_true", help="encode every note and print the shape")
    parser.add_argument("--matrix", action="store_true", help="print the full similarity matrix")
    parser.add_argument("--notes", default=str(NOTES_DIR), help="folder of .txt notes (default my_notes/)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    notes = load_notes(args.notes)
    if len(notes) < 2:
        print(f"Put at least two .txt notes in {args.notes} first.")
        return 1
    if offline_mode() and not is_downloaded(EMBEDDINGS_MODEL):
        print(not_downloaded_help(EMBEDDINGS_MODEL))
        return 1
    try:
        return run(args, notes)
    except OSError:                      # not downloaded yet and no internet
        print(not_downloaded_help(EMBEDDINGS_MODEL))
        return 1


def run(args, notes) -> int:
    """Do what the command line asked for."""
    if args.shape:
        with Stopwatch() as watch:
            vectors = encode([note.text for note in notes])
        print(f"Encoded {len(notes)} notes with {EMBEDDINGS_MODEL} in {watch.seconds:.1f} s")
        print(f"Shape: {vectors.shape}   ({len(notes)} notes x {vectors.shape[1]} numbers each)")
        first_five = [round(float(x), 3) for x in vectors[0][:5]]
        lengths = [round(float(x), 2) for x in np.linalg.norm(vectors, axis=1)]
        print(f"First 5 numbers of {notes[0].name}: {first_five}")
        print(f"Length of every vector: {lengths}  (normalised, so a dot product is the cosine)")
        return 0

    if args.matrix:
        labels = [note.name for note in notes]
        matrix = similarity_matrix(encode([note.text for note in notes]))
        print("Cosine similarity of every note with every note (-1 to 1):\n")
        print(format_matrix(matrix, labels))
        i, j, score = most_similar_pair(matrix)
        print(f"\nMost similar pair:  {labels[i]} + {labels[j]}  ({score:.2f})")
        i, j, score = least_similar_pair(matrix)
        print(f"Least similar pair: {labels[i]} + {labels[j]}  ({score:+.2f})")
        return 0

    if not args.query:
        build_parser().print_help()
        return 1
    print(f'Query: "{args.query}"\n')
    by_name = {note.name: note for note in notes}
    for place, (name, score) in enumerate(search(args.query, notes, args.k), start=1):
        print(f"{place}. {score:+.3f}  {name:<16} {shorten(by_name[name].text, 60)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
