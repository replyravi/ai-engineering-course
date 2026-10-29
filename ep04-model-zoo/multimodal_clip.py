"""
multimodal_clip.py - CLIP matches your photos with sentences (step 8).

CLIP was pre-trained on a huge number of picture + caption pairs from the internet.
It learned to pull a picture and its own caption close together, and push the wrong
captions away (contrastive learning). So it can score ANY labels you write, with no
training on your photos.

It cannot write a caption. It only scores how well each sentence fits each photo:
another foundation model that generates nothing.

Run:  python multimodal_clip.py                          (the five labels below)
      python multimodal_clip.py "a dog" "a car" "a tree"   (your own labels)
      python multimodal_clip.py --grid                     (look inside: photo x caption similarity)
"""

import os
import sys
import time

# This checkpoint stores its weights as pytorch_model.bin. Without this line, transformers
# also downloads a second copy (about 600 MB, .safetensors format) in the background.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import torch  # noqa: E402  (imported after the line above on purpose)
from PIL import Image, ImageOps  # noqa: E402
from transformers import pipeline  # noqa: E402

from common import console, count_parameters, find_photos, in_millions, make_table  # noqa: E402

MODEL_ID = "openai/clip-vit-base-patch32"
LABELS = ["a cup", "a plant", "a shoe", "a book", "a bicycle"]   # the bicycle matches none of the photos
TEMPLATE = "a photo of {}."                                       # CLIP compares whole sentences


def load_clip():
    """Load CLIP (the first run downloads about 600 MB)."""
    return pipeline("zero-shot-image-classification", model=MODEL_ID, device="cpu")


def open_photo(path) -> Image.Image:
    """Open a photo the right way up (phones store the rotation separately), in RGB colours."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def score_photo(clip, image, labels: list[str]) -> list[tuple[str, float]]:
    """(label, score) pairs, best first. The scores of all labels add up to 1."""
    results = clip(image, candidate_labels=labels, hypothesis_template=TEMPLATE)
    return [(r["label"], r["score"]) for r in results]


def expected_label(path, labels: list[str]) -> str | None:
    """'a cup' for a file called cup.jpg or my_cup_2.png. None if the name matches no label."""
    name = path.stem.lower()
    for label in labels:
        if label.split()[-1] in name:   # the last word of the label: "cup"
            return label
    return None


def similarity_grid(clip, images, captions: list[str]) -> torch.Tensor:
    """Cosine similarity of every photo (rows) with every caption (columns)."""
    with torch.no_grad():
        text = clip.tokenizer(captions, padding=True, return_tensors="pt")
        pixels = clip.image_processor(images, return_tensors="pt")
        out = clip.model(**text, **pixels)
    return out.image_embeds @ out.text_embeds.T   # both are unit-length vectors, so this is the cosine


def show_scores(clip, photos, labels: list[str]) -> None:
    rows, right, checked = [], 0, 0
    for path in photos:
        scores = score_photo(clip, open_photo(path), labels)
        (best, best_score), (second, second_score) = scores[0], scores[1]
        expected = expected_label(path, labels)
        if expected is None:
            verdict = "-"
        else:
            checked += 1
            right += best == expected
            verdict = "yes" if best == expected else f"no, expected {expected}"
        rows.append([path.name, best, f"{best_score:.1%}", f"{second} ({second_score:.1%})", verdict])
    console.print(make_table(["photo", "best label", "score", "runner-up", "matches file name?"], rows,
                             title=f"CLIP scores against: {', '.join(labels)}"))
    if checked:
        print(f"Top label matches the file name for {right} of {checked} photos.")


def show_grid(clip, photos, labels: list[str]) -> None:
    matched = [(p, expected_label(p, labels)) for p in photos]
    matched = [(p, label) for p, label in matched if label]
    if not matched:
        print("--grid needs photos named after a label, like cup.jpg or plant.png.")
        return
    captions = [TEMPLATE.format(label) for _, label in matched]
    grid = similarity_grid(clip, [open_photo(p) for p, _ in matched], captions)
    rows = []
    for (path, _), values in zip(matched, grid.tolist()):
        best = max(values)
        rows.append([path.name] + [f"[{v:.2f}]" if v == best else f"{v:.2f}" for v in values])
    console.print(make_table(["photo"] + captions, rows, title="Cosine similarity, photo x caption ([best] in each row)"))
    print("The right caption wins each row by only a few hundredths. CLIP multiplies these numbers by",
          f"{clip.model.logit_scale.exp().item():.0f} before turning them into percentages, so small gaps become big scores.")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    grid_mode = "--grid" in argv
    labels = [a for a in argv if a != "--grid"] or LABELS

    photos = find_photos()
    if not photos:
        print("No pictures found in photos/. Add a few .jpg or .png files and run again.")
        return 1

    start = time.perf_counter()
    clip = load_clip()
    params = count_parameters(clip.model)
    print(f"CLIP: {MODEL_ID}   Parameters: {params:,} (about {in_millions(params)})")
    print(f"Loaded in {time.perf_counter() - start:.1f} s. Photos: {len(photos)}\n")

    if grid_mode:
        show_grid(clip, photos, labels)
    else:
        show_scores(clip, photos, labels)
        print("CLIP wrote nothing. It only scored how well each label fits each photo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
