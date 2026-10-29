"""
skills/ask_photo.py - ask questions about your own photo, on your laptop (steps 10 and 11).

A vision-language model (VLM) cuts the photo into small square patches, turns every patch into
a vector, and feeds those "image tokens" to a small language model next to the tokens of your
question. The language model attends to both and writes the answer, word by word.
There is no search and no database anywhere in this: the answer is generated from the pixels.

do_image_splitting=False sends the photo as ONE 512 x 512 image instead of many tiles.
Answers get a little less detailed, but on a laptop CPU it is many times faster.

Run (inside ep16-local-ai-toolbox):
    python -m skills.ask_photo                                     (photos/desk.jpg, two questions)
    python -m skills.ask_photo photos/my_desk.jpg "What is on the shelf?"
    python -m skills.ask_photo photos/desk.jpg "How many pencils are on the desk?" --truth 5
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from common import OUTPUTS_DIR, SAMPLE_PHOTO, VISION_MODEL, Stopwatch, device, find_photos, \
    is_downloaded, not_downloaded_help, offline_mode
from skills.lazy import lazy_load

QUESTIONS = ["What objects are on the desk?", "What colour is the mug?"]
LIMITS_LOG = OUTPUTS_DIR / "limits.csv"
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


@dataclass
class Answer:
    text: str
    seconds: float


def load_vlm():
    """Processor (prepares photo + text) and model, wrapped in one image-text-to-text pipeline."""
    from transformers import AutoProcessor, pipeline

    processor = AutoProcessor.from_pretrained(VISION_MODEL, do_image_splitting=False)
    return pipeline("image-text-to-text", model=VISION_MODEL, processor=processor,
                    dtype="auto", device=device())


def open_photo(path):
    """The photo the right way up (phones store a rotation flag) and in RGB colours."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def ask(image, question: str, max_new_tokens: int = 40) -> Answer:
    """One question about one photo. Returns the short answer and the seconds it took."""
    vlm = lazy_load("vision", load_vlm)
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": question}]}]
    with Stopwatch() as watch:
        output = vlm(text=messages, return_full_text=False,
                     generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False})
    return Answer(output[0]["generated_text"].strip(), watch.seconds)


def answer_matches(answer: str, truth: str) -> bool:
    """Does the answer contain the right value? '5' also matches 'five', and the other way round."""
    words = set(re.findall(r"[a-z0-9]+", answer.lower()))
    wanted = {truth.lower().strip()}
    if truth.strip().isdigit() and int(truth) < len(NUMBER_WORDS):
        wanted.add(NUMBER_WORDS[int(truth)])
    if truth.lower().strip() in NUMBER_WORDS:
        wanted.add(str(NUMBER_WORDS.index(truth.lower().strip())))
    return bool(words & wanted)


def log_limit(photo, question: str, answer: str, truth: str, path=None) -> Path:
    """Write a wrong answer into outputs/limits.csv, so you remember what this model cannot do."""
    path = Path(path or LIMITS_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["when", "model", "photo", "question", "answer", "truth"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), VISION_MODEL, Path(photo).name,
                         question, answer, truth])
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m skills.ask_photo",
                                     description="Ask a small vision-language model about your photo.")
    parser.add_argument("photo", nargs="?", help="a .jpg or .png (default: the first picture in photos/)")
    parser.add_argument("questions", nargs="*", help="one or more questions in quotes")
    parser.add_argument("--truth", help="the right answer, if you know it: a wrong answer is logged")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    photo = Path(args.photo) if args.photo else (find_photos()[:1] or [SAMPLE_PHOTO])[0]
    if not photo.is_file():
        print(f"No photo at {photo}. Put a .jpg or .png in photos/ (a photo YOU took, no other people's faces).")
        return 1
    questions = args.questions or QUESTIONS
    if offline_mode() and not is_downloaded(VISION_MODEL):
        print(not_downloaded_help(VISION_MODEL))
        return 1

    try:
        image = open_photo(photo)
    except OSError:
        print(f"Cannot read {photo.name} as a picture. Use a .jpg or .png file.")
        return 1

    try:
        if not is_downloaded(VISION_MODEL):
            print(f"First run: downloading {VISION_MODEL} (about 0.5 GB, only once)...")
        with Stopwatch() as load:
            lazy_load("vision", load_vlm)
        print(f"Photo: {photo.name} ({image.width}x{image.height})   Model: {VISION_MODEL} on {device()} "
              f"(ready in {load.seconds:.1f} s)\n")
        for question in questions:
            answer = ask(image, question)
            print(f"Q: {question}\nA: {answer.text}   ({answer.seconds:.1f} s)")
            if args.truth:
                if answer_matches(answer.text, args.truth):
                    print(f"   right (the true answer is {args.truth})")
                else:
                    where = log_limit(photo, question, answer.text, args.truth)
                    print(f"   WRONG: the true answer is {args.truth}. Logged in {where.parent.name}/{where.name}")
            print()
    except OSError:
        print(not_downloaded_help(VISION_MODEL))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
