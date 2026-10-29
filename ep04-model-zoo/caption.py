"""
caption.py - Gemini looks at one photo and describes it (step 9).

A multimodal model takes more than one kind of input. Here ONE message carries text
AND a picture, as two content blocks. Gemini then writes a sentence: unlike CLIP, it generates.

We shrink the photo first (longest side 1024 pixels). A 4 MB phone photo becomes a few
hundred KB, which uploads faster. The free tier may use what you send to improve Google's
products, so send only photos you are happy to share (no faces of other people, no documents).

Run:  python caption.py                     (the first photo in photos/)
      python caption.py photos/plant.jpg      (a photo you choose)
"""

import base64
import io
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from PIL import Image, ImageOps

from common import HERE, find_photos
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

QUESTION = "Describe this photo in one sentence for someone who cannot see it."
MAX_SIDE = 1024   # pixels


def image_block(path, max_side: int = MAX_SIDE) -> dict:
    """The photo as a LangChain image content block: the JPEG bytes as base64 text, plus its type."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # right way up, RGB colours
    image.thumbnail((max_side, max_side))                              # keeps the shape, only ever shrinks
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return {"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/jpeg"}


def build_message(path, question: str = QUESTION) -> HumanMessage:
    """One message, two content blocks: our question (text) and the photo (image)."""
    return HumanMessage(content=[{"type": "text", "text": question}, image_block(path)])


def caption(path, llm=None) -> str:
    """Ask the model to describe the photo. Returns its sentence as plain text."""
    if llm is None:
        llm = get_llm()
    return llm.invoke([build_message(path)]).text   # .text, not .content (Gemini 3 returns a list of blocks)


def main(argv=None, llm=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    photos = [Path(argv[0])] if argv else find_photos()[:1]
    if not photos or not photos[0].is_file():
        print("No photo found. Put a .jpg or .png in photos/, or run: python caption.py path/to/photo.jpg")
        return 1
    photo = photos[0]

    try:
        provider = get_provider()
        print(f"Provider: {provider}   Model: {get_model_name(provider)}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")        # friendly help (no key, or a typo in LLM_PROVIDER)
        return 1

    block = image_block(photo)
    sent_kb = len(base64.b64decode(block["base64"])) // 1024
    full = photo.resolve()
    shown = full.relative_to(HERE).as_posix() if full.is_relative_to(HERE) else str(photo)
    print(f"Photo: {shown} (sent as a {sent_kb} KB JPEG)")
    print(f"Question: {QUESTION}")

    start = time.perf_counter()
    sentence = caption(photo, llm)
    print(f"\nCaption ({time.perf_counter() - start:.1f} s):\n{sentence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
