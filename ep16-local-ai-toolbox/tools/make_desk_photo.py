"""
make_desk_photo.py - draws the sample picture photos/desk.jpg (you do not need to run it).

The best picture for steps 10 and 11 is a photo of YOUR own desk, taken with your phone.
This drawing is the fallback, so the demo also works before you have taken one.
It is drawn here with Pillow, shape by shape, so nobody else's photo is copied, and there
are no people in it. Released under CC0 (public domain): use it for anything.

What is in the picture (so we know the right answers):
  an open laptop, a red mug, a green plant in a pot, a stack of three books,
  and FIVE yellow pencils lying side by side.

    python tools/make_desk_photo.py              (inside ep16-local-ai-toolbox)

Careful: it overwrites photos/desk.jpg, so give your own photo another name (for example my_desk.jpg).
The tests draw their own copy in a temporary folder instead.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent.parent
PHOTOS = HERE / "photos"
WIDTH, HEIGHT = 960, 640
DESK_Y = 360           # where the desk top starts
SEED = 16              # same "random" wood grain on every run
PENCILS = 5            # the answer to the counting question in step 11


def wall_and_desk(rng: random.Random) -> Image.Image:
    """A pale wall on top and a wooden desk at the bottom."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        if y < DESK_Y:                                      # wall: light grey-blue
            t = y / DESK_Y
            colour = (int(222 - 14 * t), int(228 - 14 * t), int(234 - 12 * t))
        else:                                               # desk: warm brown, darker towards us
            t = (y - DESK_Y) / (HEIGHT - DESK_Y)
            colour = (int(176 - 46 * t), int(124 - 34 * t), int(80 - 24 * t))
        draw.line([(0, y), (WIDTH, y)], fill=colour)
    for _ in range(60):                                     # wood grain
        y = rng.randint(DESK_Y + 4, HEIGHT)
        shade = rng.randint(-20, 8)
        draw.line([(0, y), (WIDTH, y + rng.randint(-5, 5))], fill=(152 + shade, 104 + shade, 66 + shade))
    draw.rectangle((0, DESK_Y - 6, WIDTH, DESK_Y + 4), fill=(120, 80, 48))   # back edge of the desk
    return img


def soft_shadow(img: Image.Image, box) -> None:
    """A blurred dark ellipse under an object, so it sits on the desk."""
    layer = Image.new("L", img.size, 0)
    ImageDraw.Draw(layer).ellipse(box, fill=140)
    img.paste((60, 40, 25), (0, 0), layer.filter(ImageFilter.GaussianBlur(12)))


def laptop(img: Image.Image, rng: random.Random) -> None:
    """An open laptop in the middle: dark screen with lines of 'code', grey keyboard base."""
    soft_shadow(img, (250, 500, 690, 560))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((300, 150, 640, 400), radius=12, fill=(60, 64, 72))      # lid
    draw.rectangle((316, 166, 624, 384), fill=(22, 28, 44))                          # screen
    colours = [(0, 200, 230), (120, 220, 120), (240, 200, 90), (200, 200, 210)]
    for row in range(12):                                                            # code lines
        indent = rng.choice([0, 0, 20, 40])
        length = rng.randint(60, 220)
        y = 180 + row * 16
        draw.line([(332 + indent, y), (332 + indent + length, y)], fill=rng.choice(colours), width=5)
    draw.polygon([(270, 520), (670, 520), (640, 400), (300, 400)], fill=(178, 182, 190))   # base
    for row in range(4):                                                             # keys
        y0 = 412 + row * 22
        left = 312 - row * 7
        for col in range(12):
            x0 = left + col * (27 + row * 0.6)
            draw.rectangle((x0, y0, x0 + 20, y0 + 14), fill=(96, 100, 110))
    draw.rectangle((420, 500, 520, 514), fill=(150, 154, 162))                       # touchpad


def mug(img: Image.Image) -> None:
    """A red mug on the right."""
    soft_shadow(img, (708, 486, 860, 524))
    draw = ImageDraw.Draw(img)
    draw.arc((800, 390, 870, 470), start=-90, end=90, fill=(170, 30, 36), width=14)  # handle
    draw.rectangle((720, 370, 830, 500), fill=(200, 38, 44))                         # body
    for x in range(722, 760):                                                        # shine
        draw.line([(x, 378), (x, 494)], fill=(222, 70, 74) if x < 740 else (210, 52, 58))
    draw.ellipse((720, 490, 830, 510), fill=(200, 38, 44))                           # round base
    draw.ellipse((720, 358, 830, 382), fill=(214, 60, 64), outline=(150, 24, 30), width=3)   # rim
    draw.ellipse((728, 362, 822, 378), fill=(90, 50, 26))                            # coffee


def plant(img: Image.Image) -> None:
    """A small green plant in a terracotta pot on the left."""
    soft_shadow(img, (60, 470, 220, 510))
    draw = ImageDraw.Draw(img)
    leaves = [(-50, 200), (-30, 175), (-8, 160), (12, 165), (34, 178), (52, 200), (0, 190)]
    for dx, top in leaves:                                                           # leaves
        x = 140 + dx
        draw.ellipse((x - 22, top, x + 22, top + 110), fill=(52, 140, 70), outline=(30, 96, 46), width=2)
        draw.line([(x, top + 20), (140, 330)], fill=(36, 110, 52), width=3)
    draw.polygon([(88, 320), (192, 320), (178, 488), (102, 488)], fill=(192, 98, 58))   # pot
    draw.rectangle((80, 306, 200, 330), fill=(206, 110, 66), outline=(150, 72, 40), width=2)  # rim


def books(img: Image.Image) -> None:
    """A stack of three books at the back left, each a different colour."""
    soft_shadow(img, (20, 330, 260, 372))
    draw = ImageDraw.Draw(img)
    stack = [((26, 322, 256, 352), (40, 90, 160)),       # bottom: blue
             ((38, 294, 244, 322), (40, 140, 90)),       # middle: green
             ((30, 268, 236, 294), (210, 150, 40))]      # top: yellow-orange
    for box, colour in stack:
        draw.rectangle(box, fill=colour, outline=(40, 30, 30), width=2)
        x0, y0, x1, y1 = box
        draw.rectangle((x1 - 14, y0 + 3, x1 - 4, y1 - 3), fill=(244, 238, 222))     # page edge


def pencils(img: Image.Image, count: int = PENCILS) -> None:
    """`count` yellow pencils lying side by side at the front of the desk."""
    draw = ImageDraw.Draw(img)
    for i in range(count):
        y = 560 + i * 15
        x0, x1 = 300 + i * 6, 560 + i * 6
        draw.rectangle((x0, y, x1, y + 9), fill=(246, 196, 40), outline=(160, 120, 20))   # body
        draw.rectangle((x1, y, x1 + 12, y + 9), fill=(190, 190, 196))                     # metal band
        draw.rectangle((x1 + 12, y, x1 + 22, y + 9), fill=(236, 120, 140))                # eraser
        draw.polygon([(x0, y), (x0 - 22, y + 4.5), (x0, y + 9)], fill=(236, 200, 150))    # wood tip
        draw.polygon([(x0 - 14, y + 2), (x0 - 22, y + 4.5), (x0 - 14, y + 7)], fill=(50, 50, 50))  # lead


def draw_desk() -> Image.Image:
    rng = random.Random(SEED)
    img = wall_and_desk(rng)
    books(img)
    plant(img)
    laptop(img, rng)
    mug(img)
    pencils(img)
    return img.filter(ImageFilter.GaussianBlur(0.7))     # soften the hard drawn edges a little


def main(folder=PHOTOS) -> Path:
    """Draw desk.jpg into folder (photos/ unless you pass another folder; the tests do)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "desk.jpg"
    draw_desk().save(path, quality=90)
    print(f"wrote {folder.name}/{path.name}  ({path.stat().st_size // 1024} KB, {WIDTH}x{HEIGHT})")
    return path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PHOTOS)
