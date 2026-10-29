"""
make_sample_photos.py - draws the four sample pictures in photos/ (you do not need to run it).

The best pictures for step 8 are four photos YOU take at home (a cup, a plant, a shoe, a book).
These drawings are the fallback, so the demo also works before you have taken any.
They are drawn here with Pillow, shape by shape, so nobody else's photo is copied.
Released under CC0 (public domain): use them for anything.

    python tools/make_sample_photos.py        (inside ep04-model-zoo)

Careful: it overwrites cup.jpg, plant.jpg, shoe.jpg and book.jpg in photos/, so move your own
photos somewhere else first. The tests draw their own copy in a temporary folder instead.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent.parent
PHOTOS = HERE / "photos"
SIZE = 640
TABLE_Y = 430          # where the table top starts
SEED = 42              # same "random" wood grain on every run


def background(seed: int) -> Image.Image:
    """A soft wall on top and a wooden table at the bottom."""
    rng = random.Random(seed)
    img = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(img)
    for y in range(SIZE):
        if y < TABLE_Y:                                # wall: light, slightly warm
            t = y / TABLE_Y
            colour = (int(232 - 18 * t), int(226 - 18 * t), int(214 - 16 * t))
        else:                                          # table: warm brown, darker towards us
            t = (y - TABLE_Y) / (SIZE - TABLE_Y)
            colour = (int(170 - 40 * t), int(118 - 30 * t), int(76 - 22 * t))
        draw.line([(0, y), (SIZE, y)], fill=colour)
    for _ in range(40):                                # wood grain
        y = rng.randint(TABLE_Y + 4, SIZE)
        shade = rng.randint(-18, 10)
        draw.line([(0, y), (SIZE, y + rng.randint(-6, 6))],
                  fill=(150 + shade, 102 + shade, 64 + shade), width=1)
    return img


def shadow(img: Image.Image, box) -> None:
    """A blurred dark ellipse under an object."""
    layer = Image.new("L", img.size, 0)
    ImageDraw.Draw(layer).ellipse(box, fill=150)
    layer = layer.filter(ImageFilter.GaussianBlur(14))
    img.paste((60, 40, 25), (0, 0), layer)


def shaded_rect(draw, x0, y0, x1, y1, dark, light) -> None:
    """Fill a rectangle with a left-dark, centre-light, right-dark shading (a round object)."""
    for x in range(x0, x1):
        t = (x - x0) / max(1, x1 - x0)
        k = 1 - abs(t - 0.38) * 1.6                   # brightest a little left of centre
        k = max(0.0, min(1.0, k))
        colour = tuple(int(d + (l - d) * k) for d, l in zip(dark, light))
        draw.line([(x, y0), (x, y1)], fill=colour)


def cup() -> Image.Image:
    img = background(SEED)
    shadow(img, (170, 470, 470, 540))
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 455, 490, 535), fill=(236, 236, 232), outline=(200, 200, 196), width=3)   # saucer
    draw.ellipse((210, 470, 430, 515), fill=(222, 222, 218))
    draw.arc((372, 300, 470, 430), start=-80, end=95, fill=(214, 214, 210), width=22)          # handle
    shaded_rect(draw, 220, 260, 420, 480, dark=(196, 196, 192), light=(250, 250, 248))       # body
    draw.ellipse((220, 462, 420, 498), fill=(222, 222, 218))                                    # rounded base
    shaded_rect(draw, 220, 270, 420, 478, dark=(196, 196, 192), light=(250, 250, 248))
    draw.ellipse((220, 240, 420, 290), fill=(240, 240, 236), outline=(190, 190, 186), width=3)  # rim
    draw.ellipse((232, 248, 408, 284), fill=(120, 70, 35))                                      # tea
    draw.ellipse((262, 254, 330, 266), fill=(150, 95, 55))                                      # shine on tea
    for i, x in enumerate((280, 320, 360)):                                                      # steam
        points = [(x + 12 * math.sin((y + 30 * i) / 18), y) for y in range(110, 230, 4)]
        draw.line(points, fill=(255, 255, 255), width=6, joint="curve")
    return img.filter(ImageFilter.GaussianBlur(0.8))


def leaf(img: Image.Image, base, angle: float, length: int, width: int, green) -> None:
    """One rounded leaf on a short stem, pointing at `angle` degrees (0 = straight up)."""
    a = math.radians(angle - 90)
    centre = (base[0] + 0.62 * length * math.cos(a), base[1] + 0.62 * length * math.sin(a))
    ImageDraw.Draw(img).line([base, centre], fill=(46, 110, 50), width=5)             # stem
    layer = Image.new("RGBA", (length * 2, length * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse((length - width // 2, length // 2, length + width // 2, length * 3 // 2),
              fill=green + (255,), outline=(30, 90, 40, 255), width=2)
    d.line([(length, length // 2 + 6), (length, length * 3 // 2 - 6)], fill=(30, 100, 45, 255), width=2)
    layer = layer.rotate(-angle, resample=Image.BICUBIC)
    img.paste(layer, (int(centre[0] - length), int(centre[1] - length)), layer)


def plant() -> Image.Image:
    img = background(SEED + 1)
    shadow(img, (200, 490, 440, 545))
    base = (320, 322)
    # (angle, length, width): broad round leaves, like a money plant on a window sill
    for angle, length, width in [(-62, 120, 62), (62, 120, 62), (-38, 140, 70), (38, 140, 70),
                                 (-14, 160, 76), (14, 158, 76), (0, 120, 66), (-80, 96, 54), (80, 96, 54)]:
        green = (46 + abs(int(angle)) // 3, 140 - abs(int(angle)) // 4, 62)
        leaf(img, base, angle, length, width, green)
    draw = ImageDraw.Draw(img)
    draw.polygon([(225, 330), (415, 330), (390, 505), (250, 505)], fill=(190, 92, 52))  # pot
    shaded_rect(draw, 238, 345, 402, 500, dark=(160, 72, 40), light=(214, 112, 66))
    draw.polygon([(225, 330), (238, 345), (238, 500), (250, 505)], fill=(170, 80, 45))
    draw.polygon([(415, 330), (402, 345), (402, 500), (390, 505)], fill=(150, 66, 36))
    draw.rectangle((212, 312, 428, 345), fill=(204, 102, 60), outline=(150, 70, 40), width=2)  # rim
    draw.ellipse((226, 306, 414, 326), fill=(70, 45, 30))                                      # soil
    return img.filter(ImageFilter.GaussianBlur(0.8))


def shoe() -> Image.Image:
    img = background(SEED + 2)
    shadow(img, (110, 470, 540, 535))
    draw = ImageDraw.Draw(img)
    upper = [(140, 470), (150, 360), (200, 330), (260, 345), (300, 380), (360, 400),
             (450, 415), (510, 440), (525, 470)]
    draw.polygon(upper, fill=(40, 90, 170), outline=(25, 55, 110))                             # upper
    draw.polygon([(150, 360), (200, 330), (230, 338), (190, 380), (150, 400)], fill=(30, 70, 140))  # collar
    draw.polygon([(430, 412), (510, 440), (525, 470), (440, 470)], fill=(230, 230, 235))       # toe cap
    draw.polygon([(140, 470), (148, 400), (180, 410), (175, 470)], fill=(230, 230, 235))       # heel tab
    draw.line([(250, 440), (330, 395), (420, 445)], fill=(240, 240, 245), width=12)            # side stripe
    for i in range(5):                                                                          # laces
        x = 262 + i * 26
        draw.line([(x, 350 + i * 9), (x + 20, 372 + i * 8)], fill=(250, 250, 250), width=6)
    draw.rounded_rectangle((128, 462, 538, 498), radius=16, fill=(245, 245, 245),
                           outline=(190, 190, 190), width=3)                                    # sole
    draw.line([(140, 488), (528, 488)], fill=(200, 200, 200), width=4)                         # sole grip
    return img.filter(ImageFilter.GaussianBlur(0.8))


def book() -> Image.Image:
    img = background(SEED + 3)
    shadow(img, (120, 450, 540, 540))
    draw = ImageDraw.Draw(img)
    # a closed hardback lying on the table, seen from the front and a little above
    draw.polygon([(150, 470), (500, 470), (520, 500), (170, 500)], fill=(245, 238, 220))   # page block
    for y in range(474, 500, 4):
        draw.line([(155 + (y - 470) * 0.66, y), (505 + (y - 470) * 0.66, y)], fill=(215, 205, 185))
    draw.polygon([(150, 470), (170, 500), (170, 512), (148, 480)], fill=(120, 30, 40))     # spine edge
    draw.polygon([(170, 500), (520, 500), (522, 512), (170, 512)], fill=(130, 32, 42))     # back cover
    draw.polygon([(170, 250), (470, 250), (500, 470), (150, 470)], fill=(150, 36, 48),
                 outline=(100, 20, 30))                                                    # front cover
    draw.polygon([(190, 250), (205, 250), (182, 470), (166, 470)], fill=(120, 28, 38))     # spine band
    draw.rectangle((250, 300, 420, 350), fill=(232, 196, 110))                             # title label
    draw.line([(270, 318), (400, 318)], fill=(120, 80, 30), width=6)                       # title text
    draw.line([(285, 334), (385, 334)], fill=(120, 80, 30), width=4)
    draw.line([(240, 420), (430, 420)], fill=(232, 196, 110), width=4)                     # author line
    return img.filter(ImageFilter.GaussianBlur(0.8))


def main(folder=PHOTOS) -> None:
    """Draw the four pictures into folder (photos/ unless you pass another folder; the tests do)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, make in [("cup", cup), ("plant", plant), ("shoe", shoe), ("book", book)]:
        path = folder / f"{name}.jpg"
        make().save(path, quality=90)
        print(f"wrote {folder.name}/{path.name}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
