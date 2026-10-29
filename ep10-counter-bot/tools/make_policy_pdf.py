"""
tools/make_policy_pdf.py - writes data/policy.pdf (2 pages) and the tiny test PDF tests/fixtures/two_pages.pdf.

You do NOT need to run this: both PDFs are already in the folder.
It shows how they were made. Making your own policy? Write it in Google Docs or LibreOffice
and use File > Download > PDF (Google Docs) or File > Export as PDF (LibreOffice) instead.

It uses matplotlib, which is already in the course libraries.

Run (from the episode folder):  python tools/make_policy_pdf.py

Author: Ravi Shanker Singh (TechWithRavi). Kadak Kettle and every rule in this policy are invented.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                        # draw into files, never open a window
matplotlib.rcParams["pdf.fonttype"] = 42     # store real text in the PDF, so a loader can read it back
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
POLICY_PDF = HERE / "data" / "policy.pdf"
FIXTURE_PDF = HERE / "tests" / "fixtures" / "two_pages.pdf"

TITLE = "Kadak Kettle: Allergens, Pickup and Refunds"

# Each page is a list of (kind, text). kind: "title", "h" (heading), "p" (paragraph) or "b" (bullet).
# Two sentences say "Please tell" and "Send /forget" instead of "Tell" and "Type": in this font the PDF
# tucks the "e" and "y" under the "T", and pypdf then reads "T ell" and "T ype" (Episode 07 met the same thing).
POLICY_PAGES = [
    [
        ("title", TITLE),
        ("p", "Version 1.0, October 2026. Kadak Kettle is an invented cafe used for teaching. "
              "Every rule on these two pages is made up."),
        ("h", "1. Allergens on our menu"),
        ("p", "Three items contain nuts: kesar badam doodh (almonds and pistachios), kanda poha "
              "(roasted peanuts) and peanut chikki (peanuts)."),
        ("b", "Milk: every chai except lemon ginger tea, kesar badam doodh, bun maska and the paneer kathi roll."),
        ("b", "Wheat (gluten): vada pav, samosa, khasta kachori, bun maska, egg bhurji pav and the kathi roll."),
        ("b", "Egg: egg bhurji pav. It is cooked on its own pan, but in the same kitchen."),
        ("h", "2. Our kitchen is not nut-free"),
        ("p", "Everything is cooked in one small kitchen. Peanuts, almonds and pistachios are handled on "
              "the same counter, and all fried snacks share the same oil. We cannot promise that any item "
              "is free from traces of nuts, even an item whose recipe has no nuts."),
        ("h", "3. Tell us before you order"),
        ("p", "Please tell the person at the counter about any allergy before you pay. The counter bot "
              "can remind you of this rule, but it cannot check your food for you."),
    ],
    [
        ("h", "4. Pickup orders"),
        ("p", "Send us your order and your name. A pickup order is ready in about 15 minutes, and we hold "
              "it at the counter for 30 minutes. After that the order is cancelled and fried snacks go to "
              "the next customer while they are still hot."),
        ("b", "Pay when you collect, in cash or by UPI."),
        ("b", "Orders of more than 20 items need one hour of notice."),
        ("h", "5. Refunds and replacements"),
        ("p", "If an item is wrong, cold or not fresh, tell us within 2 hours and show your bill. We will "
              "replace it, or refund it in full. A drink that is more than half finished cannot be refunded."),
        ("h", "6. Your chat with the counter bot"),
        ("p", "The counter bot keeps your conversation so it can remember your name, your order and your "
              "allergies between visits. Send /forget in the chat, or ask the staff, and we delete the "
              "whole conversation. We never use your chat for anything else."),
    ],
]

# A tiny 2-page PDF for the tests: short, exact sentences you can check by eye.
FIXTURE_PAGES = [
    [("title", "Test policy"), ("p", "Page one says hello.")],
    [("p", "Page two says goodbye.")],
]

PAGE_W, PAGE_H = 8.27, 11.69      # A4 in inches
LEFT, TOP = 0.09, 0.93            # where the text starts, as fractions of the page
WRAP = 88                         # characters per line at 10 pt


def draw_page(fig, items, footer):
    """Write one page from top to bottom."""
    y = TOP
    for kind, text in items:
        if kind == "title":
            fig.text(LEFT, y, text, fontsize=16, weight="bold")
            y -= 0.035
            continue
        if kind == "h":
            y -= 0.012
            fig.text(LEFT, y, text, fontsize=12.5, weight="bold")
            y -= 0.026
            continue
        prefix = "- " if kind == "b" else ""
        for i, line in enumerate(textwrap.wrap(text, width=WRAP - len(prefix))):
            fig.text(LEFT, y, (prefix if i == 0 else " " * len(prefix)) + line, fontsize=10)
            y -= 0.018
        y -= 0.008
    if footer:
        fig.text(LEFT, 0.04, footer, fontsize=8, color="0.4")


def write_pdf(path, pages, title, footer=True):
    """Save `pages` as a PDF file. CreationDate=None keeps the file the same on every run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": title, "Author": "Kadak Kettle (invented)", "CreationDate": None}
    with PdfPages(path, metadata=metadata) as pdf:
        for number, items in enumerate(pages, start=1):
            fig = Figure(figsize=(PAGE_W, PAGE_H))
            draw_page(fig, items, f"{title} - page {number} of {len(pages)}" if footer else "")
            pdf.savefig(fig)


def main():
    write_pdf(POLICY_PDF, POLICY_PAGES, TITLE)
    write_pdf(FIXTURE_PDF, FIXTURE_PAGES, "Test policy", footer=False)
    for path in (POLICY_PDF, FIXTURE_PDF):
        print(f"Wrote {path.relative_to(HERE)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
