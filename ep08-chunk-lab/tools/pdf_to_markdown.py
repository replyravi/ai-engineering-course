"""
tools/pdf_to_markdown.py - turn our 3-page food safety PDF into Markdown, so the header splitter can see its sections.

You do NOT need to run this: data/handbook/food_safety.md is already made.
It shows the idea behind tools like MarkItDown or Docling on a PDF whose layout we know:
the first line is the title, numbered lines like "3. Allergens in our food" are headings,
lines starting with "- " are bullet points, and the page footers are thrown away.

Run (from the episode folder):  python tools/pdf_to_markdown.py

The PDF is a copy of Episode 07's handbook/food_safety.pdf (same file, same MD5).
Author: Ravi Shanker Singh (TechWithRavi). The company and every rule in it are invented.
"""

import re
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent.parent
PDF_FILE = HERE / "data" / "pdf" / "food_safety.pdf"
MD_FILE = HERE / "data" / "handbook" / "food_safety.md"

HEADING = re.compile(r"^\d+\. [A-Z][^.]*$")      # "3. Allergens in our food" (no full stop at the end)
FOOTER = re.compile(r" - page \d+ of \d+$")      # "... - page 1 of 3"
SHORT_LINE = 70                                  # a line shorter than this that ends with "." ends a paragraph


def pdf_to_markdown(pdf_path=PDF_FILE):
    """Read the PDF page by page and rebuild headings, paragraphs and bullet points."""
    blocks = []     # finished Markdown blocks: "# title", "## heading", "a paragraph", "- a bullet"
    current = []    # the lines of the paragraph or bullet we are building
    is_bullet = False

    def finish_block():
        nonlocal current, is_bullet
        if current:
            text = " ".join(current)
            blocks.append("- " + text if is_bullet else text)
        current, is_bullet = [], False

    for page in PdfReader(pdf_path).pages:
        for line in page.extract_text().split("\n"):
            if FOOTER.search(line):
                continue                                  # page footer: not part of the content
            if not blocks and not current:
                blocks.append("# " + line.strip())        # the very first line is the title
            elif HEADING.match(line.strip()):
                finish_block()
                blocks.append("## " + line.strip())
            elif line.startswith("- "):
                finish_block()
                is_bullet, current = True, [line[2:].strip()]
            elif line.startswith("  ") and is_bullet:
                current.append(line.strip())              # a bullet that wrapped onto the next line
            else:
                if is_bullet:
                    finish_block()
                current.append(line.strip())
                if len(line) < SHORT_LINE and line.rstrip().endswith("."):
                    finish_block()                        # a short last line closes the paragraph
    finish_block()

    # Join the blocks: bullets of one list sit on neighbouring lines, everything else gets a blank line.
    markdown = blocks[0]
    for previous, block in zip(blocks, blocks[1:]):
        same_list = previous.startswith("- ") and block.startswith("- ")
        markdown += ("\n" if same_list else "\n\n") + block
    return markdown + "\n"


def main():
    markdown = pdf_to_markdown()
    MD_FILE.write_text(markdown, encoding="utf-8", newline="\n")     # "\n" line ends on Windows too
    headings = [line for line in markdown.splitlines() if line.startswith("#")]
    print(f"Wrote {MD_FILE.relative_to(HERE).as_posix()}: {len(markdown):,} characters, {len(headings)} headings")
    for heading in headings:
        print(f"  {heading}")


if __name__ == "__main__":
    main()
