"""
common.py - small helpers shared by the Episode 04 scripts.

  load_reviews()       read data/reviews.txt, one review per line (step 2)
  count_parameters()   how many learned numbers a model has (steps 3, 5, 8)
  in_millions()        22713216 -> "22.7 million"
  find_photos()        the pictures in photos/ (steps 8, 9)
  make_table()         a neat table for the terminal (uses the rich library)
  OWNER_SYSTEM, OWNER_PROMPT   the ONE prompt we give every text model (steps 5, 6, 12)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

HERE = Path(__file__).resolve().parent
REVIEWS_FILE = HERE / "data" / "reviews.txt"
PHOTOS_DIR = HERE / "photos"
OUTPUTS = HERE / "outputs"            # results we save for later steps (git ignores this folder)
LOCAL_RESULT_FILE = OUTPUTS / "local_decoder.json"   # written by decoder_local.py, read by decoder_cloud.py
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp"}

console = Console()

# The same request goes to the small local model (step 5) and to Gemini (step 6).
OWNER_SYSTEM = "You are the owner of Kettle Point, a small tea stall. Reply to customers politely and briefly."
OWNER_PROMPT = (
    'A customer wrote this review: "My filter coffee arrived lukewarm and the cup was sticky on the outside." '
    "Write a two-sentence reply that says sorry and promises one fix."
)


def load_reviews(path=REVIEWS_FILE) -> list[str]:
    """Return the reviews in the file, one per line. Empty lines are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def count_parameters(model) -> int:
    """Count every learned number (weight) inside a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def in_millions(count: int) -> str:
    """22713216 -> '22.7 million'. Easier to say out loud than eight digits."""
    return f"{count / 1_000_000:.1f} million"


def find_photos(folder=PHOTOS_DIR) -> list[Path]:
    """All pictures in the folder (jpg, jpeg, png, webp), sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_TYPES)


def make_table(columns: list[str], rows: list[list[str]], title: str | None = None,
               keep_whole: tuple[str, ...] = ()) -> Table:
    """Build a rich Table: print it with console.print(table).

    keep_whole: names of columns that must never wrap (for example model names).

    Every cell is plain Text, so square brackets in your data are shown as they are
    (rich would otherwise read "[best]" as a style tag and hide it).
    """
    table = Table(title=Text(title) if title else None, box=box.ROUNDED, title_justify="left")
    for name in columns:
        # wrap long text onto the next line, never cut it off with "..."
        table.add_column(name, overflow="fold", no_wrap=name in keep_whole)
    for row in rows:
        table.add_row(*[Text(str(cell)) for cell in row])
    return table
