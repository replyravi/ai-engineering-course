"""
zoo_report.py - which model for which job? (step 11)

Reads model_cards.json (checked by model_cards_schema.py) and prints one row per job
we did today. Choose by the job, not by the hype: the smallest model that does the
job well, with a licence that allows what you plan to do.

Run:  python zoo_report.py
"""

from common import console, make_table
from model_cards_schema import ModelCard, load_cards

# (the job, the id of the model we used for it, the script that did it today)
JOBS = [
    ("Find similar reviews", "sentence-transformers/all-MiniLM-L6-v2", "encoder.py"),
    ("Write text offline", "Qwen/Qwen2.5-0.5B-Instruct", "decoder_local.py"),
    ("Write fluent text, label reviews", "gemini-3.5-flash-lite", "decoder_cloud.py, zero_shot.py"),
    ("Label photos with your words", "openai/clip-vit-base-patch32", "multimodal_clip.py"),
    ("Caption a photo", "gemini-3.5-flash-lite", "caption.py"),
]

COLUMNS = ["job", "model", "family", "writes?", "runs on", "business use"]


def report_rows(cards: list[ModelCard], jobs=JOBS) -> list[list[str]]:
    """One row per job, filled in from the matching model card."""
    by_id = {card.id: card for card in cards}
    rows = []
    for job, model_id, script in jobs:
        if model_id not in by_id:
            raise KeyError(f"no card for {model_id} in model_cards.json (job: {job})")
        card = by_id[model_id]
        rows.append([job, card.name, card.family, "yes" if card.generates else "no",
                     card.runs_on, card.business_use])
    return rows


def main() -> int:
    try:
        cards = load_cards()
        rows = report_rows(cards)
    except (ValueError, KeyError) as error:
        print(str(error).strip("'\""))
        print("Run: python model_cards_schema.py")
        return 1

    console.print(make_table(COLUMNS, rows, title="Which model for which job", keep_whole=("model", "writes?")))
    oldest = min(card.checked for card in cards)
    print(f"{len(rows)} jobs, {len(cards)} model cards. Facts checked on {oldest}.")
    print("Licences and model versions change: read the card again before you build a product on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
