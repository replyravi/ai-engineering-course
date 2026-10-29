"""
models.py - read the model card BEFORE you build on a model (step 2).

The Hugging Face Hub is a huge public library. Every model has a card, like the label on the
spine of a library book: what it does, how big it is, which languages, and who may use it for what.
"Free to download" does NOT always mean "free for your business": some licences forbid
commercial use, and gated models make you accept terms (and log in with a token) first.

In Episode 04 we typed model cards by hand. Here huggingface_hub reads them for us,
and saves them in model_cards.json so the report also works offline.

Run (inside ep16-local-ai-toolbox):
    python models.py            ask the Hub (internet), save model_cards.json, print the report
    python models.py --saved    print the report from model_cards.json (no internet needed)
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from common import CHAT_MODEL, EMBEDDINGS_MODEL, HERE, VISION_MODEL

CARDS_FILE = HERE / "model_cards.json"

# The three models this project downloads, and the job each one does.
OUR_MODELS = [
    (CHAT_MODEL, "summarise + translate"),
    (VISION_MODEL, "ask about a photo"),
    (EMBEDDINGS_MODEL, "search notes"),
]

# Models you meet in older tutorials or may try next. We only READ their cards; nothing is downloaded.
OTHER_MODELS = [
    ("facebook/bart-large-cnn", "old summarization pipeline"),
    ("facebook/nllb-200-distilled-600M", "old translation pipeline"),
    ("Salesforce/blip-vqa-base", "old visual Q&A"),
    ("Qwen/Qwen2.5-3B-Instruct", "a bigger sibling of our chat model"),
    ("black-forest-labs/FLUX.1-dev", "text to image"),
]

# What each licence tag means for a business. Always read the licence text before a real product.
PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0"}


def business_use(licence: str, licence_name: str | None = None) -> str:
    """A short, careful label for the licence tag on the card."""
    licence = (licence or "unknown").lower()
    if licence in PERMISSIVE:
        return "yes (keep the notice)"
    if licence.startswith("cc-by-nc") or "non-commercial" in (licence_name or ""):
        return "NO: non-commercial"
    return f"read it ({licence_name or licence})"


def weights_bytes(siblings) -> int:
    """Size of the main weights files (the .safetensors at the top of the repo, else the .bin)."""
    files = [(s.rfilename, s.size or 0) for s in siblings if "/" not in s.rfilename]
    safetensors = [size for name, size in files if name.endswith(".safetensors")]
    return sum(safetensors) if safetensors else sum(size for name, size in files if name.endswith(".bin"))


def fetch_card(model_id: str, job: str, group: str) -> dict:
    """Ask the Hub for one model's card and file list (needs internet, no token for public models)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, files_metadata=True)
    card = info.card_data or {}
    languages = card.get("language") or []
    return {
        "id": model_id,
        "group": group,
        "job": job,
        "task": info.pipeline_tag,
        "licence": card.get("license") or "unknown",
        "licence_name": card.get("license_name"),
        "gated": bool(info.gated),                           # False, or "auto"/"manual" on gated repos
        "parameters": info.safetensors.total if info.safetensors else None,
        "weights_bytes": weights_bytes(info.siblings),
        "languages": [languages] if isinstance(languages, str) else list(languages),
        "last_modified": info.last_modified.date().isoformat() if info.last_modified else None,
        "url": f"https://huggingface.co/{model_id}",
        "checked": date.today().isoformat(),
    }


def fetch_all() -> list[dict]:
    return ([fetch_card(m, job, "ours") for m, job in OUR_MODELS]
            + [fetch_card(m, job, "other") for m, job in OTHER_MODELS])


def save_cards(cards: list[dict], path=CARDS_FILE) -> None:
    Path(path).write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_cards(path=CARDS_FILE) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached_sizes() -> dict[str, int]:
    """model id -> bytes it takes in YOUR Hugging Face cache folder (works offline)."""
    try:
        from huggingface_hub import scan_cache_dir

        return {repo.repo_id: repo.size_on_disk for repo in scan_cache_dir().repos if repo.repo_type == "model"}
    except Exception:          # no cache folder yet
        return {}


def size_text(n_bytes) -> str:
    if not n_bytes:
        return "-"
    return f"{n_bytes / 1e9:.1f} GB" if n_bytes >= 999.5e6 else f"{n_bytes / 1e6:.0f} MB"


def params_text(n) -> str:
    if not n:
        return "-"
    return f"{n / 1e9:.1f} B" if n >= 1e9 else f"{n / 1e6:.1f} M"


def languages_text(languages: list[str]) -> str:
    if not languages:
        return "-"
    return ", ".join(languages) if len(languages) <= 2 else f"{len(languages)} listed"


def report(cards: list[dict], cache: dict[str, int] | None = None) -> str:
    """The licence and size report: two lines per model, ours first, then the others."""
    cache = cache or {}
    lines = []
    titles = {"ours": "Models this project downloads",
              "other": "Models from older tutorials, or next steps (we only read their cards)"}
    for group, title in titles.items():
        rows = [c for c in cards if c["group"] == group]
        if not rows:
            continue
        lines.append(f"{title}:")
        lines.append(f"  {'model':<26}{'licence':<15}{'business use':<24}{'weights':>8}{'params':>9}"
                     f"  {'gated':<6}{'on disk':>8}")
        for c in rows:
            name = c["id"].split("/")[-1]
            gated = "YES" if c["gated"] else "no"
            lines.append(f"  {name:<26}{c['licence']:<15}{business_use(c['licence'], c['licence_name']):<24}"
                         f"{size_text(c['weights_bytes']):>8}{params_text(c['parameters']):>9}"
                         f"  {gated:<6}{size_text(cache.get(c['id'])):>8}")
            lines.append(f"      task {c['task'] or '-'} | languages {languages_text(c['languages'])} "
                         f"| updated {c['last_modified'] or '-'} | used for: {c['job']}")
        lines.append("")
    checked = sorted({c["checked"] for c in cards})
    lines.append(f"Cards checked on {', '.join(checked)}. Licences change: read the card again before a product.")
    if any(c["gated"] for c in cards):
        lines.append("Gated = accept the terms on the model page, then log in with a read-only HF_TOKEN in .env.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Licence and size report for every model we use.")
    parser.add_argument("--saved", action="store_true", help="use model_cards.json, do not ask the Hub")
    args = parser.parse_args(argv)

    if args.saved:
        cards = load_cards()
        print(f"From the saved {CARDS_FILE.name} (no internet used)\n")
    else:
        try:
            cards = fetch_all()
            save_cards(cards)
            print(f"From the Hugging Face Hub (saved to {CARDS_FILE.name})\n")
        except Exception as error:      # offline, HF_HUB_OFFLINE=1, or the Hub is busy
            if not CARDS_FILE.exists():
                print(f"Could not reach the Hub and there is no saved {CARDS_FILE.name} yet: {error}")
                return 1
            cards = load_cards()
            print(f"Could not reach the Hub ({type(error).__name__}). Showing the saved {CARDS_FILE.name}\n")
    print(report(cards, cached_sizes()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
