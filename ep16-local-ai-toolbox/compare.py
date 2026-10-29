"""
compare.py - the same job on your laptop and in the cloud, side by side (steps 5, 6, 11 and 14).

  local  the small Hugging Face models: private, free, work with Wi-Fi off, but clumsier
  cloud  Gemini through get_llm() (or whatever LLM_PROVIDER says): stronger, but your text
         or photo leaves your laptop, and it needs internet and a key

Every run is added to outputs/runs.csv, and "python compare.py table" turns those runs into
one local-versus-cloud table: speed, privacy, cost and offline.

Run (inside ep16-local-ai-toolbox):
    python compare.py summarise [note]                          step 5
    python compare.py translate [note]                          step 6
    python compare.py ask-photo [photo] "question" --truth 5    step 11
    python compare.py table                                     the table from every run so far
Add --local-only to skip the cloud, or --cloud-only to skip the local model.
"""

import argparse
import base64
import csv
import io
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from common import CHAT_MODEL, NOTES_DIR, OUTPUTS_DIR, SAMPLE_NOTE, SAMPLE_PHOTO, VISION_MODEL, \
    Stopwatch, not_downloaded_help, read_note, shorten, word_count
from llm import MissingAPIKeyError, check_api_key, get_llm, get_model_name, get_provider

RUNS_FILE = OUTPUTS_DIR / "runs.csv"
COUNTING_QUESTION = "How many pencils are on the desk?"

# Where each cloud provider lives, to check the internet before we try (Ollama runs on your laptop).
CLOUD_HOSTS = {"gemini": "generativelanguage.googleapis.com", "openai": "api.openai.com",
               "anthropic": "api.anthropic.com"}
WHERE = {"gemini": "Google's servers", "openai": "OpenAI's servers", "anthropic": "Anthropic's servers",
         "ollama": "your laptop (Ollama)"}


@dataclass
class Row:
    engine: str      # "local" or "cloud"
    model: str
    where: str       # where the model ran, so where your data went
    seconds: float
    text: str


# ---------------------------------------------------------------------------
# 1. Is the cloud model ready? (internet first, then the key)
# ---------------------------------------------------------------------------

def internet_ok(host: str, timeout: float = 3.0) -> bool:
    """Can we open a connection to host? Sends no data, just knocks on the door."""
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except Exception:            # no Wi-Fi, no DNS, a firewall, or the test guard
        return False


def cloud_problem() -> str:
    """'' when the cloud model can be called, otherwise a friendly explanation."""
    provider = get_provider()
    host = CLOUD_HOSTS.get(provider)
    if host and not internet_ok(host):
        return (f"No internet: skipped the cloud column ({get_model_name(provider)}).\n"
                "The local results above came from your laptop and never needed the internet.")
    try:
        check_api_key(provider)
    except MissingAPIKeyError as error:
        return str(error).strip()
    return ""


# ---------------------------------------------------------------------------
# 2. The local side (the skills) and the cloud side (one LangChain call each)
# ---------------------------------------------------------------------------

def local_row(skill: str, item, question: str = "") -> Row:
    """Run one local skill. item = note text (summarise, translate) or a photo path (ask-photo)."""
    if skill == "summarise":
        from skills.summarise import summarise
        reply = summarise(item)
        return Row("local", CHAT_MODEL, "your laptop", reply.seconds, reply.text)
    if skill == "translate":
        from skills.translate import translate
        reply = translate(item, "Hindi")
        return Row("local", CHAT_MODEL, "your laptop", reply.seconds, reply.text)
    from skills.ask_photo import ask, open_photo
    answer = ask(open_photo(item), question)
    return Row("local", VISION_MODEL, "your laptop", answer.seconds, answer.text)


def photo_block(path, max_side: int = 1024) -> dict:
    """The photo as a LangChain image block: a smaller JPEG, as base64 text."""
    from skills.ask_photo import open_photo

    image = open_photo(path)
    image.thumbnail((max_side, max_side))        # a phone photo shrinks to a few hundred KB
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return {"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/jpeg"}


def cloud_messages(skill: str, item, question: str = "") -> list:
    """The SAME instructions the local model gets, as LangChain messages (the last one is from the human)."""
    if skill == "summarise":
        from skills.summarise import SYSTEM
        return [SystemMessage(SYSTEM), HumanMessage(f"Summarise this note:\n\n{item}")]
    if skill == "translate":
        from skills.translate import system_message
        return [SystemMessage(system_message("Hindi")), HumanMessage(item)]
    return [HumanMessage(content=[{"type": "text", "text": question}, photo_block(item)])]


def cloud_row(skill: str, llm, item, question: str = "") -> Row:
    """One call to the cloud model, timed. .text (not .content): Gemini 3 returns a list of blocks."""
    provider = get_provider()
    model = getattr(llm, "model", None) or getattr(llm, "model_name", None) or get_model_name(provider)
    with Stopwatch() as watch:
        reply = llm.invoke(cloud_messages(skill, item, question))
    return Row("cloud", str(model).removeprefix("models/"), WHERE.get(provider, "the cloud"),
               watch.seconds, reply.text.strip())


# ---------------------------------------------------------------------------
# 3. Printing and the run log
# ---------------------------------------------------------------------------

def print_rows(rows: list[Row]) -> None:
    for row in rows:
        print(f"{row.engine:<6} {row.model}  ({row.where}, {row.seconds:.1f} s, {word_count(row.text)} words)")
        print(f"  {row.text}\n")
    print(f"{'engine':<8}{'where it ran':<24}{'seconds':>8}{'words':>7}   data left your laptop?")
    for row in rows:
        left = "no" if row.where.startswith("your laptop") else "yes"
        print(f"{row.engine:<8}{row.where:<24}{row.seconds:>8.1f}{word_count(row.text):>7}   {left}")


def log_rows(skill: str, rows: list[Row], path=None) -> None:
    """Add the rows to outputs/runs.csv (made on the first run)."""
    path = Path(path or RUNS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["when", "skill", "engine", "model", "where", "seconds", "words", "answer"])
        for row in rows:
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), skill, row.engine, row.model,
                             row.where, f"{row.seconds:.2f}", word_count(row.text), row.text])


def read_runs(path=None) -> list[dict]:
    path = Path(path or RUNS_FILE)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def comparison_table(runs: list[dict]) -> str:
    """Local versus cloud in one table, built from the logged runs."""
    def speed(engine: str, skill: str) -> str:
        times = [float(r["seconds"]) for r in runs if r["engine"] == engine and r["skill"] == skill]
        if not times:
            return "not run yet"
        return f"{sum(times) / len(times):.1f} s ({len(times)} run{'s' if len(times) > 1 else ''})"

    cloud_models = sorted({r["model"] for r in runs if r["engine"] == "cloud"})
    cloud_name = ", ".join(cloud_models) or get_model_name(get_provider())
    cloud_where = next((r["where"] for r in runs if r["engine"] == "cloud"), WHERE.get(get_provider(), "the cloud"))
    cloud_is_local = cloud_where.startswith("your laptop")
    lines = [f"{'':<18}{'local (Hugging Face models)':<34}cloud ({cloud_name})"]
    for skill in ("summarise", "translate", "ask-photo"):
        lines.append(f"{skill + ' speed':<18}{speed('local', skill):<34}{speed('cloud', skill)}")
    lines += [
        f"{'privacy':<18}{'stays on your laptop':<34}"
        f"{'stays on your laptop' if cloud_is_local else 'sent to ' + cloud_where}",
        f"{'cost':<18}{'free (your electricity)':<34}{'free' if cloud_is_local else 'free tier with limits, or paid'}",
        f"{'works offline':<18}{'yes (HF_HUB_OFFLINE=1)':<34}{'yes' if cloud_is_local else 'no'}",
        f"{'quality':<18}you judge: read the answers in outputs/runs.csv and outputs/limits.csv",
    ]
    if get_provider() == "gemini" and not cloud_is_local:
        lines.append("Gemini free tier: Google may use what you send to improve its products. Send nothing private.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Command line
# ---------------------------------------------------------------------------

def add_engine_switches(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-only", action="store_true", help="skip the cloud model")
    group.add_argument("--cloud-only", action="store_true", help="skip the local model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python compare.py", description="Local versus cloud, side by side.")
    sub = parser.add_subparsers(dest="skill", required=True)
    for name, default in (("summarise", SAMPLE_NOTE), ("translate", NOTES_DIR / "plant_care.txt")):
        p = sub.add_parser(name, help=f"{name} a note both ways")
        p.add_argument("note", nargs="?", default=str(default))
        add_engine_switches(p)
    p = sub.add_parser("ask-photo", help="ask the same question about a photo both ways")
    p.add_argument("photo", nargs="?", default=str(SAMPLE_PHOTO))
    p.add_argument("question", nargs="?", default=COUNTING_QUESTION)
    p.add_argument("--truth", help="the right answer, to mark each engine right or wrong")
    add_engine_switches(p)
    sub.add_parser("table", help="local versus cloud, from every run in outputs/runs.csv")
    return parser


def main(argv=None, llm=None) -> int:
    args = build_parser().parse_args(argv)
    if args.skill == "table":
        runs = read_runs()
        print(f"Local vs cloud, from {len(runs)} logged answer(s) in outputs/runs.csv\n")
        print(comparison_table(runs))
        return 0

    item_path = Path(args.photo if args.skill == "ask-photo" else args.note)
    if not item_path.is_file():
        print(f"No file at {item_path}.")
        return 1
    item = item_path if args.skill == "ask-photo" else read_note(item_path)
    question = getattr(args, "question", "")
    print(f"{args.skill}: {item_path.name}" + (f'  Q: "{question}"' if question else "") + "\n")

    rows, problem = [], ""
    if not args.cloud_only:
        try:
            rows.append(local_row(args.skill, item, question))
        except OSError:
            print(not_downloaded_help(VISION_MODEL if args.skill == "ask-photo" else CHAT_MODEL))
            return 1
    if not args.local_only:
        problem = "" if llm is not None else cloud_problem()
        if not problem:
            try:
                rows.append(cloud_row(args.skill, llm or get_llm(), item, question))
            except Exception as error:       # 429 free-tier limit, a network drop, a wrong model name...
                problem = f"The cloud call failed: {shorten(str(error), 200)}"

    if rows:
        print_rows(rows)
        log_rows(args.skill, rows)
    if args.skill == "ask-photo" and args.truth:
        from skills.ask_photo import answer_matches
        for row in rows:
            verdict = "right" if answer_matches(row.text, args.truth) else "WRONG"
            print(f"{row.engine}: {verdict} (the true answer is {args.truth})")
    if problem:
        print(f"\n{problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
