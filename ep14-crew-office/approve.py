"""
approve.py - Step 7: the human approval gate. A junior writes the letter; the manager signs it.

The crew stops at drafts/pending/. Nothing moves on until a PERSON reads each draft and types:

    a  approve  -> saved in drafts/approved/ with a mailto: link. Click it and YOUR mail app opens
                   with the email filled in. You read it once more and press Send yourself.
    e  edit     -> the draft opens as a text file; change it, save it, press Enter, then decide
                   (skip an edited draft and your edit is kept in drafts/pending/)
    r  reject   -> moved to drafts/rejected/
    s  skip     -> stays pending for later

This program cannot send email: it has no mail server, no password, no network code at all
(tests/test_approve.py checks every file of the project for that). That is on purpose: it is
the "excessive agency" lesson from Episode 05. Writing to REAL people? You also need their
consent and must follow your country's rules (in India: the DPDP Act 2023, see the README).

    python approve.py

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from claims_check import as_value, find_numbers, unsupported_numbers
from models import LeadBrief

HERE = Path(__file__).resolve().parent
DRAFTS = HERE / "drafts"
# Added by CODE to every draft (outreach_crew.py) and kept after every edit: a way to say no.
OPT_OUT = "If this is not useful, just reply 'no thanks' and we will not write again."


def shown(path: Path) -> str:
    """drafts/approved/x.md (short) when the file is inside this folder, else the full path."""
    return path.relative_to(HERE).as_posix() if path.is_relative_to(HERE) else str(path)


def mailto_link(draft: dict) -> str:
    """A mailto: link. It only OPENS your mail app with the email filled in; it sends nothing."""
    return (f"mailto:{quote(draft['to_email'], safe='@')}"
            f"?subject={quote(draft['subject'])}&body={quote(draft['body'])}")


def load_pending(drafts: Path = DRAFTS) -> list[tuple[Path, dict]]:
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted((drafts / "pending").glob("*.json"))]


def show(draft: dict, number: int, total: int, lead: str) -> None:
    verdict = draft["verdict"]
    print("=" * 70)
    print(f"Draft {number} of {total}: {lead}")
    print(f"To:        {draft['to_name']} <{draft['to_email']}>   (from {draft['lead_file']}, not from the model)")
    print(f"Subject:   {draft['subject']}")
    print(f"Written:   {draft['written_by']}, {draft['made_at']}")
    print("-" * 70)
    print(draft["body"])
    print("-" * 70)
    fixes = "; ".join(verdict["fixes"]) or "none"
    print(f"Reviewer:  {verdict['status']}   tone: {verdict['tone']}   fixes: {fixes}")


def check_edit(draft: dict) -> list[str]:
    """After a human edit: numbers that are not in the brief (a person can make mistakes too)."""
    brief = LeadBrief(**draft["brief"])
    allowed = [as_value(n)[0] for item in brief.facts + brief.product_points for n in find_numbers(item.text)]
    return unsupported_numbers(f"{draft['subject']}\n{draft['body']}", allowed)


def apply_edit(draft: dict, text: str) -> dict:
    """text = 'Subject: ...' on the first line, then the body. The opt-out line always stays."""
    first, _, body = text.strip().partition("\n")
    subject = first.removeprefix("Subject:").strip() or draft["subject"]
    body = body.strip()
    if OPT_OUT not in body:
        body = f"{body}\n\n{OPT_OUT}"          # our code puts it back: every email offers a way out
    return {**draft, "subject": subject, "body": body, "edited_by_human": True}


def approve(path: Path, draft: dict, drafts: Path = DRAFTS) -> Path:
    """Save the approved draft as a readable file plus its JSON. Sends NOTHING."""
    folder = drafts / "approved"
    folder.mkdir(parents=True, exist_ok=True)
    approved = {**draft, "status": "approved by a human, NOT sent",
                "approved_at": datetime.now().isoformat(timespec="seconds")}
    text = folder / f"{path.stem}.md"
    text.write_text(
        f"To: {draft['to_name']} <{draft['to_email']}>\n"
        f"Subject: {draft['subject']}\n"
        f"Status: approved by a human at {approved['approved_at']}. NOT sent: send it yourself.\n\n"
        f"{draft['body']}\n\n"
        f"Open in your mail app: {mailto_link(draft)}\n",
        encoding="utf-8",
    )
    (folder / path.name).write_text(json.dumps(approved, indent=2, ensure_ascii=False), encoding="utf-8")
    path.unlink()
    return text


def reject(path: Path, drafts: Path = DRAFTS) -> Path:
    folder = drafts / "rejected"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    shutil.move(str(path), target)
    return target


def edit(path: Path, draft: dict, ask, drafts: Path = DRAFTS) -> dict:
    """Write the draft to a text file, let the human change it, read it back."""
    folder = drafts / "editing"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"{path.stem}.txt"
    file.write_text(f"Subject: {draft['subject']}\n\n{draft['body']}\n", encoding="utf-8")
    ask(f"Edit {shown(file)} (in VS Code), save it, then press Enter here ")
    edited = apply_edit(draft, file.read_text(encoding="utf-8"))
    file.unlink()
    outside = check_edit(edited)
    if outside:
        print(f"WARNING: your edit has numbers that are not in the brief: {outside}. Check them.")
    return edited


def review_all(ask=input, drafts: Path = DRAFTS) -> dict:
    """Show each pending draft and do what the human types. Returns the counts."""
    pending = load_pending(drafts)
    counts = {"approved": 0, "rejected": 0, "skipped": 0}
    if not pending:
        print("No drafts waiting in drafts/pending/. Make some first:  python outreach_crew.py")
        return counts
    print(f"{len(pending)} draft(s) waiting in drafts/pending/ (written by outreach_crew.py)\n")
    for number, (path, draft) in enumerate(pending, start=1):
        show(draft, number, len(pending), path.stem)
        choice = ask("[a]pprove  [e]dit  [r]eject  [s]kip ? ").strip().lower()[:1]
        edited = choice == "e"
        if edited:
            draft = edit(path, draft, ask, drafts)
            print(f"\nEdited subject: {draft['subject']}\n{draft['body']}\n")
            choice = ask("[a]pprove  [r]eject  [s]kip the edited draft ? ").strip().lower()[:1]
        if choice == "a":
            saved = approve(path, draft, drafts)
            counts["approved"] += 1
            print(f"Approved. Saved {shown(saved)}")
            print("Click this link to open it in your mail app; you press Send there, or not at all:")
            print(mailto_link(draft) + "\n")
        elif choice == "r":
            reject(path, drafts)
            counts["rejected"] += 1
            print(f"Rejected. Moved to drafts/rejected/{path.name}\n")
        else:
            counts["skipped"] += 1
            if edited:                      # keep the person's work: the edited text replaces the old draft
                path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
                print("Skipped. Your edit is saved, and the draft stays in drafts/pending/.\n")
            else:
                print("Skipped. It stays in drafts/pending/.\n")
    saved = counts["approved"]
    print(f"Saved {saved} draft{'' if saved == 1 else 's'}. Nothing was sent.")
    if counts["rejected"] or counts["skipped"]:
        print(f"({counts['rejected']} rejected, {counts['skipped']} still pending)")
    return counts


if __name__ == "__main__":
    try:
        review_all()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Nothing was sent.")
        sys.exit(1)
