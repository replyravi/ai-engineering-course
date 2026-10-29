"""
test_approve.py - Step 7: the human gate, and proof that no code here can send an email.
"""

import ast
import json
from pathlib import Path

import approve
from models import EmailDraft, LeadBrief, Verdict
from office_helpers import KESAR, PROJECT
from replay import scripted_brief, scripted_draft

# Python modules that can talk to a mail server, or send any web request of our own. None may appear.
NETWORK_MODULES = {"smtplib", "imaplib", "poplib", "ftplib", "telnetlib", "socket", "ssl", "http",
                   "urllib.request", "requests", "httpx", "aiohttp", "urllib3", "websocket"}
SEND_WORDS = ("sendmail(", "send_message(", "starttls(")
# The only imports that reach the internet, and why: the model calls, and the one-time model downloads.
# (CrewAI's "onnx" embedder in Step 8 also downloads MiniLM once, through chromadb, inside crewai.)
ALLOWED_ONLINE = {"crewai": "calls the AI model you chose in .env",
                  "huggingface_hub": "downloads the tone model once (Step 3)"}


def make_pending(drafts: Path, monkeypatch, plant: bool = False) -> Path:
    import outreach_crew

    monkeypatch.setattr(outreach_crew, "PENDING", drafts / "pending")
    brief = LeadBrief(**scripted_brief(KESAR))
    draft = EmailDraft(**scripted_draft(KESAR, plant=plant))
    verdict = Verdict(status="pass", tone="[scripted] positive 0.73, ok")
    return outreach_crew.save_pending(KESAR, brief, draft, verdict, "scripted replay (no AI model)")


def answers(*replies):
    """A pretend person at the keyboard: each call to ask() returns the next reply."""
    queue = list(replies)
    return lambda prompt="": queue.pop(0)


def test_approve_saves_a_file_and_a_mailto_link_and_sends_nothing(tmp_path, monkeypatch, capsys):
    pending = make_pending(tmp_path, monkeypatch)
    draft = json.loads(pending.read_text(encoding="utf-8"))
    assert draft["to_email"] == "nandini.apte@example.com"               # from the lead file
    assert draft["body"].endswith(approve.OPT_OUT)                       # added by code

    counts = approve.review_all(ask=answers("a"), drafts=tmp_path)

    out = capsys.readouterr().out
    assert counts == {"approved": 1, "rejected": 0, "skipped": 0}
    assert "Saved 1 draft. Nothing was sent." in out
    assert "mailto:nandini.apte@example.com?subject=%5Bscripted%5D%20Pick-up" in out
    saved = (tmp_path / "approved" / "kesar-crumb-bakehouse.md").read_text(encoding="utf-8")
    assert "NOT sent: send it yourself." in saved
    assert not pending.exists()


def test_reject_and_skip(tmp_path, monkeypatch, capsys):
    make_pending(tmp_path, monkeypatch)
    assert approve.review_all(ask=answers("s"), drafts=tmp_path)["skipped"] == 1
    assert approve.review_all(ask=answers("r"), drafts=tmp_path)["rejected"] == 1
    assert (tmp_path / "rejected" / "kesar-crumb-bakehouse.json").exists()
    assert "Saved 0 drafts. Nothing was sent." in capsys.readouterr().out


def test_edit_keeps_the_opt_out_and_warns_about_new_numbers(tmp_path, monkeypatch, capsys):
    make_pending(tmp_path, monkeypatch)

    def person(prompt=""):
        if prompt.startswith("Edit"):        # the person edits the text file, then presses Enter
            file = tmp_path / "editing" / "kesar-crumb-bakehouse.txt"
            file.write_text("Subject: A shorter subject\n\nDear Nandini,\nPick-ups go 50% faster.\n", encoding="utf-8")
            return ""
        return "e" if prompt.startswith("[a]pprove  [e]dit") else "a"

    approve.review_all(ask=person, drafts=tmp_path)

    out = capsys.readouterr().out
    assert "WARNING: your edit has numbers that are not in the brief: ['50%']" in out
    saved = json.loads((tmp_path / "approved" / "kesar-crumb-bakehouse.json").read_text(encoding="utf-8"))
    assert saved["subject"] == "A shorter subject"
    assert saved["body"].endswith(approve.OPT_OUT) and saved["edited_by_human"]


def test_an_edited_draft_you_skip_keeps_your_edit(tmp_path, monkeypatch, capsys):
    pending = make_pending(tmp_path, monkeypatch)

    def person(prompt=""):
        if prompt.startswith("Edit"):
            file = tmp_path / "editing" / "kesar-crumb-bakehouse.txt"
            file.write_text("Subject: Decide tomorrow\n\nDear Nandini,\nA shorter email.\n", encoding="utf-8")
            return ""
        return "e" if prompt.startswith("[a]pprove  [e]dit") else "s"

    assert approve.review_all(ask=person, drafts=tmp_path)["skipped"] == 1

    assert "Your edit is saved" in capsys.readouterr().out
    kept = json.loads(pending.read_text(encoding="utf-8"))
    assert kept["subject"] == "Decide tomorrow" and kept["edited_by_human"]
    assert kept["body"].endswith(approve.OPT_OUT)


def test_no_pending_drafts_is_a_friendly_message(tmp_path, capsys):
    approve.review_all(ask=answers(), drafts=tmp_path)
    assert "python outreach_crew.py" in capsys.readouterr().out


def test_no_code_in_this_project_can_send_an_email():
    """Read every Python file of the project (not the tests) and check what it imports."""
    files = [p for p in PROJECT.rglob("*.py") if "tests" not in p.parts and ".venv" not in str(p)]
    assert len(files) >= 20
    imported = set()
    for path in files:
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                imported.add(name.split(".")[0])
                assert name not in NETWORK_MODULES and name.split(".")[0] not in NETWORK_MODULES, \
                    f"{path.name} imports {name}"
        assert not any(word in source.lower() for word in SEND_WORDS), f"{path.name} mentions sending"
    online = {name for name in imported
              if name in ALLOWED_ONLINE or name in ("chromadb", "crewai_tools", "litellm", "openai", "google")}
    assert online == set(ALLOWED_ONLINE)


def test_the_outreach_crew_has_no_tool_that_can_send():
    import excessive_agency

    assert excessive_agency.part_two() == 0


def test_privacy_comes_first_wherever_crewai_or_a_local_model_loads():
    """CrewAI and ONNX Runtime read their telemetry switches when they start, so privacy.py must come first."""
    import privacy

    assert privacy.SWITCHES["ORT_DISABLE_TELEMETRY"] == "1"        # ONNX Runtime: no usage data, no device id
    assert privacy.SWITCHES["CREWAI_DISABLE_TELEMETRY"] == "true"
    heavy = {"crewai", "crewai_tools", "chromadb", "onnxruntime", "lancedb"}
    files = [p for p in PROJECT.rglob("*.py")
             if not {"tests", "testing"} & set(p.parts) and ".venv" not in str(p)]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = []
        for node in tree.body:                                     # top-level imports, in order
            if isinstance(node, ast.Import):
                names += [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
        first_heavy = next((i for i, name in enumerate(names) if name in heavy), None)
        if first_heavy is not None:
            assert "privacy" in names[:first_heavy], f"{path.name} loads {names[first_heavy]} before privacy"
