"""
safety_gate.py - a small safety layer in front of the model (steps 10, 11 and 13).

Each piece answers one risk from the OWASP Top 10 for LLM Applications, 2025 edition
(OWASP GenAI Security Project, genai.owasp.org, content licensed CC BY-SA 4.0):

    redact_pii()        LLM02 Sensitive Information Disclosure   step 10
    check_length()      LLM10 Unbounded Consumption              step 11
    RateLimiter         LLM10 Unbounded Consumption              step 11
    escape_for_html()   LLM05 Improper Output Handling           step 13
    (injection_demo.py  LLM01 Prompt Injection                   step 12)

All personal details in this file are invented: example.com is a reserved demo domain,
and +91 00000 12345 cannot belong to anyone (no real Indian number starts with 0 after +91).

Run:  python safety_gate.py               (the three offline demos, no key, no model call)
      python safety_gate.py redact        (step 10)
      python safety_gate.py limits        (step 11)
      python safety_gate.py escape        (step 13)
      python safety_gate.py ask "your message"    (one guarded call to the real model)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import argparse
import html
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "logs" / "gate.log"            # git ignores logs/
REPORTS_DIR = HERE / "reports"

MAX_INPUT_CHARS = 1500     # longer messages are refused before they cost anything
MAX_CALLS = 5              # at most 5 model calls ...
PER_SECONDS = 60           # ... per 60 seconds

# Patterns for two kinds of personal data. Patterns only find patterns: names and addresses slip through.
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# A 10-digit phone number written the Indian way: maybe after +91 or 0, maybe split 5 + 5.
PHONE = re.compile(r"(?<![\d+])(?:\+91[\s-]?|0)?\d{5}[\s-]?\d{5}(?!\d)")

SAMPLE_MESSAGE = ("Hi, I am Ananya. My order has not arrived yet. "
                  "Email me at ananya@example.com or call +91 00000 12345.")

ASSISTANT_SYSTEM = ("You help customers of a small online shop. Placeholders such as [EMAIL] and [PHONE] "
                    "stand for details we removed on purpose. Never ask the customer to type them again.")


# ---------------------------------------------------------------------------
# 1. Redact personal data before it leaves your computer (step 10)
# ---------------------------------------------------------------------------

def redact_pii(text: str) -> tuple[str, dict]:
    """Replace email addresses and phone numbers with placeholders. Returns (clean text, counts)."""
    text, emails = EMAIL.subn("[EMAIL]", text)
    text, phones = PHONE.subn("[PHONE]", text)
    return text, {"email": emails, "phone": phones}


# ---------------------------------------------------------------------------
# 2. Limits: message length and calls per minute (step 11)
# ---------------------------------------------------------------------------

def check_length(text: str, max_chars: int = MAX_INPUT_CHARS) -> str | None:
    """A friendly refusal for a message that is too long, or None when it is fine."""
    if len(text) <= max_chars:
        return None
    return (f"Sorry, that message is too long ({len(text):,} characters). "
            f"Please keep it under {max_chars:,} characters, or send it in parts.")


class RateLimiter:
    """Allow at most `max_calls` calls in any window of `per_seconds` seconds (a sliding window)."""

    def __init__(self, max_calls: int = MAX_CALLS, per_seconds: float = PER_SECONDS, clock=time.monotonic):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self.clock = clock               # tests pass a fake clock, so they never wait
        self.calls = deque()             # the times of the recent calls

    def try_call(self) -> tuple[bool, float]:
        """(True, 0) if a call is allowed now, else (False, seconds to wait)."""
        now = self.clock()
        while self.calls and now - self.calls[0] >= self.per_seconds:
            self.calls.popleft()         # forget calls that left the window
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True, 0.0
        if not self.calls:               # max_calls = 0: nothing is ever allowed
            return False, float(self.per_seconds)
        return False, self.per_seconds - (now - self.calls[0])


# ---------------------------------------------------------------------------
# 3. The gate: every message goes through it before the model sees it
# ---------------------------------------------------------------------------

@dataclass
class GateDecision:
    allowed: bool
    text: str                     # what the model may see (already redacted)
    message: str = ""             # the friendly refusal, when not allowed
    found: dict = field(default_factory=dict)


def guard(text: str, limiter: RateLimiter, max_chars: int = MAX_INPUT_CHARS) -> GateDecision:
    """Length first, then the rate limit, then redaction."""
    too_long = check_length(text, max_chars)
    if too_long:
        return GateDecision(False, "", too_long)
    allowed, wait = limiter.try_call()
    if not allowed:
        return GateDecision(False, "", f"Slow down please: the limit is {limiter.max_calls} messages per "
                                       f"{limiter.per_seconds:g} seconds. Try again in {math.ceil(wait)} seconds.")
    clean, found = redact_pii(text)
    return GateDecision(True, clean, found=found)


def log_prompt(text: str, path=LOG_FILE) -> str:
    """Append the prompt (already redacted) to the log file. Returns the line written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


def ask_safely(text: str, llm, limiter: RateLimiter, log_path=LOG_FILE) -> str:
    """Gate -> log -> model. The model and the log only ever see the redacted text."""
    decision = guard(text, limiter)
    if not decision.allowed:
        return decision.message
    log_prompt(decision.text, log_path)
    reply = llm.invoke([SystemMessage(ASSISTANT_SYSTEM), HumanMessage(decision.text)])
    return reply.text


# ---------------------------------------------------------------------------
# 4. Escape model output before it goes into a web page (step 13)
# ---------------------------------------------------------------------------

def escape_for_html(text: str) -> str:
    """Turn < > & " ' into HTML entities, so the browser SHOWS them instead of running them."""
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_redact(log_path=LOG_FILE) -> None:
    print("Step 10 - Redact personal details before they leave your computer\n")
    print(f"What the customer typed:\n  {SAMPLE_MESSAGE}\n")
    decision = guard(SAMPLE_MESSAGE, RateLimiter())
    print(f"What the model (and the log file) will see:\n  {decision.text}\n")
    print(f"Found: {decision.found['email']} email address, {decision.found['phone']} phone number")
    line = log_prompt(decision.text, log_path)
    print(f"Written to {Path(log_path).parent.name}/{Path(log_path).name}:\n  {line}")
    print("\nNot caught: the name 'Ananya'. Patterns find emails and numbers, not names or addresses.")


def demo_limits(clock=time.monotonic) -> None:
    print("Step 11 - Cap the length and the calls per minute\n")
    limiter = RateLimiter(clock=clock)
    pasted = "Please summarise this: " + "lorem ipsum " * 200
    print(f"1. A {len(pasted):,}-character message (someone pasted a whole document):")
    print(f"   {guard(pasted, limiter).message}\n")
    print(f"2. Six quick messages, with a limit of {limiter.max_calls} per {limiter.per_seconds:g} seconds:")
    for number in range(1, 7):
        decision = guard(f"Question number {number}", limiter)
        print(f"   message {number}: " + ("allowed" if decision.allowed else decision.message))
    print("\nNo model was called in this demo, so it cost nothing.")


def demo_escape(reports_dir=REPORTS_DIR) -> None:
    print("Step 13 - Escape model output before it goes into a web page\n")
    answer = ("Great lunch plan for busy weeks! <script>alert('this should never run')</script> "
              "<img src=x onerror=alert(1)>")
    print(f"A pretend model answer (we wrote it) that repeats tags it read on a web page:\n  {answer}\n")
    safe = escape_for_html(answer)
    print(f"After escape_for_html():\n  {safe}\n")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    page = ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>Escaping demo</title></head><body>\n"
            "<h1>Model answer</h1>\n<p>" + safe + "</p>\n</body></html>\n")
    (reports_dir / "escape_demo.html").write_text(page, encoding="utf-8")
    print("Saved reports/escape_demo.html. Open it in a browser: the tags show as text, and nothing runs.")


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="Safety gate demos.")
    parser.add_argument("demo", nargs="?", choices=["redact", "limits", "escape", "ask"], help="which demo (default: the three offline ones)")
    parser.add_argument("message", nargs="?", default=SAMPLE_MESSAGE, help="for ask: the message to send")
    args = parser.parse_args(argv)

    if args.demo == "ask":
        from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

        try:
            provider = get_provider()
            if llm is None:
                llm = get_llm()
        except (MissingAPIKeyError, ValueError) as error:
            print(f"\n{str(error).strip()}")
            return 1
        print(f"Provider: {provider}   Model: {get_model_name(provider)}")
        print(f"You typed:         {args.message}")
        print(f"The model gets:    {redact_pii(args.message)[0]}")
        try:
            print(f"Reply: {ask_safely(args.message, llm, RateLimiter())}")
        except Exception as error:  # no internet, the free-tier limit...
            print(f"The model call failed: {error}")
            return 1
        return 0

    demos = {"redact": demo_redact, "limits": demo_limits, "escape": demo_escape}
    chosen = [args.demo] if args.demo else list(demos)
    for number, name in enumerate(chosen):
        if number:
            print("\n" + "-" * 72 + "\n")
        demos[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
