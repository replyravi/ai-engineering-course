"""
claims_check.py - Step 14: every number traced to a tool, and a disclaimer stamped by code.

A model can write "volatility is about 18.5%" in a calm, confident voice, even when no tool
ever said 18.5. So after the research desk writes a report, plain Python checks it:

  1. Every number and date in the parts a MODEL wrote must come from a tool (the Ledger
     saved next to the report). Rounding is fine: 12.3 matches a tool's 12.34.
  2. No sentence may sound like advice ("we recommend", "you should buy", "price target" ...).
  3. The disclaimer must be there. Code puts it there, so no model can forget or soften it.

    python claims_check.py reports/DEMO1.md     check a report the desk wrote (Step 14)
    python claims_check.py --demo              the same check on a report with ONE planted number

The email crew uses the same number finder (tools/brief_check.py): "no number outside the brief".

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The banner and footer our CODE adds to every report. Ravi reviewed the wording once; keep it.
BANNER = "> **Education only. Synthetic data. Not investment advice.**"
FOOTER = ("_DEMO1, DEMO2 and DEMO3 are invented tickers. Their prices and news are invented sample data, "
          "not real companies or prices. This note explains numbers for learning; it does not tell "
          "anyone to buy, sell or hold anything._")

# The checker reads only what a model wrote: the text between these two invisible markers.
MODEL_START = "<!-- model text starts -->"
MODEL_END = "<!-- model text ends -->"

ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)          # "1. " at the start of a line
# A number not glued to a letter or a dot before it ("DEMO1" and "v1.2" are names, not numbers).
NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+−]?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9_])")

ADVICE_PATTERNS = [
    r"\b(?:we|i)\s+(?:would\s+)?(?:recommend|suggest|advise)\b",
    r"\byou\s+(?:should|must|could|might want to)\s+(?:buy|sell|hold|invest|short|exit|add)\b",
    r"\b(?:strong\s+)?(?:buy|sell|hold)\s+(?:rating|signal|call|recommendation)\b",
    r"\bstrong\s+(?:buy|sell)\b",
    r"\bprice\s+target\b",
    r"\btarget\s+price\b",
    r"\bposition\s+siz(?:e|es|ing)\b",
    r"\b(?:go|going)\s+(?:long|short)\b",
    r"\b(?:good|great|right)\s+time\s+to\s+(?:buy|sell|invest)\b",
]


# ---------------------------------------------------------------------------
# 1. Find numbers and dates in text
# ---------------------------------------------------------------------------

def find_dates(text: str) -> list[str]:
    return ISO_DATE.findall(text)


def find_numbers(text: str) -> list[str]:
    """Every number written with digits, as written: '12.34%', '-8.2', '1,250'. Dates are left out."""
    text = ISO_DATE.sub(" ", text)
    text = LIST_MARKER.sub(" ", text)
    return NUMBER.findall(text)


def as_value(number_text: str) -> tuple[float, int]:
    """'−12.34%' -> (12.34, 2): the size of the number (no sign) and how many decimals it shows."""
    clean = number_text.replace(",", "").replace("−", "-").rstrip("%").lstrip("+-")
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return float(clean), decimals


def is_supported(number_text: str, allowed: list[float]) -> bool:
    """True when some tool number, rounded the way the text rounds it, gives this number.

    'fell 8.2%' matches a tool's -8.21 (we compare sizes, so 'fell' and a minus sign both work).
    """
    value, decimals = as_value(number_text)
    return any(round(abs(v), decimals) == value for v in allowed)


def unsupported_numbers(text: str, allowed: list[float]) -> list[str]:
    """The numbers in `text` that no tool (or brief) produced. Each one appears once."""
    found = []
    for number in find_numbers(text):
        if not is_supported(number, allowed) and number not in found:
            found.append(number)
    return found


def unsupported_dates(text: str, allowed_dates: set[str]) -> list[str]:
    """Dates (and stand-alone years like 2024) that no tool returned: the 'frozen calendar' catch."""
    found = [d for d in find_dates(text) if d not in allowed_dates]
    years = {d[:4] for d in allowed_dates}
    found += [y for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", ISO_DATE.sub(" ", text)) if y not in years]
    return sorted(set(found))


def advice_phrases(text: str) -> list[str]:
    """Sentences that sound like advice. A word list is crude, so read each hit yourself."""
    hits = []
    for pattern in ADVICE_PATTERNS:
        hits += [m.group(0) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
    return hits


# ---------------------------------------------------------------------------
# 2. The disclaimer, stamped by code
# ---------------------------------------------------------------------------

def stamp_disclaimer(markdown: str) -> str:
    """Put the banner on top (under the title) and the footer at the end. Running it twice changes nothing."""
    lines = markdown.strip().splitlines()
    if BANNER not in lines:
        at = 1 if lines and lines[0].startswith("# ") else 0
        lines[at:at] = ["", BANNER] + ([""] if at < len(lines) and lines[at].strip() else [])
    text = "\n".join(lines).strip()
    if FOOTER not in text:
        text += "\n\n---\n\n" + FOOTER
    return text + "\n"


def has_disclaimer(markdown: str) -> bool:
    return BANNER in markdown and FOOTER in markdown


# ---------------------------------------------------------------------------
# 3. Check one report against the numbers its tools returned
# ---------------------------------------------------------------------------

def model_text(markdown: str) -> str:
    """Only the parts between the markers: the code-made parts (header, tables, disclaimer) are skipped."""
    parts = re.findall(re.escape(MODEL_START) + r"(.*?)" + re.escape(MODEL_END), markdown, flags=re.DOTALL)
    return "\n".join(parts)


def check_report(markdown: str, ledger: dict) -> dict:
    text = model_text(markdown)
    return {
        "numbers_checked": len(find_numbers(text)),
        "unsupported_numbers": unsupported_numbers(text, ledger["numbers"]),
        "unsupported_dates": unsupported_dates(text, set(ledger["dates"])),
        "advice_phrases": advice_phrases(text),
        "disclaimer_present": has_disclaimer(markdown),
        "model_text_found": bool(text.strip()),
    }


def print_result(result: dict) -> int:
    """Print the check like a scorecard. Returns 0 when everything is fine."""
    print(f"Numbers in the model's text ..... {result['numbers_checked']}")
    print(f"Not from any tool ............... {len(result['unsupported_numbers'])} "
          f"{result['unsupported_numbers'] or ''}".rstrip())
    print(f"Dates or years not from a tool .. {len(result['unsupported_dates'])} "
          f"{result['unsupported_dates'] or ''}".rstrip())
    print(f"Advice phrases .................. {len(result['advice_phrases'])} "
          f"{result['advice_phrases'] or ''}".rstrip())
    print(f"Disclaimer ...................... {'present' if result['disclaimer_present'] else 'MISSING'}")
    problems = (len(result["unsupported_numbers"]) + len(result["unsupported_dates"])
                + len(result["advice_phrases"]) + (not result["disclaimer_present"])
                + (not result["model_text_found"]))
    if not result["model_text_found"]:
        print("PROBLEM: no model text found between the markers. Was this report made by desk_crew.py?")
    disclaimer = "disclaimer present" if result["disclaimer_present"] else "disclaimer MISSING"
    print(f"\nResult: {len(result['unsupported_numbers'])} unsupported numbers, {disclaimer}")
    return 0 if problems == 0 else 1


# ---------------------------------------------------------------------------
# 4. A planted mistake, so you can see the checker catch one
# ---------------------------------------------------------------------------

DEMO_LEDGER = {"numbers": [8.21, 12.34, 20.0, 31.07, 50.0], "dates": ["2026-06-15", "2026-10-23"]}
DEMO_REPORT = f"""# DEMO1 research note (demo)

{MODEL_START}
DEMO1 fell 8.2% on 2026-06-15, the day of an invented recall headline. Over the period it rose 12.34%.
Its 20-day average is above its 50-day average. Volatility is about 18.5% a year, which is moderate.
{MODEL_END}
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check that every number in a report came from a tool.")
    parser.add_argument("report", nargs="?", help="a report made by desk_crew.py, for example reports/DEMO1.md")
    parser.add_argument("--demo", action="store_true", help="check a demo report with one planted number")
    args = parser.parse_args(argv)

    if args.demo:
        print("Demo report (the tools said 8.21, 12.34, 20, 31.07 and 50; the 'model' wrote 18.5 by itself),\n"
              "with the disclaimer our code stamps on every report:\n")
        stamped = stamp_disclaimer(DEMO_REPORT)        # show exactly the text we check
        print(stamped)
        return print_result(check_report(stamped, DEMO_LEDGER))

    if not args.report:
        parser.error("give a report file, for example: python claims_check.py reports/DEMO1.md")
    report = Path(args.report)
    numbers_file = report.with_suffix(".numbers.json")
    if not report.is_file() or not numbers_file.is_file():
        print(f"Missing {report} or {numbers_file}. Run Step 12 first:  python desk_crew.py --ticker DEMO1")
        return 1
    ledger = json.loads(numbers_file.read_text(encoding="utf-8"))
    print(f"Checking {report.as_posix()} against {len(ledger['numbers'])} numbers and "
          f"{len(ledger['dates'])} dates from {numbers_file.as_posix()}\n")
    return print_result(check_report(report.read_text(encoding="utf-8"), ledger))


if __name__ == "__main__":
    sys.exit(main())
