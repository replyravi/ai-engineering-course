"""
test_claims.py - Steps 6 and 14: every number traced to a source, the disclaimer stamped by code.
"""

from claims_check import (BANNER, DEMO_LEDGER, DEMO_REPORT, FOOTER, MODEL_END, MODEL_START, advice_phrases,
                          check_report, find_numbers, has_disclaimer, is_supported, stamp_disclaimer,
                          unsupported_dates, unsupported_numbers)
from models import LeadBrief
from replay import scripted_brief
from office_helpers import KESAR
from tools.brief_check import check_against_brief
from tools.calculator import CalculatorTool, calculate


def test_numbers_are_found_but_names_dates_and_list_markers_are_not():
    text = "1. DEMO1 rose 12.34% on 2026-06-15, the 20-day average is 1,250.5 and v1.2 is a name."
    assert find_numbers(text) == ["12.34%", "20", "1,250.5"]


def test_rounding_is_fine_but_a_new_number_is_not():
    assert is_supported("12.3%", [12.34])
    assert is_supported("12%", [12.34])
    assert is_supported("8.2", [-8.21])             # "fell 8.2%" for a tool's -8.21
    assert not is_supported("18.5%", [12.34, 26.82])
    assert unsupported_numbers("rose 12.3%, volatility 18.5%, again 18.5%", [12.34]) == ["18.5%"]


def test_the_planted_number_in_the_demo_is_caught():
    result = check_report(stamp_disclaimer(DEMO_REPORT), DEMO_LEDGER)
    assert result["unsupported_numbers"] == ["18.5%"]
    assert result["numbers_checked"] == 5
    assert result["disclaimer_present"]


def test_a_stale_year_is_caught():
    # the "frozen calendar": a model that thinks it is still an older year
    assert unsupported_dates("As of 2024 the price rose. On 2026-06-15 it fell.", {"2026-06-15", "2026-10-23"}) == ["2024"]


def test_advice_is_flagged_but_the_name_buy_and_hold_is_not():
    assert advice_phrases("Buy-and-hold returned 12.27% on paper. The rule was in the stock 118 days.") == []
    hits = advice_phrases("We recommend it. You should buy now. A price target of 600 and a strong buy rating.")
    assert hits == ["We recommend", "You should buy", "strong buy rating", "strong buy", "price target"]


def test_the_disclaimer_is_stamped_once_and_found():
    stamped = stamp_disclaimer("# DEMO1 note\n\nSome text.")
    assert stamped.splitlines()[:4] == ["# DEMO1 note", "", BANNER, ""]
    assert stamped.rstrip().endswith(FOOTER)
    assert stamp_disclaimer(stamped) == stamped
    assert has_disclaimer(stamped) and not has_disclaimer("# DEMO1 note")


def test_only_the_model_text_is_checked():
    report = f"Written by gemini-3.8-flash\n{MODEL_START}\nIt rose 5%.\n{MODEL_END}\n| 99 |"
    result = check_report(stamp_disclaimer(report), {"numbers": [5.0], "dates": []})
    assert result["numbers_checked"] == 1 and result["unsupported_numbers"] == []


def test_email_numbers_must_be_in_the_brief():
    brief = LeadBrief(**scripted_brief(KESAR))
    ok = check_against_brief("Opened in August 2026. Set-up takes about 15 minutes, a 30-day trial.", brief)
    assert ok["ok"] and ok["numbers_outside_brief"] == []
    bad = check_against_brief("You will see 40% fewer missed pick-ups. Write to nandini@kesar.example", brief)
    assert bad["numbers_outside_brief"] == ["40%"]
    assert bad["links_or_addresses"] == ["nandini@kesar.example"]
    assert not bad["ok"]


def test_the_episode_11_calculator_works_here_and_feeds_the_ledger():
    tool = CalculatorTool()
    assert tool.run(expression="12.27 - 10.86") == "1.41"
    assert 1.41 in [round(n, 2) for n in tool.ledger.numbers]
    assert tool.run(expression="__import__('os')").startswith("Error: only numbers")
    assert calculate("938.40 / 12 * 17") == 1329.4      # the Episode 11 example
