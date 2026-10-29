"""
test_files.py - Step 4: the file tool opens only the files the Researcher needs.
"""

import pytest
from crewai.tools import ToolFailure

from notebook import CaseFile, Ledger
from tools.files import data_file_tool, lead_file_tool

LEAD = "leads/kesar-crumb-bakehouse.md"


def test_a_lead_file_comes_back_with_file_and_line_numbers():
    case = CaseFile()
    text = lead_file_tool(case).run(path=LEAD)
    assert "leads/kesar-crumb-bakehouse.md:10: - Recent news: opened the Viman Nagar outlet in August 2026" in text
    assert case.line_at(f"{LEAD}:7") == "- Email: nandini.apte@example.com"
    assert case.run["tool_calls"] == ["read_lead_file"]


def test_the_product_sheet_is_allowed():
    assert "product.md:8: - Free trial: 30 days, no card needed" in lead_file_tool().run(path="product.md")


@pytest.mark.parametrize("path", ["../.env", "leads/../../.env", "/etc/passwd", "llm.py",
                                  "../ep13-shelf-sifter-crew/crew.py", "~/.ssh/id_rsa"])
def test_everything_else_is_refused(path):
    result = lead_file_tool().run(path=path)
    assert isinstance(result, ToolFailure)
    assert result.message.startswith("Refused:")


def test_a_missing_lead_is_a_clear_failure():
    result = lead_file_tool().run(path="leads/no-such-lead.md")
    assert isinstance(result, ToolFailure) and "there is no file" in result.message


def test_the_data_tool_reads_only_the_news_file(tmp_path):
    from office_helpers import ensure_prices

    ensure_prices()
    ledger = Ledger()
    tool = data_file_tool(ledger)
    assert isinstance(tool.run(path="data/prices.csv"), ToolFailure)     # the Analyst gets prices from a tool
    assert isinstance(tool.run(path=LEAD), ToolFailure)
    text = tool.run(path="data/news.csv")
    assert "DEMO1,Invented headline: DEMO1 recalls one pump model for a safety check" in text
    assert "2026-06-15" in ledger.dates
    assert ledger.calls == ["read_data_file"]


def test_a_case_line_that_was_never_read_is_none():
    case = CaseFile()
    assert case.line_at(f"{LEAD}:10") is None             # the tool has not read it yet
    lead_file_tool(case).run(path=LEAD)
    assert case.line_at(f"{LEAD}:99") is None             # no such line
    assert case.line_at("leads/other.md:1") is None
