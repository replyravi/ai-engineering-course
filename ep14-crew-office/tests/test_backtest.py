"""
test_backtest.py - Step 11: the paper backtest on tiny price series, checked by hand.
"""

import json

import pytest

from notebook import Ledger
from office_helpers import ensure_prices, series
from tools.backtest import PaperBacktestTool, crossover_backtest


def test_five_prices_by_hand():
    # windows 1 and 2: the 1-day "average" is the price itself.
    # 2-day averages: -, 10.5, 11.5, 11.5, 12   ->  price above average on days 2, 3 and 5
    # The test starts on day 2 (11). In the stock on day 3 and 4 (decided the day before), cash on day 5.
    result = crossover_backtest(series(10, 11, 12, 11, 13), fast=1, slow=2)
    assert result["test_from"] == "2026-01-06"
    assert result["test_days"] == 3
    assert result["rule_return_pct"] == 0.0              # +9.09% on day 3, then -8.33% on day 4: back to 11
    assert result["buy_and_hold_return_pct"] == 18.18    # 13 / 11 - 1
    assert result["difference_pct_points"] == -18.18
    assert result["days_in_the_stock"] == 2
    assert result["switches_in_or_out"] == 2             # in on day 3, out on day 5
    assert result["costs_included"] is False


def test_no_peeking_at_todays_price():
    # The price doubles on the last day. The rule only "sees" the jump at that day's close,
    # so it is still in cash during the jump. A backtest that peeks would claim +100%.
    result = crossover_backtest(series(10, 10, 10, 20), fast=1, slow=2)
    assert result["rule_return_pct"] == 0.0
    assert result["buy_and_hold_return_pct"] == 100.0


def test_too_few_prices_is_an_error():
    with pytest.raises(ValueError, match="need more than 50 prices"):
        crossover_backtest(series(*range(1, 51)))


def test_the_tool_is_labelled_paper_and_synthetic():
    ensure_prices()
    ledger = Ledger()
    result = json.loads(PaperBacktestTool(ledger=ledger).run(ticker="DEMO1"))
    assert result["label"] == "paper result on synthetic data, not a forecast and not advice"
    assert result["data"] == "invented sample data, not real companies or prices"
    assert result["rule_return_pct"] in ledger.numbers and result["buy_and_hold_return_pct"] in ledger.numbers
    assert result["test_days"] == 200
