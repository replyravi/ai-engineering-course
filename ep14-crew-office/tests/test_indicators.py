"""
test_indicators.py - Step 10: the indicator maths on tiny price series, checked by hand.
"""

import json
import math
import statistics

import pytest
from crewai.tools import ToolFailure

import make_prices
from notebook import Ledger
from office_helpers import ensure_prices, series
from tools.indicators import (PriceIndicatorsTool, annual_volatility, compute_indicators, indicators_for,
                              load_prices, max_drawdown)

SIX = series(100, 110, 99, 120, 108, 132)          # 2026-01-05 (Mon) to 2026-01-12 (Mon)


def test_max_drawdown_is_the_worst_fall_from_a_high():
    # 120 -> 90 is a 25% fall; 130 -> 104 is only 20%
    assert max_drawdown(series(100, 120, 90, 130, 104)) == (-25.0, "2026-01-06", "2026-01-07")


def test_volatility_is_daily_spread_times_root_252():
    # returns +10% and -10%; sample standard deviation 0.1414214; x sqrt(252) (15.8745) = 2.244994 -> 224.5%
    assert annual_volatility(series(100, 110, 99)) == 224.5


def test_all_indicators_on_six_prices():
    result = compute_indicators(SIX, fast=2, slow=3)
    daily = [110 / 100 - 1, 99 / 110 - 1, 120 / 99 - 1, 108 / 120 - 1, 132 / 108 - 1]
    assert result["from"] == "2026-01-05" and result["to"] == "2026-01-12"
    assert result["trading_days"] == 6
    assert result["period_return_pct"] == 32.0                      # 132 / 100 - 1
    assert result["return_last_2_days_pct"] == 10.0                 # 132 / 120 - 1
    assert result["average_2_days"] == 120.0                        # (108 + 132) / 2
    assert result["average_3_days"] == 120.0                        # (120 + 108 + 132) / 3
    assert result["volatility_pct_per_year"] == round(statistics.stdev(daily) * math.sqrt(252) * 100, 2)
    assert result["max_drawdown_pct"] == -10.0
    assert result["biggest_up_day"] == {"date": "2026-01-12", "change_pct": 22.22}
    assert result["biggest_down_day"] == {"date": "2026-01-07", "change_pct": -10.0}
    assert result["short_window_days"] == 2 and result["long_window_days"] == 3


def test_the_generator_is_seeded():
    first, _ = make_prices.make_prices()
    second, _ = make_prices.make_prices()
    assert first.equals(second)
    assert len(first) == 750 and first["ticker"].nunique() == 3
    assert first["date"].max() == "2026-10-23"
    assert not first.equals(make_prices.make_prices(seed=15)[0])


def test_the_tool_records_every_number_in_the_ledger():
    ensure_prices()
    ledger = Ledger()
    result = json.loads(PriceIndicatorsTool(ledger=ledger).run(ticker="demo1"))
    assert result["ticker"] == "DEMO1"
    assert result["data"] == "invented sample data, not real companies or prices"
    assert result["volatility_pct_per_year"] in ledger.numbers
    assert {20.0, 50.0, 250.0, 252.0} <= set(ledger.numbers)
    assert result["max_drawdown_low_on"] in ledger.dates
    assert result == indicators_for("DEMO1")


def test_an_unknown_ticker_is_a_clear_failure():
    ensure_prices()
    result = PriceIndicatorsTool().run(ticker="AAPL")
    assert isinstance(result, ToolFailure) and "unknown ticker 'AAPL'" in result.message


def test_no_prices_file_says_how_to_make_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="python make_prices.py"):
        load_prices(tmp_path / "prices.csv")
