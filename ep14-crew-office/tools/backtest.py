"""
tools/backtest.py - Step 11: a PAPER test of one simple rule, on invented prices.

The rule (a "moving-average crossover"):
    be in the stock while its 20-day average is above its 50-day average, otherwise stay in cash.
The baseline ("buy-and-hold" on paper):
    be in the stock every day.

We only ask: on THIS invented price series, which of the two paper results was higher?
No money, no broker, no orders, no real prices. A rule that did well on one invented year
tells you nothing about any real stock's future.

Two details that honest backtests need, and many tutorials skip:
  1. No peeking. Today's position is decided with YESTERDAY's averages (shift(1)), because
     nobody knows today's closing price before the day ends.
  2. Same start for both. Both start on the first day the 50-day average exists.
  (Still missing on purpose, to keep it small: trading costs and taxes. The report says so.)

    python -m tools.backtest DEMO1        (the tool by hand, no model)

Invented sample data, not real companies or prices. Not investment advice.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import json
import sys
from pathlib import Path

import pandas as pd
from crewai.tools import BaseTool, ToolFailure
from pydantic import BaseModel, ConfigDict, Field

from notebook import Ledger
from tools.indicators import DATA_LABEL, FAST, PRICES_FILE, SLOW, TickerInput, load_prices, pct


def crossover_backtest(close: pd.Series, fast: int = FAST, slow: int = SLOW) -> dict:
    """Paper results of the crossover rule and of buy-and-hold on one price series."""
    close = close.dropna()
    if len(close) <= slow:
        raise ValueError(f"need more than {slow} prices, got {len(close)}")
    fast_avg = close.rolling(fast).mean()
    slow_avg = close.rolling(slow).mean()
    above = fast_avg > slow_avg                           # True on days the short average is higher

    window = close.iloc[slow - 1:]                        # from the first day the slow average exists
    daily = window.pct_change().iloc[1:]                  # each day's change, from day 2 of the window
    in_market = above.iloc[slow - 1:].shift(1).iloc[1:].astype(bool)   # decided at YESTERDAY's close

    rule_growth = (1 + daily.where(in_market, 0.0)).prod()  # cash days earn 0%
    hold_growth = window.iloc[-1] / window.iloc[0]
    switches = int((in_market != in_market.shift(1, fill_value=False)).sum())

    rule_pct, hold_pct = pct(rule_growth - 1), pct(hold_growth - 1)
    return {
        "rule": f"in the stock while the {fast}-day average is above the {slow}-day average, else cash",
        "short_window_days": fast,
        "long_window_days": slow,
        "test_from": str(window.index[0].date()),
        "test_to": str(window.index[-1].date()),
        "test_days": len(daily),
        "rule_return_pct": rule_pct,
        "buy_and_hold_return_pct": hold_pct,
        "difference_pct_points": round(rule_pct - hold_pct, 2),
        "days_in_the_stock": int(in_market.sum()),
        "switches_in_or_out": switches,
        "costs_included": False,
        "label": "paper result on synthetic data, not a forecast and not advice",
    }


def backtest_for(ticker: str, path: Path = PRICES_FILE) -> dict:
    prices = load_prices(path)
    ticker = ticker.strip().upper()
    if ticker not in prices.columns:
        raise KeyError(f"unknown ticker {ticker!r}; the file has {', '.join(prices.columns)}")
    return {"ticker": ticker, "data": DATA_LABEL, **crossover_backtest(prices[ticker])}


class PaperBacktestTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "paper_backtest"
    description: str = (
        "Run a PAPER backtest on the synthetic prices of one invented ticker: the 20/50-day "
        "moving-average crossover rule versus buy-and-hold, same start day, no trading costs. "
        "Returns JSON with both paper returns and their difference. For learning only; call it once."
    )
    args_schema: type[BaseModel] = TickerInput
    ledger: Ledger = Field(default_factory=Ledger)
    prices_file: Path = PRICES_FILE

    def _run(self, ticker: str):
        try:
            result = backtest_for(ticker, self.prices_file)
        except (FileNotFoundError, KeyError, ValueError) as error:
            return ToolFailure(message=str(error).strip('"'))
        self.ledger.record(self.name, result)
        return json.dumps(result)


if __name__ == "__main__":
    try:
        result = backtest_for(sys.argv[1] if len(sys.argv) > 1 else "DEMO1")
    except (FileNotFoundError, KeyError) as error:
        raise SystemExit(str(error).strip('"'))
    print(f"{result['ticker']}  PAPER backtest on synthetic data ({result['data']})")
    print(f"Rule: {result['rule']}")
    print(f"Test: {result['test_from']} to {result['test_to']}, {result['test_days']} days, no trading costs\n")
    print(f"  Crossover rule (paper) ...... {result['rule_return_pct']:+.2f}%")
    print(f"  Buy-and-hold (paper) ........ {result['buy_and_hold_return_pct']:+.2f}%")
    print(f"\n  Difference: {result['difference_pct_points']:+.2f} percentage points.  "
          f"Days in the stock: {result['days_in_the_stock']} of {result['test_days']}.  "
          f"Switches in or out: {result['switches_in_or_out']}.")
    print("\nSynthetic data, paper only: not a forecast, not advice.")
