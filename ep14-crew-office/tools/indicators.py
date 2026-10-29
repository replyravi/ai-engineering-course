"""
tools/indicators.py - Step 10: the numbers of one ticker, computed by pandas, never by a model.

A cricket commentator reads the scoreboard; he does not guess the run rate. Our Analyst is
the commentator, this tool is the scoreboard. For one invented ticker it computes:

  return over the whole period and over the last 20 days   (how much the price changed, in %)
  20-day and 50-day moving averages                        (the average close of the last 20 or 50 days)
  volatility                                               (how much it jumps around, in % per year)
  max drawdown                                             (the worst fall from a high point, in %)
  the biggest up day and the biggest down day

Every value is rounded to 2 decimals, and every number the tool returns is written into the
Ledger, so the claims checker (Step 14) can trace each number in the report back to here.

Invented sample data, not real companies or prices. Not investment advice.

    python -m tools.indicators DEMO1        (the tool by hand: a small table, no model)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import json
import math
import sys
from pathlib import Path

import pandas as pd
from crewai.tools import BaseTool, ToolFailure
from pydantic import BaseModel, ConfigDict, Field

from notebook import Ledger

PROJECT = Path(__file__).resolve().parent.parent
PRICES_FILE = PROJECT / "data" / "prices.csv"
DATA_LABEL = "invented sample data, not real companies or prices"
FAST, SLOW = 20, 50              # moving-average windows, in trading days
TRADING_DAYS_PER_YEAR = 252      # a common convention for turning daily volatility into yearly


def load_prices(path: Path = PRICES_FILE) -> pd.DataFrame:
    """prices.csv (date, ticker, close) -> one column of closing prices per ticker, one row per day."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{Path(path).name} not found. Make it first:  python make_prices.py")
    rows = pd.read_csv(path, comment="#", parse_dates=["date"])      # lines starting with # are notes
    return rows.pivot(index="date", columns="ticker", values="close").sort_index()


def pct(value: float) -> float:
    """0.1234 -> 12.34 (a fraction as a percentage, 2 decimals)."""
    return round(float(value) * 100, 2)


def max_drawdown(close: pd.Series) -> tuple[float, str, str]:
    """The worst fall from a previous high, in %, with the dates of that high and of the low."""
    running_high = close.cummax()
    drawdown = close / running_high - 1                 # 0 at a new high, negative below it
    trough = drawdown.idxmin()
    peak = close.loc[:trough].idxmax()
    return pct(drawdown.min()), str(peak.date()), str(trough.date())


def annual_volatility(close: pd.Series) -> float:
    """Standard deviation of daily returns x square root of 252, in %."""
    return pct(close.pct_change().std() * math.sqrt(TRADING_DAYS_PER_YEAR))


def compute_indicators(close: pd.Series, fast: int = FAST, slow: int = SLOW) -> dict:
    """All indicators for one price series (a date index and closing prices)."""
    close = close.dropna()
    daily = close.pct_change()
    drawdown, peak, trough = max_drawdown(close)
    best, worst = daily.idxmax(), daily.idxmin()
    return {
        "from": str(close.index[0].date()),
        "to": str(close.index[-1].date()),
        "trading_days": len(close),
        "short_window_days": fast,
        "long_window_days": slow,
        "first_close": round(float(close.iloc[0]), 2),
        "last_close": round(float(close.iloc[-1]), 2),
        "period_return_pct": pct(close.iloc[-1] / close.iloc[0] - 1),
        f"return_last_{fast}_days_pct": pct(close.iloc[-1] / close.iloc[-1 - fast] - 1),
        f"average_{fast}_days": round(float(close.iloc[-fast:].mean()), 2),
        f"average_{slow}_days": round(float(close.iloc[-slow:].mean()), 2),
        "volatility_pct_per_year": annual_volatility(close),
        "trading_days_per_year_used": TRADING_DAYS_PER_YEAR,
        "max_drawdown_pct": drawdown,
        "max_drawdown_from_high_on": peak,
        "max_drawdown_low_on": trough,
        "biggest_up_day": {"date": str(best.date()), "change_pct": pct(daily.max())},
        "biggest_down_day": {"date": str(worst.date()), "change_pct": pct(daily.min())},
    }


def indicators_for(ticker: str, path: Path = PRICES_FILE) -> dict:
    prices = load_prices(path)
    ticker = ticker.strip().upper()
    if ticker not in prices.columns:
        raise KeyError(f"unknown ticker {ticker!r}; the file has {', '.join(prices.columns)}")
    return {"ticker": ticker, "data": DATA_LABEL, **compute_indicators(prices[ticker])}


class TickerInput(BaseModel):
    ticker: str = Field(description="An invented ticker from data/prices.csv: DEMO1, DEMO2 or DEMO3")


class PriceIndicatorsTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "price_indicators"
    description: str = (
        "Compute price indicators for one invented ticker from the synthetic file data/prices.csv: "
        "period return, 20-day return, 20- and 50-day averages, yearly volatility, max drawdown and "
        "the biggest up and down days. Returns JSON. Quote its numbers exactly; call it once per ticker."
    )
    args_schema: type[BaseModel] = TickerInput
    ledger: Ledger = Field(default_factory=Ledger)
    prices_file: Path = PRICES_FILE

    def _run(self, ticker: str):
        try:
            result = indicators_for(ticker, self.prices_file)
        except (FileNotFoundError, KeyError) as error:
            return ToolFailure(message=str(error).strip('"'))
        self.ledger.record(self.name, result)
        return json.dumps(result)


def print_table(result: dict) -> None:
    print(f"{result['ticker']}  ({result['data']})")
    print(f"{result['from']} to {result['to']}, {result['trading_days']} trading days\n")
    for key, value in result.items():
        if key in ("ticker", "data", "from", "to", "trading_days", "short_window_days", "long_window_days"):
            continue
        if isinstance(value, dict):
            value = f"{value['change_pct']:+.2f}% on {value['date']}"
        print(f"  {key + ' ':.<34} {value}")


if __name__ == "__main__":
    try:
        print_table(indicators_for(sys.argv[1] if len(sys.argv) > 1 else "DEMO1"))
    except (FileNotFoundError, KeyError) as error:
        raise SystemExit(str(error).strip('"'))
