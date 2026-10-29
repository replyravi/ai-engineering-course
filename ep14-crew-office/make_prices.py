"""
make_prices.py - Step 9: invented prices for three invented companies, made by a seeded generator.

Why invented? Real prices change every day, belong to data vendors, and tempt people to treat
a lesson as a tip. Invented prices are the same on every computer (same seed, same numbers),
so your table matches the video, and nobody can mistake them for a real stock.

How: each day the price moves by a small random percentage (a "random walk"). A few days get
an extra jump, and news.csv gets an invented headline for that day.

    python make_prices.py        -> data/prices.csv and data/news.csv

Invented sample data, not real companies or prices. Not investment advice.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 14                       # Episode 14: change it and every number changes
LAST_DAY = "2026-10-23"         # the generator's last trading day (a Friday)
DAYS = 250                      # about one year of weekdays (no holidays: it is invented)
NOTE = "# invented sample data, not real companies or prices. Made by make_prices.py, seed 14."

# ticker: (invented name, first price, average daily change, daily wobble)
COMPANIES = {
    "DEMO1": ("Invented Solar Pumps Co.", 420.0, 0.0006, 0.016),
    "DEMO2": ("Invented Tea Trucks Co.", 180.0, 0.0001, 0.011),
    "DEMO3": ("Invented Cloud Kitchens Co.", 95.0, -0.0003, 0.024),
}

# The days that get an extra jump, and the invented headline for each (no numbers in headlines).
EVENTS = [
    ("DEMO1", "2026-03-18", +0.06, "Invented headline: DEMO1 wins a large order for farm pumps"),
    ("DEMO1", "2026-06-15", -0.09, "Invented headline: DEMO1 recalls one pump model for a safety check"),
    ("DEMO2", "2026-08-04", -0.05, "Invented headline: floods slow DEMO2's tea trucks on two highways"),
    ("DEMO3", "2026-02-10", -0.12, "Invented headline: DEMO3 loses its biggest delivery partner"),
    ("DEMO3", "2026-09-21", +0.08, "Invented headline: DEMO3 opens kitchens in two new cities"),
]


def make_prices(seed: int = SEED, last_day: str = LAST_DAY, days: int = DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, news) as tables. Same seed -> exactly the same numbers."""
    dates = pd.bdate_range(end=last_day, periods=days)          # weekdays only
    rng = np.random.default_rng(seed)
    rows = []
    for ticker, (_, first_price, drift, wobble) in COMPANIES.items():
        changes = rng.normal(drift, wobble, size=days)          # one random % change per day
        changes[0] = 0.0                                        # day 1 is the starting price
        for event_ticker, day, jump, _ in EVENTS:
            if event_ticker == ticker:
                changes[dates.get_loc(pd.Timestamp(day))] += jump
        closes = first_price * np.cumprod(1 + changes)
        rows += [{"date": d.date().isoformat(), "ticker": ticker, "close": round(float(c), 2)}
                 for d, c in zip(dates, closes)]
    news = pd.DataFrame([{"date": day, "ticker": t, "headline": text} for t, day, _, text in EVENTS])
    return pd.DataFrame(rows), news.sort_values(["date", "ticker"]).reset_index(drop=True)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(NOTE + "\n")                              # the label travels with the data
        table.to_csv(file, index=False)


def main() -> int:
    prices, news = make_prices()
    write_csv(prices, DATA / "prices.csv")
    write_csv(news, DATA / "news.csv")

    tickers = prices["ticker"].nunique()
    days = prices["date"].nunique()
    print(f"prices.csv, {tickers} tickers x {days} days   ({prices['date'].min()} to {prices['date'].max()})")
    print(f"news.csv, {len(news)} invented headlines")
    print("Invented sample data, not real companies or prices.\n")
    for ticker, (name, *_rest) in COMPANIES.items():
        closes = prices.loc[prices["ticker"] == ticker, "close"]
        print(f"  {ticker}  {name:<28} first close {closes.iloc[0]:>7.2f}   last close {closes.iloc[-1]:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
