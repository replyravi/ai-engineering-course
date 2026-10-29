"""
seed_db.py - Step 5: create mandi.db, a small SQLite database of INVENTED vegetable prices.

    python seed_db.py

SAMPLE DATA, NOT REAL PRICES. Every number is made up with a fixed random seed, so every run
(and every laptop) gets exactly the same rows. Do not use them to buy or sell anything.

Two tables, built with SQLAlchemy Core (tables and columns in Python, no SQL text):
    cities        3 rows: Nashik, Pune, Indore and their states
    daily_prices  150 rows: 5 vegetables x 3 cities x 10 sample days
                  price_rs_per_kg     the day's modal price (the price most trading happened at)
                  arrivals_quintals   how much reached the market that day (1 quintal = 100 kg)

This is the ONLY script that writes to the database. The agent's tools open it read-only.
Running it again deletes the old file and builds it fresh.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import random
from pathlib import Path

from sqlalchemy import (Column, Float, ForeignKey, Integer, MetaData, String, Table, create_engine, func,
                        insert, select)
from sqlalchemy.engine import URL

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "mandi.db"

SEED = 15            # same seed -> same "random" prices on every run
DAYS = 10            # sample days 1 to 10 (not calendar dates)

CITIES = [("Nashik", "Maharashtra"), ("Pune", "Maharashtra"), ("Indore", "Madhya Pradesh")]

# Invented starting price (rupees per kg) and daily drift of each vegetable.
# Tomato drifts up about 3 percent a day, so its price clearly rises over the ten days.
VEGETABLES = {
    "onion":   {"start": 22.0, "drift": 0.000},
    "tomato":  {"start": 26.0, "drift": 0.030},
    "potato":  {"start": 18.0, "drift": -0.005},
    "cabbage": {"start": 12.0, "drift": 0.000},
    "brinjal": {"start": 30.0, "drift": 0.010},
}

# Invented: how a city's prices compare with the average, and its usual daily arrivals in quintals.
CITY_PRICE_FACTOR = {"Nashik": 0.95, "Pune": 1.06, "Indore": 0.99}
USUAL_ARRIVALS = {
    "onion":   {"Nashik": 900, "Pune": 420, "Indore": 520},
    "tomato":  {"Nashik": 260, "Pune": 340, "Indore": 300},
    "potato":  {"Nashik": 180, "Pune": 380, "Indore": 610},
    "cabbage": {"Nashik": 150, "Pune": 210, "Indore": 170},
    "brinjal": {"Nashik": 120, "Pune": 160, "Indore": 140},
}

# ---------------------------------------------------------------------------
# 1. Describe the tables in Python: MetaData holds every Table, each Table its Columns
# ---------------------------------------------------------------------------

metadata = MetaData()

cities = Table(
    "cities", metadata,
    Column("city", String(20), primary_key=True),
    Column("state", String(30), nullable=False),
)

daily_prices = Table(
    "daily_prices", metadata,
    Column("id", Integer, primary_key=True),
    Column("day", Integer, nullable=False),                      # 1 to 10, sample days
    Column("city", String(20), ForeignKey("cities.city"), nullable=False),
    Column("vegetable", String(20), nullable=False),             # lowercase: onion, tomato, ...
    Column("price_rs_per_kg", Float, nullable=False),            # invented modal price
    Column("arrivals_quintals", Integer, nullable=False),        # invented
)


# ---------------------------------------------------------------------------
# 2. Make the invented rows (plain Python, easy to test)
# ---------------------------------------------------------------------------

def make_rows(seed=SEED):
    """150 invented rows as dicts: every vegetable, every city, every day."""
    rng = random.Random(seed)
    rows = []
    for vegetable, info in VEGETABLES.items():
        for city, factor in CITY_PRICE_FACTOR.items():
            price = info["start"] * factor * rng.uniform(0.93, 1.07)
            for day in range(1, DAYS + 1):
                if day > 1:   # a small random step each day, plus the vegetable's drift
                    price *= 1 + info["drift"] + rng.uniform(-0.07, 0.07)
                arrivals = USUAL_ARRIVALS[vegetable][city] * rng.uniform(0.8, 1.2)
                rows.append({
                    "day": day,
                    "city": city,
                    "vegetable": vegetable,
                    "price_rs_per_kg": round(price, 1),
                    "arrivals_quintals": int(round(arrivals)),
                })
    return rows


# ---------------------------------------------------------------------------
# 3. Create the file, the tables and the rows
# ---------------------------------------------------------------------------

def sqlite_url(db_path):
    """A SQLAlchemy URL for a SQLite file, built from the path (safe with spaces in folder names)."""
    return URL.create("sqlite", database=str(db_path))


def seed(db_path=DB_PATH, seed=SEED):
    """Build a fresh database. Returns {table name: number of rows}."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()                                   # start fresh every time

    engine = create_engine(sqlite_url(db_path))            # the engine knows how to reach the database
    metadata.create_all(engine)                            # CREATE TABLE for every Table in metadata

    with engine.begin() as conn:                           # one transaction: all rows, or none
        conn.execute(insert(cities), [{"city": c, "state": s} for c, s in CITIES])
        conn.execute(insert(daily_prices), make_rows(seed))  # a list of dicts = many rows at once

    with engine.connect() as conn:
        counts = {table.name: conn.execute(select(func.count()).select_from(table)).scalar_one()
                  for table in (cities, daily_prices)}
    engine.dispose()                                       # close the file (Windows keeps it locked otherwise)
    return counts


def main():
    counts = seed()
    print(f"{DB_PATH.name}, {counts['daily_prices']} rows")
    print(f"  daily_prices  {counts['daily_prices']} rows  "
          f"({len(VEGETABLES)} vegetables x {len(CITIES)} cities x {DAYS} days)")
    print(f"  cities        {counts['cities']} rows")
    print("Sample data, not real prices.")


if __name__ == "__main__":
    main()
