"""
costing.py - plain Python that does every rupee of maths (Steps 6 and 10).

The model writes the menu and the grocery list. It never adds anything up.
This file:
  - reads YOUR prices from data/prices.csv              load_prices()
  - scales "grams per guest" up to the whole party      scale_and_cost()
  - adds up the bill exactly, to the paisa              (Decimal, not float)
  - decides if the bill fits the budget                 budget_check()

No model, no key, no internet: that is why these numbers are exact and why
tests/test_costing.py can check every one of them.

Run:  python costing.py                    (Step 6: the sample list for 12 guests)
      python costing.py --guests 20        (the same list for another party size)
      python costing.py --budget 800       (Step 10: what "over budget" looks like)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import csv
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
PRICES_FILE = HERE / "data" / "prices.csv"

PAISA = Decimal("0.01")          # money is rounded to the paisa, once, per line
SHOP_UNITS = ("kg", "litre", "piece")

# The model thinks in grams and millilitres per guest; the shop sells kilos and litres.
UNIT_TO_SHOP = {"g": ("kg", 1000), "ml": ("litre", 1000), "piece": ("piece", 1)}


# ---------------------------------------------------------------------------
# 1. The typed form the model fills in (Step 5) and Python reads (Step 6)
# ---------------------------------------------------------------------------

class Ingredient(BaseModel):
    """One grocery item, as the amount ONE guest needs."""

    item: str = Field(description="The item name, copied exactly from the price list")
    qty_per_person: float = Field(description="How much ONE guest needs, in the unit below")
    unit: Literal["g", "ml", "piece"] = Field(
        description="g for solids, ml for liquids, piece for things you count")


class Ingredients(BaseModel):
    """The grocery list for the whole menu."""

    items: list[Ingredient] = Field(description="Every item the menu needs, one entry per item")


# A grocery list written BY HAND (not by a model), so Step 6 runs without a key.
# It is what a South Indian lunch of sambar rice, beans poriyal, rasam and
# semiya payasam might need. 'asafoetida' is in it on purpose: it has no price yet.
SAMPLE_ITEMS = [
    Ingredient(item="rice", qty_per_person=120, unit="g"),
    Ingredient(item="toor dal", qty_per_person=35, unit="g"),
    Ingredient(item="tamarind", qty_per_person=8, unit="g"),
    Ingredient(item="sambar powder", qty_per_person=6, unit="g"),
    Ingredient(item="rasam powder", qty_per_person=4, unit="g"),
    Ingredient(item="tomato", qty_per_person=70, unit="g"),
    Ingredient(item="drumstick", qty_per_person=40, unit="g"),
    Ingredient(item="beans", qty_per_person=60, unit="g"),
    Ingredient(item="coconut", qty_per_person=0.15, unit="piece"),
    Ingredient(item="coconut oil", qty_per_person=12, unit="ml"),
    Ingredient(item="curd", qty_per_person=80, unit="g"),
    Ingredient(item="vermicelli", qty_per_person=20, unit="g"),
    Ingredient(item="milk", qty_per_person=120, unit="ml"),
    Ingredient(item="jaggery", qty_per_person=25, unit="g"),
    Ingredient(item="cashew nuts", qty_per_person=5, unit="g"),
    Ingredient(item="ghee", qty_per_person=8, unit="g"),
    Ingredient(item="curry leaves bunch", qty_per_person=0.1, unit="piece"),
    Ingredient(item="banana leaf", qty_per_person=1, unit="piece"),
    Ingredient(item="papad", qty_per_person=1, unit="piece"),
    Ingredient(item="asafoetida", qty_per_person=0.5, unit="g"),
]


# ---------------------------------------------------------------------------
# 2. Read the price list (YOUR prices: edit data/prices.csv)
# ---------------------------------------------------------------------------

@dataclass
class Price:
    item: str
    unit: str          # the shop unit: kg, litre or piece
    rupees: Decimal    # the price of ONE shop unit


class PriceFileError(ValueError):
    """data/prices.csv has a problem. The message names the line."""


class EmptyListError(ValueError):
    """The model did not fill in the Ingredients form (it sent no grocery list)."""


def load_prices(path=None) -> dict[str, Price]:
    """Read the price list into {item name in small letters: Price}.

    Lines starting with # are notes and are skipped. A bad line stops with a clear message.
    """
    path = Path(path or PRICES_FILE)
    # "utf-8-sig" also reads files saved by Excel as "CSV UTF-8", which start with an invisible mark (BOM)
    with open(path, newline="", encoding="utf-8-sig") as file:
        # keep the real line numbers, so an error message points at the right line
        numbered = [(n, line) for n, line in enumerate(file, start=1)
                    if line.strip() and not line.lstrip().startswith("#")]
    if not numbered:
        raise PriceFileError(f"{path.name} is empty.")

    for line_number, line in numbered:
        if ";" in line and "," not in line:     # Excel in some countries saves the whole file like this
            raise PriceFileError(f"{path.name} line {line_number}: the values are separated by ';'. "
                                 "Use commas (in Excel: save as 'CSV UTF-8 (comma delimited)').")

    reader = csv.DictReader(line for _, line in numbered)
    needed = {"item", "unit", "price_rupees"}
    if not needed.issubset(reader.fieldnames or []):
        raise PriceFileError(f"{path.name} needs the columns item,unit,price_rupees "
                             f"(found: {','.join(reader.fieldnames or [])}).")

    prices = {}
    for (line_number, line), row in zip(numbered[1:], reader):
        name = " ".join((row["item"] or "").lower().split())
        unit = (row["unit"] or "").strip().lower()
        if not name:
            raise PriceFileError(f"{path.name} line {line_number}: the item name is empty.")
        if unit not in SHOP_UNITS:
            raise PriceFileError(f"{path.name} line {line_number}: unit '{unit}' must be kg, litre or piece.")
        try:
            rupees = Decimal((row["price_rupees"] or "").strip())
        except InvalidOperation:
            raise PriceFileError(f"{path.name} line {line_number}: price '{row['price_rupees']}' "
                                 "is not a number.") from None
        if not rupees.is_finite() or rupees <= 0:
            raise PriceFileError(f"{path.name} line {line_number}: the price must be above zero.")
        if name in prices:
            raise PriceFileError(f"{path.name} line {line_number}: '{name}' is listed twice.")
        prices[name] = Price(item=name, unit=unit, rupees=rupees)
    return prices


def find_price(name: str, prices: dict[str, Price]) -> Price | None:
    """The price of an item, also when the model writes a plural ('tomatoes')."""
    key = " ".join(name.lower().split())
    for candidate in (key, key.removesuffix("s"), key.removesuffix("es")):
        if candidate in prices:
            return prices[candidate]
    return None


# ---------------------------------------------------------------------------
# 3. Scale to the party and cost it (Step 6)
# ---------------------------------------------------------------------------

@dataclass
class Line:
    item: str
    per_guest: Decimal   # in the model's unit: g, ml or piece
    unit: str            # g, ml or piece
    to_buy: Decimal      # in the shop unit
    shop_unit: str       # kg, litre or piece
    price: Decimal       # rupees for one shop unit
    cost: Decimal        # rupees, to the paisa


@dataclass
class Bill:
    guests: int
    lines: list[Line]
    not_costed: list[str]   # items we could not price, with the reason
    total: Decimal          # the sum of the line costs, exact


def scale_and_cost(items, guests: int, prices: dict[str, Price]) -> Bill:
    """Turn 'per guest' amounts into a costed shopping list for the whole party.

    items:  an Ingredients object (from the model) or a list of Ingredient
    guests: how many people eat
    prices: from load_prices()
    """
    if items is None:
        raise EmptyListError("the model sent no grocery list.")
    if isinstance(items, Ingredients):
        items = items.items
    if guests < 1:
        raise ValueError("guests must be at least 1")

    # The same item can appear twice (tomato for sambar AND rasam): add those up first.
    needed: dict[str, list] = {}           # price item -> [Price, unit, per-guest amount]
    not_costed = []
    for ingredient in items:
        price = find_price(ingredient.item, prices)
        per_guest = Decimal(str(ingredient.qty_per_person))   # str() keeps 0.15 as 0.15
        if price is None:
            not_costed.append(f"{ingredient.item} (no price in data/prices.csv)")
        elif UNIT_TO_SHOP[ingredient.unit][0] != price.unit:
            not_costed.append(f"{ingredient.item} (the list says {ingredient.unit}, "
                              f"prices.csv sells it per {price.unit})")
        elif per_guest <= 0:
            not_costed.append(f"{ingredient.item} (amount {per_guest} is not above zero)")
        elif price.item in needed:
            needed[price.item][2] += per_guest
        else:
            needed[price.item] = [price, ingredient.unit, per_guest]

    lines = []
    for item, (price, unit, per_guest) in needed.items():
        shop_unit, units_per_shop_unit = UNIT_TO_SHOP[unit]
        to_buy = per_guest * guests / units_per_shop_unit
        if shop_unit == "piece":
            to_buy = Decimal(math.ceil(to_buy))      # nobody sells 1.8 coconuts: round UP
        cost = (to_buy * price.rupees).quantize(PAISA, rounding=ROUND_HALF_UP)
        lines.append(Line(item, per_guest, unit, to_buy, shop_unit, price.rupees, cost))

    total = sum((line.cost for line in lines), Decimal("0"))
    return Bill(guests=guests, lines=lines, not_costed=not_costed, total=total)


# ---------------------------------------------------------------------------
# 4. The budget check: Python decides (Step 10)
# ---------------------------------------------------------------------------

@dataclass
class BudgetCheck:
    budget: Decimal
    total: Decimal
    over_by: Decimal     # 0 when the bill fits
    left: Decimal        # 0 when the bill is over

    @property
    def is_over(self) -> bool:
        return self.over_by > 0

    def message(self) -> str:
        if self.is_over:
            return (f"Over budget by {rupees(self.over_by)} "
                    f"(bill {rupees(self.total)}, budget {rupees(self.budget, whole=True)}).")
        if self.left == 0:
            return f"Exactly on budget: {rupees(self.total)}."
        return (f"Within budget: {rupees(self.left)} left "
                f"(bill {rupees(self.total)}, budget {rupees(self.budget, whole=True)}).")


def budget_check(total, budget) -> BudgetCheck:
    """Compare the bill with the budget. Plain subtraction, exact to the paisa."""
    total, budget = Decimal(total), Decimal(budget)
    difference = total - budget
    return BudgetCheck(budget=budget, total=total,
                       over_by=max(difference, Decimal("0")),
                       left=max(-difference, Decimal("0")))


# ---------------------------------------------------------------------------
# 5. Pipeline steps: plain functions that read and write the running dict.
#    party.py wraps them in RunnableLambda (the old TransformChain).
# ---------------------------------------------------------------------------

def cost_step(state: dict) -> Bill:
    """Read 'items' and 'guests' from the running dict, return the bill."""
    return scale_and_cost(state["items"], state["guests"], load_prices())


def budget_step(state: dict) -> BudgetCheck:
    """Read 'bill' and 'budget' from the running dict, return the check."""
    return budget_check(state["bill"].total, state["budget"])


# ---------------------------------------------------------------------------
# 6. Printing
# ---------------------------------------------------------------------------

def rupees(amount, whole: bool = False) -> str:
    """Rs 1,234.56 (whole=True drops the paise when there are none: Rs 6,000)."""
    amount = Decimal(amount)
    if whole and amount == amount.to_integral_value():
        return f"Rs {amount:,.0f}"
    return f"Rs {amount:,.2f}"


def amount(value: Decimal, unit: str) -> str:
    """120 g, 1.44 kg, 0.15 piece, 2 pieces."""
    text = f"{value.normalize():f}"
    if unit == "piece" and value > 1:
        unit = "pieces"
    return f"{text} {unit}"


def print_bill(bill: Bill) -> None:
    print(f"Shopping list for {bill.guests} guests (prices: data/prices.csv, SAMPLE prices)")
    print(f"{'Item':<20} {'Per guest':<12} {'To buy':<12} {'Price':<18} {'Cost':>12}")
    for line in bill.lines:
        price = f"{rupees(line.price, whole=True)} / {line.shop_unit}"
        print(f"{line.item:<20} {amount(line.per_guest, line.unit):<12} "
              f"{amount(line.to_buy, line.shop_unit):<12} {price:<18} {rupees(line.cost):>12}")
    print("-" * 78)
    print(f"{'Total for ' + str(len(bill.lines)) + ' items':<64} {rupees(bill.total):>13}")
    for reason in bill.not_costed:
        print(f"Not costed: {reason}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scale, cost and budget-check a grocery list (no model).")
    parser.add_argument("--guests", type=int, default=12)
    parser.add_argument("--budget", type=int, default=6000, help="in rupees")
    args = parser.parse_args(argv)

    try:
        prices = load_prices()
    except (OSError, PriceFileError) as error:
        print(f"Could not read the price list: {error}")
        return 1
    print(f"Price list: {len(prices)} items from data/prices.csv\n")

    bill = scale_and_cost(SAMPLE_ITEMS, args.guests, prices)   # the sample list, written by hand
    print_bill(bill)

    # The SAME function as a pipeline step. RunnableLambda wraps any Python function,
    # so it gets invoke / batch / stream like every other step (Step 6).
    from langchain_core.runnables import RunnableLambda

    step = RunnableLambda(cost_step)
    same_bill = step.invoke({"items": SAMPLE_ITEMS, "guests": args.guests})
    print(f"\nRunnableLambda(cost_step).invoke(...) -> total {rupees(same_bill.total)} (same function, now a step)")

    print("\nBudget check (Python decides):")
    print("  " + budget_check(bill.total, args.budget).message())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
