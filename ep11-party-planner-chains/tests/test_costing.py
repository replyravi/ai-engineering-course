"""
tests/test_costing.py - the plain Python maths (Steps 6 and 10). No model at all.

Every number here is worked out by hand, so a test fails the moment the maths changes.
"""

from decimal import Decimal

import pytest

import costing
from costing import (SAMPLE_ITEMS, Ingredient, Ingredients, PriceFileError, budget_check, cost_step,
                     find_price, load_prices, rupees, scale_and_cost)
from party_fixtures import ITEMS, small_prices  # noqa: F401


# ---------------------------------------------------------------------------
# Reading prices.csv
# ---------------------------------------------------------------------------

def test_the_course_price_list_loads():
    prices = load_prices()
    assert len(prices) == 24
    assert prices["rice"].unit == "kg" and prices["rice"].rupees == Decimal("64")
    assert prices["banana leaf"].unit == "piece"


def test_note_lines_are_skipped(small_prices):
    prices = load_prices(small_prices)
    assert list(prices) == ["rice", "curd", "coconut", "milk"]
    # Excel's "CSV UTF-8" puts an invisible mark (BOM) before the first note line: still skipped
    small_prices.write_text(small_prices.read_text(encoding="utf-8"), encoding="utf-8-sig")
    assert small_prices.read_bytes().startswith(b"\xef\xbb\xbf")
    assert list(load_prices(small_prices)) == ["rice", "curd", "coconut", "milk"]


@pytest.mark.parametrize("bad_row, message", [
    ("rice,kgs,60", "line 3: unit 'kgs' must be kg, litre or piece"),
    ("rice,kg,sixty", "line 3: price 'sixty' is not a number"),
    ("rice,kg,0", "line 3: the price must be above zero"),
    (",kg,10", "line 3: the item name is empty"),
])
def test_a_bad_line_is_named(tmp_path, bad_row, message):
    path = tmp_path / "prices.csv"
    path.write_text(f"# note\nitem,unit,price_rupees\n{bad_row}\n", encoding="utf-8")
    with pytest.raises(PriceFileError, match=message):
        load_prices(path)


def test_semicolons_get_a_clear_message(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("item,unit,price_rupees\nrice;kg;60\n", encoding="utf-8")
    with pytest.raises(PriceFileError, match="line 2: the values are separated by ';'"):
        load_prices(path)
    # the usual case: Excel saved the WHOLE file with semicolons, the header too
    path.write_text("# note\nitem;unit;price_rupees\nrice;kg;60\n", encoding="utf-8")
    with pytest.raises(PriceFileError, match="line 2: the values are separated by ';'"):
        load_prices(path)


def test_a_duplicate_item_is_refused(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("item,unit,price_rupees\nrice,kg,60\nRice,kg,62\n", encoding="utf-8")
    with pytest.raises(PriceFileError, match="line 3: 'rice' is listed twice"):
        load_prices(path)


def test_wrong_columns_are_refused(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("name,price\nrice,60\n", encoding="utf-8")
    with pytest.raises(PriceFileError, match="needs the columns item,unit,price_rupees"):
        load_prices(path)


def test_plural_names_still_find_their_price(small_prices):
    prices = load_prices(small_prices)
    assert find_price("Coconuts", prices).item == "coconut"
    assert find_price("  CURD ", prices).item == "curd"
    assert find_price("saffron", prices) is None


# ---------------------------------------------------------------------------
# scale_and_cost
# ---------------------------------------------------------------------------

def test_scale_and_cost_on_fixed_inputs(small_prices):
    bill = scale_and_cost(ITEMS, 12, load_prices(small_prices))
    rice, curd, coconut = bill.lines
    assert (rice.to_buy, rice.shop_unit, rice.cost) == (Decimal("1.2"), "kg", Decimal("72.00"))
    assert (curd.to_buy, curd.cost) == (Decimal("0.6"), Decimal("42.00"))
    assert (coconut.to_buy, coconut.cost) == (Decimal("3"), Decimal("105.00"))   # 0.25 x 12 = 3
    assert bill.total == Decimal("219.00")
    assert bill.not_costed == []


def test_pieces_are_rounded_up(small_prices):
    items = [Ingredient(item="coconut", qty_per_person=0.15, unit="piece")]
    bill = scale_and_cost(items, 12, load_prices(small_prices))      # 1.8 coconuts -> buy 2
    assert bill.lines[0].to_buy == Decimal("2")
    assert bill.total == Decimal("70.00")


def test_unknown_items_and_wrong_units_are_reported_not_guessed(small_prices):
    items = [Ingredient(item="saffron", qty_per_person=0.1, unit="g"),
             Ingredient(item="milk", qty_per_person=100, unit="g"),        # milk is sold per litre
             Ingredient(item="rice", qty_per_person=0, unit="g"),
             Ingredient(item="curd", qty_per_person=50, unit="g")]
    bill = scale_and_cost(items, 10, load_prices(small_prices))
    assert [line.item for line in bill.lines] == ["curd"]
    assert bill.not_costed == [
        "saffron (no price in data/prices.csv)",
        "milk (the list says g, prices.csv sells it per litre)",
        "rice (amount 0.0 is not above zero)",
    ]
    assert bill.total == Decimal("35.00")


def test_the_same_item_twice_is_added_up(small_prices):
    items = [Ingredient(item="curd", qty_per_person=50, unit="g"),
             Ingredient(item="curd", qty_per_person=30, unit="g")]
    bill = scale_and_cost(items, 10, load_prices(small_prices))
    assert len(bill.lines) == 1
    assert bill.lines[0].per_guest == Decimal("80.0")
    assert bill.total == Decimal("56.00")                            # 0.8 kg x Rs 70


def test_money_is_exact_where_floats_are_not(small_prices):
    # 0.1 kg of rice for 3 guests at Rs 60: as floats 0.1 * 3 is 0.30000000000000004
    items = [Ingredient(item="rice", qty_per_person=100, unit="g")]
    bill = scale_and_cost(items, 3, load_prices(small_prices))
    assert bill.lines[0].to_buy == Decimal("0.3")
    assert bill.total == Decimal("18.00")


def test_the_sample_list_costs_938_40_for_12_guests():
    bill = scale_and_cost(SAMPLE_ITEMS, 12, load_prices())
    assert len(bill.lines) == 19
    assert bill.total == Decimal("938.40")
    assert bill.not_costed == ["asafoetida (no price in data/prices.csv)"]


def test_zero_guests_is_refused(small_prices):
    with pytest.raises(ValueError, match="at least 1"):
        scale_and_cost(ITEMS, 0, load_prices(small_prices))


def test_cost_step_reads_the_running_dict(small_prices):
    bill = cost_step({"items": ITEMS, "guests": 12, "menu": "ignored"})
    assert bill.total == Decimal("219.00")


# ---------------------------------------------------------------------------
# budget_check
# ---------------------------------------------------------------------------

def test_over_budget():
    check = budget_check(Decimal("938.40"), 800)
    assert check.is_over and check.over_by == Decimal("138.40") and check.left == 0
    assert check.message() == "Over budget by Rs 138.40 (bill Rs 938.40, budget Rs 800)."


def test_within_budget():
    check = budget_check(Decimal("938.40"), 6000)
    assert not check.is_over and check.left == Decimal("5061.60")
    assert check.message() == "Within budget: Rs 5,061.60 left (bill Rs 938.40, budget Rs 6,000)."


def test_exactly_on_budget():
    check = budget_check(Decimal("500.00"), 500)
    assert not check.is_over and check.left == 0
    assert check.message() == "Exactly on budget: Rs 500.00."


def test_rupees_format():
    assert rupees(Decimal("4386543")) == "Rs 4,386,543.00"
    assert rupees(6000, whole=True) == "Rs 6,000"
    assert rupees(Decimal("64.5"), whole=True) == "Rs 64.50"


def test_costing_main_prints_the_table(capsys):
    assert costing.main([]) == 0
    output = capsys.readouterr().out
    assert "Total for 19 items" in output and "Rs 938.40" in output
    assert "Not costed: asafoetida (no price in data/prices.csv)" in output
    assert "RunnableLambda(cost_step).invoke(...) -> total Rs 938.40" in output
    assert "Within budget: Rs 5,061.60 left" in output


def test_ingredients_object_or_list_both_work(small_prices):
    prices = load_prices(small_prices)
    assert scale_and_cost(ITEMS, 12, prices).total == scale_and_cost(list(ITEMS.items), 12, prices).total
    assert isinstance(ITEMS, Ingredients)
