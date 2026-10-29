"""
tests/test_sql_tools.py - seeding (Step 5), the read-only SQL tools and the guard (Step 6),
and the Episode 12 copy the guard comes from. Every test uses a temporary database.
"""

import sqlite3

import pytest
from mandi_fixtures import EPISODE, temp_db, without_header  # noqa: F401  (pytest fixture)
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

import seed_db
from tools import tool_belt
from tools.sql import RUN_SELECT_BASIC, RUN_SELECT_CAREFUL, make_price_tool, make_sql_tools, read_only_engine


def test_the_copy_matches_episode_12():
    original = EPISODE.parent / "ep12-dabbaloop-help-desk" / "sql_guard.py"
    if not original.exists():
        pytest.skip("Episode 12's folder is not here (this folder was copied on its own)")
    copy = EPISODE / "tools" / "ep12_sql_guard.py"
    assert without_header(copy) == original.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 5 - seeding
# ---------------------------------------------------------------------------

def test_seed_makes_150_price_rows_and_3_cities(tmp_path):
    counts = seed_db.seed(tmp_path / "mandi.db")
    assert counts == {"cities": 3, "daily_prices": 150}


def test_every_vegetable_city_and_day_appears_once():
    rows = seed_db.make_rows()
    keys = {(r["vegetable"], r["city"], r["day"]) for r in rows}
    assert len(rows) == len(keys) == 5 * 3 * 10
    assert all(r["price_rs_per_kg"] > 0 and r["arrivals_quintals"] > 0 for r in rows)


def test_the_seed_makes_the_same_rows_every_time():
    assert seed_db.make_rows() == seed_db.make_rows()
    onion = {(r["city"], r["day"]): r["price_rs_per_kg"] for r in seed_db.make_rows() if r["vegetable"] == "onion"}
    assert (onion[("Indore", 3)], onion[("Pune", 3)], onion[("Nashik", 3)]) == (22.9, 23.5, 24.6)


def test_seeding_twice_starts_fresh(tmp_path):
    seed_db.seed(tmp_path / "mandi.db")
    assert seed_db.seed(tmp_path / "mandi.db")["daily_prices"] == 150     # not 300


# ---------------------------------------------------------------------------
# Step 6 - the tools
# ---------------------------------------------------------------------------

def test_list_tables(temp_db):
    list_tables, _, _ = make_sql_tools(temp_db)
    assert list_tables.invoke({}) == "cities, daily_prices"


def test_describe_table_shows_columns_values_and_examples(temp_db):
    _, describe_table, _ = make_sql_tools(temp_db)
    result = describe_table.invoke({"table_name": "daily_prices"})
    assert result.startswith("Table daily_prices: 150 rows (sample data, not real prices)")
    assert "values: brinjal, cabbage, onion, potato, tomato" in result
    assert "values: Indore, Nashik, Pune" in result
    assert "price_rs_per_kg    FLOAT        from 11.5 to 38.1" in result
    assert result.endswith("(3 rows)")


def test_describe_table_with_a_wrong_name_explains_instead_of_crashing(temp_db):
    _, describe_table, _ = make_sql_tools(temp_db)
    assert describe_table.invoke({"table_name": "prices"}) == \
        "Error: there is no table called 'prices'. Tables: cities, daily_prices."


def test_run_select_runs_a_select_and_adds_a_limit(temp_db):
    _, _, run_select = make_sql_tools(temp_db)
    result = run_select.invoke({"query": "SELECT city FROM daily_prices WHERE vegetable = 'onion' AND day = 3 "
                                         "ORDER BY price_rs_per_kg"})
    assert result.splitlines()[0].endswith("LIMIT 20")
    assert result.splitlines()[3] == "Indore"


@pytest.mark.parametrize("query, reason", [
    ("DROP TABLE daily_prices", "Blocked: only SELECT queries can run here (this one starts with DROP)."),
    ("SELECT * FROM daily_prices; DELETE FROM daily_prices", "Blocked: only one statement at a time"),
    ("SELECT name FROM sqlite_master", "Blocked: table sqlite_master is not on the allow-list."),
    ("SELECT price FROM daily_prices", "Error: no such column: price"),     # a typical model mistake
])
def test_run_select_refuses_or_explains(temp_db, query, reason):
    _, _, run_select = make_sql_tools(temp_db)
    assert run_select.invoke({"query": query}).startswith(reason)


def test_the_data_is_unchanged_after_a_drop_attempt(temp_db):
    _, _, run_select = make_sql_tools(temp_db)
    run_select.invoke({"query": "DROP TABLE daily_prices"})
    with sqlite3.connect(temp_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0] == 150


def test_the_sqlalchemy_engine_of_the_tools_cannot_write(temp_db):
    with read_only_engine(temp_db).connect() as conn:
        with pytest.raises(OperationalError, match="readonly"):
            conn.execute(text("DELETE FROM daily_prices"))


def test_basic_and_careful_descriptions():
    assert "brackets" not in RUN_SELECT_BASIC
    assert RUN_SELECT_CAREFUL.startswith(RUN_SELECT_BASIC)
    assert "put brackets around the OR part or use IN" in RUN_SELECT_CAREFUL


def test_the_belt_picks_the_description(temp_db):
    basic = tool_belt(["sql"], careful=False, db_path=temp_db)[2]
    careful = tool_belt(["sql"], careful=True, db_path=temp_db)[2]
    assert (basic.name, careful.name) == ("run_select", "run_select")
    assert (basic.description, careful.description) == (RUN_SELECT_BASIC, RUN_SELECT_CAREFUL)


def test_get_prices_writes_the_sql_itself(temp_db):
    get_prices = make_price_tool(temp_db)
    result = get_prices.invoke({"vegetable": "Onion", "cities": ["nashik", "Pune"], "days": [3]})
    assert "city IN ('Nashik', 'Pune') AND daily_prices.day IN (3)" in result
    assert "(2 rows)" in result and "726" in result and "349" in result


def test_get_prices_with_no_cities_or_days_means_all(temp_db):
    result = make_price_tool(temp_db).invoke({"vegetable": "potato", "cities": ["Indore"]})
    assert "(10 rows)" in result


def test_the_belt_order_and_an_unknown_kind(temp_db):
    names = [t.name for t in tool_belt(["calc", "sql"], db_path=temp_db)]
    assert names == ["calculator", "list_tables", "describe_table", "run_select"]
    names = [t.name for t in tool_belt(["sql", "calc"], price_tool=True, db_path=temp_db)]
    assert names == ["calculator", "list_tables", "describe_table", "get_prices"]
    with pytest.raises(ValueError, match="Unknown tool kind"):
        tool_belt(["weather"])
