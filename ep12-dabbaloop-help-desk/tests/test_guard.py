"""
tests/test_guard.py - the SQL guard: safe and unsafe statements, the allow-list, the read-only
connection, the LIMIT and the time limit. No model is involved: this is plain Python + SQLite.
"""

import hashlib
import sqlite3

import pytest
from desk_fixtures import JAIN_SQL, db_path  # noqa: F401

from sql_guard import MAX_ROWS, check_sql, open_read_only, run_guarded, safe_query

SAFE = [
    "SELECT COUNT(*) FROM orders",
    "select meal, count(*) from orders group by meal",
    "WITH recent AS (SELECT * FROM orders WHERE order_date >= '2026-10-16') SELECT COUNT(*) FROM recent",
    "SELECT note FROM orders WHERE note LIKE '%delete%'",          # the word is inside a string
    "SELECT 'a;b' AS text_with_semicolon",                          # so is the ;
    "SELECT COUNT(*) FROM orders;",                                 # one trailing ; is fine
]

UNSAFE = [
    ("DROP TABLE orders", "starts with DROP"),
    ("drop table orders", "starts with DROP"),
    ("DELETE FROM orders", "starts with DELETE"),
    ("UPDATE menu_items SET price_inr = 1", "starts with UPDATE"),
    ("INSERT INTO orders VALUES (1)", "starts with INSERT"),
    ("REPLACE INTO menu_items VALUES (1, 'x', 'thali', 1)", "starts with REPLACE"),
    ("PRAGMA table_info(orders)", "starts with PRAGMA"),
    ("ATTACH DATABASE 'other.db' AS other", "starts with ATTACH"),
    ("SELECT * FROM orders; DROP TABLE orders", "one statement"),
    ("SELECT 1 -- ; DROP TABLE orders", "comments"),
    ("SELECT /* hi */ 1", "comments"),
    ("WITH x AS (SELECT 1) DELETE FROM orders", "DELETE is not allowed"),
    ("", "empty"),
    ("SELECT " + "1, " * 1000 + "1", "longer than"),
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_statements_pass_the_text_rules(sql):
    result = check_sql(sql)
    assert result.ok, result.reason
    assert result.sql.endswith(f"LIMIT {MAX_ROWS}")


@pytest.mark.parametrize("sql, why", UNSAFE)
def test_unsafe_statements_are_blocked_with_a_clear_reason(sql, why):
    result = check_sql(sql)
    assert not result.ok
    assert result.reason.startswith("Blocked:")
    assert why in result.reason


def test_a_missing_limit_is_added_and_a_big_one_is_lowered():
    assert check_sql("SELECT * FROM orders").sql == "SELECT * FROM orders LIMIT 50"
    assert check_sql("SELECT * FROM orders LIMIT 500").sql == "SELECT * FROM orders LIMIT 50"
    assert check_sql("SELECT * FROM orders LIMIT 10").sql == "SELECT * FROM orders LIMIT 10"
    assert check_sql("SELECT * FROM orders LIMIT 900 OFFSET 5").sql == "SELECT * FROM orders LIMIT 50 OFFSET 5"
    assert check_sql("SELECT * FROM orders", max_rows=5).sql == "SELECT * FROM orders LIMIT 5"


def test_the_jain_question_runs_and_gives_the_real_count(db_path):
    result = run_guarded(JAIN_SQL, db_path)
    assert result.ok
    assert result.columns == ["jain_thalis_sold"]
    assert result.rows == [(16,)]


@pytest.mark.parametrize("sql, why", [
    ("SELECT email FROM customers", "column customers.email is not on the allow-list"),
    ("SELECT name FROM customers", "column customers.name is not on the allow-list"),
    ("SELECT * FROM customers", "is not on the allow-list"),          # * includes the private columns
    ("SELECT name FROM sqlite_master", "table sqlite_master is not on the allow-list"),
    ("SELECT load_extension('evil')", "load_extension() is not on the allow-list"),
    ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT x FROM n", "recursive"),
])
def test_the_allow_list_is_checked_by_sqlite_itself(db_path, sql, why):
    result = run_guarded(sql, db_path)
    assert not result.ok
    assert result.message.startswith("Blocked:")
    assert why in result.message


def test_a_join_on_allowed_columns_works(db_path):
    sql = ("SELECT c.zone, COUNT(*) AS order_lines FROM orders o "
           "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.zone ORDER BY c.zone")
    result = run_guarded(sql, db_path)
    assert result.ok
    assert [zone for zone, _ in result.rows] == ["A", "B", "C", "D"]
    assert sum(count for _, count in result.rows) == 651


def test_your_own_allow_list(db_path):
    only_names = {"menu_items": {"name"}}
    assert run_guarded("SELECT name FROM menu_items", db_path, allowed=only_names).ok
    blocked = run_guarded("SELECT price_inr FROM menu_items", db_path, allowed=only_names)
    assert "menu_items.price_inr is not on the allow-list" in blocked.message


def test_the_read_only_connection_refuses_writes_even_without_the_text_rules(db_path):
    conn = open_read_only(db_path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM orders")
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 651
    conn.close()


def test_a_slow_query_is_stopped(db_path):
    result = run_guarded("SELECT COUNT(*) FROM orders a, orders b, orders c", db_path, time_limit=0.2)
    assert result.message == "Blocked: the query ran longer than 0.2 seconds."


def test_a_giant_value_is_refused(db_path):
    result = run_guarded("SELECT printf('%.*c', 5000000, 'x') AS huge", db_path)
    assert result.message == "Error: string or blob too big"


def test_window_functions_are_allowed(db_path):
    sql = "SELECT order_id, ROW_NUMBER() OVER (ORDER BY order_id) AS n FROM orders"
    assert run_guarded(sql, db_path, max_rows=3).rows == [(1, 1), (2, 2), (3, 3)]


def test_never_more_than_max_rows(db_path):
    result = run_guarded("SELECT order_id FROM orders", db_path, max_rows=7)
    assert len(result.rows) == 7
    assert result.truncated


def test_safe_query_returns_text_and_never_raises(db_path, tmp_path):
    assert safe_query(JAIN_SQL, db_path).startswith("SQL used: SELECT SUM(o.quantity)")
    assert "16" in safe_query(JAIN_SQL, db_path)
    assert safe_query("DROP TABLE orders", db_path).startswith("Blocked:")
    assert safe_query("SELECT nope FROM orders", db_path) == "Error: no such column: nope"
    assert safe_query("SELECT 1", tmp_path / "missing.db").startswith("Error: missing.db not found")


def test_the_database_file_is_unchanged_after_every_attack(db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    for sql, _ in UNSAFE:
        run_guarded(sql, db_path)
    run_guarded("SELECT * FROM customers", db_path)
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
