# ---------------------------------------------------------------------------
# Copied from Episode 12 (ep12-dabbaloop-help-desk/sql_guard.py) so this folder runs on its own.
# Do not edit this copy: tests/test_sql_tools.py checks that it still matches the original.
# Episode 15 uses safe_query(), open_read_only() and format_rows() from it (see tools/sql.py).
# Everything below the next line is the original file, unchanged.
# ---------------------------------------------------------------------------
"""
sql_guard.py - Step 9: the SQL guard. Model-written SQL must pass it before it touches the database.

The model writes SQL. Python decides what may run. Four gates, in this order:

  1. Text rules (check_sql):  one statement, SELECT (or WITH ... SELECT) only, no comments,
                              no words that change data, and a LIMIT added (or lowered) to MAX_ROWS.
  2. A READ-ONLY connection:  SQLite opens the file with mode=ro, so even a query that slipped past
                              gate 1 cannot write ("attempt to write a readonly database").
  3. An allow-list:           SQLite itself reports every table, column and function a query touches
                              (its "authorizer"), and we refuse anything not on ALLOWED_COLUMNS.
                              So customers.name and customers.email can never reach the model.
  4. Size and time limits:    a query that runs longer than TIME_LIMIT seconds is stopped, and no
                              single value may grow past MAX_VALUE_BYTES.

Reuse it in your own projects (Episode 15's SQL agent does):

    from sql_guard import safe_query
    safe_query("SELECT COUNT(*) AS orders FROM orders")      # -> text: the SQL used + the rows

    safe_query(sql, db_path=DB_PATH, allowed=ALLOWED_COLUMNS, max_rows=MAX_ROWS, time_limit=TIME_LIMIT) -> str
        sql         the SQL a model (or a person) wrote
        db_path     the SQLite file to read (opened read-only)
        allowed     {"table": {"column", ...}} - everything else is blocked
        max_rows    LIMIT added to the query, and the most rows ever returned
        time_limit  seconds before a slow query is stopped
        returns     the rows as a small text table, or a message that starts with
                    "Blocked:" (a gate said no) or "Error:" (SQLite could not run it).
                    It never raises, so an agent reads the message and can try again.

    check_sql(sql, max_rows=MAX_ROWS) -> GuardResult(ok, sql, reason)        gate 1 only, no database
    run_guarded(sql, db_path=DB_PATH, ...) -> QueryResult(ok, sql, columns, rows, message, truncated)

Run:  python sql_guard.py "DROP TABLE orders"       one statement through all four gates
      python sql_guard.py --demo                    ten statements, safe and unsafe
      python sql_guard.py --readonly-proof          gate 2 on its own: a DELETE with gate 1 switched off

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "dabbaloop.db"

MAX_ROWS = 50          # the most rows a query may return
MAX_LENGTH = 2000      # longer SQL is refused
TIME_LIMIT = 2.0       # seconds
MAX_VALUE_BYTES = 1_000_000   # no single text value may grow bigger than 1 MB (stops memory tricks)

# Gate 3: the ONLY tables and columns a query may read. Names and emails are not listed.
ALLOWED_COLUMNS = {
    "customers": {"customer_id", "zone", "area", "joined_on"},
    "menu_items": {"item_id", "name", "category", "price_inr"},
    "orders": {"order_id", "customer_id", "item_id", "quantity", "order_date", "meal", "note"},
}

# SQL functions a query may call (LIKE and GLOB count as functions in SQLite).
ALLOWED_FUNCTIONS = {
    "count", "sum", "total", "avg", "min", "max", "round", "abs", "coalesce", "ifnull", "nullif", "iif",
    "lower", "upper", "length", "trim", "substr", "substring", "instr", "replace", "printf", "format",
    "like", "glob", "date", "time", "datetime", "julianday", "strftime", "unixepoch", "group_concat",
    "row_number", "rank", "dense_rank", "lag", "lead",
}

# Words that change data or the database. Gate 1 refuses them anywhere in the query.
FORBIDDEN_WORDS = ["insert", "update", "delete", "drop", "alter", "create", "attach", "detach",
                   "pragma", "vacuum", "reindex", "truncate", "grant", "revoke"]


@dataclass
class GuardResult:
    """What gate 1 decided. sql is the query to run (with its LIMIT) or, when blocked, the original."""
    ok: bool
    sql: str
    reason: str = ""


@dataclass
class QueryResult:
    """What happened to one query. message is "" when it ran, else "Blocked: ..." or "Error: ..."."""
    ok: bool
    sql: str
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    message: str = ""
    truncated: bool = False     # True when the result hit max_rows (there may be more rows)


# ---------------------------------------------------------------------------
# Gate 1 - text rules (no database needed)
# ---------------------------------------------------------------------------

def mask_quoted(sql):
    """Replace the inside of '...' strings and "..." names with x, so a word inside a string
    (like note LIKE '%delete%') is not mistaken for a command."""
    sql = re.sub(r"'(?:[^']|'')*'", lambda m: "'" + "x" * (len(m.group()) - 2) + "'", sql)
    return re.sub(r'"(?:[^"]|"")*"', lambda m: '"' + "x" * (len(m.group()) - 2) + '"', sql)


def check_sql(sql, max_rows=MAX_ROWS):
    """Gate 1: the text rules. Returns GuardResult(ok, sql_to_run, reason)."""
    original = (sql or "").strip()
    query = original.rstrip("; \n\t")              # one trailing ; is fine
    if not query:
        return GuardResult(False, original, "Blocked: the query is empty.")
    if len(query) > MAX_LENGTH:
        return GuardResult(False, original, f"Blocked: the query is longer than {MAX_LENGTH} characters.")

    masked = mask_quoted(query)
    if "--" in masked or "/*" in masked:
        return GuardResult(False, original, "Blocked: comments (-- or /* */) are not allowed in a query.")
    if ";" in masked:
        return GuardResult(False, original, "Blocked: only one statement at a time (found a second one after ';').")

    first_word = masked.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return GuardResult(False, original, f"Blocked: only SELECT queries can run here (this one starts with {first_word}).")

    for word in FORBIDDEN_WORDS:
        if re.search(rf"\b{word}\b", masked, flags=re.IGNORECASE):
            return GuardResult(False, original, f"Blocked: the word {word.upper()} is not allowed in a read-only query.")

    # Add a LIMIT, or lower one that is too big. (The fetch below also never reads more than max_rows.)
    limit = re.search(r"\blimit\s+(\d+)(\s+offset\s+\d+)?\s*$", masked, flags=re.IGNORECASE)
    if limit is None:
        query = f"{query} LIMIT {max_rows}"
    elif int(limit.group(1)) > max_rows:
        query = query[:limit.start(1)] + str(max_rows) + query[limit.end(1):]
    return GuardResult(True, query)


# ---------------------------------------------------------------------------
# Gate 2 - the read-only connection
# ---------------------------------------------------------------------------

def open_read_only(db_path=DB_PATH):
    """Open SQLite in read-only mode. The URI form (file:...?mode=ro) is what makes it read-only."""
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Create it first with:  python seed_db.py")
    # as_uri() turns spaces and symbols like | and & into %20, %7C, %26, which SQLite decodes.
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


# ---------------------------------------------------------------------------
# Gate 3 - the allow-list, checked by SQLite's own authorizer
# ---------------------------------------------------------------------------

def make_authorizer(allowed, problems):
    """SQLite calls this for every table, column and function a query needs, BEFORE it runs.
    We answer OK or DENY, and write down why we said no."""
    def authorize(action, arg1, arg2, db_name, trigger):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:            # arg1 = table, arg2 = column ("" for COUNT(*))
            if arg1 not in allowed:
                problems.append(f"table {arg1} is not on the allow-list")
                return sqlite3.SQLITE_DENY
            if arg2 and arg2 not in allowed[arg1]:
                problems.append(f"column {arg1}.{arg2} is not on the allow-list")
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:        # arg2 = function name
            if arg2.lower() in ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_OK
            problems.append(f"the function {arg2}() is not on the allow-list")
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_RECURSIVE:
            problems.append("recursive queries are not allowed")
            return sqlite3.SQLITE_DENY
        problems.append("only reading is allowed")   # anything else: writes, PRAGMA, ATTACH, ...
        return sqlite3.SQLITE_DENY
    return authorize


# ---------------------------------------------------------------------------
# All four gates, then run
# ---------------------------------------------------------------------------

def run_guarded(sql, db_path=DB_PATH, allowed=None, max_rows=MAX_ROWS, time_limit=TIME_LIMIT):
    """Check the SQL, run it read-only with the allow-list and a time limit. Never raises."""
    allowed = ALLOWED_COLUMNS if allowed is None else allowed
    checked = check_sql(sql, max_rows)
    if not checked.ok:
        return QueryResult(False, checked.sql, message=checked.reason)

    try:
        conn = open_read_only(db_path)
    except (FileNotFoundError, sqlite3.Error) as error:
        return QueryResult(False, checked.sql, message=f"Error: {error}")

    problems = []
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)        # part of gate 4: no giant values
    conn.set_authorizer(make_authorizer(allowed, problems))            # gate 3
    deadline = time.monotonic() + time_limit
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)   # gate 4: True = stop
    try:
        cursor = conn.execute(checked.sql)
        rows = cursor.fetchmany(max_rows)                              # never read more than max_rows
        columns = [d[0] for d in cursor.description]
    except sqlite3.Error as error:
        if problems:
            return QueryResult(False, checked.sql, message=f"Blocked: {problems[0]}.")
        if "interrupted" in str(error):
            return QueryResult(False, checked.sql, message=f"Blocked: the query ran longer than {time_limit:g} seconds.")
        return QueryResult(False, checked.sql, message=f"Error: {error}")
    finally:
        conn.close()
    return QueryResult(True, checked.sql, columns, rows, truncated=len(rows) >= max_rows)


def format_rows(columns, rows, truncated=False):
    """A small text table: a header line, a dashed line, then one line per row."""
    if not rows:
        return "(no rows)"
    texts = [[("NULL" if value is None else str(value)) for value in row] for row in rows]
    widths = [max(len(str(c)), *(len(r[i]) for r in texts)) for i, c in enumerate(columns)]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(columns, widths)).rstrip(),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in texts]
    count = f"({len(rows)} row{'s' if len(rows) != 1 else ''}"
    lines.append(count + (": the most the guard allows, there may be more)" if truncated else ")"))
    return "\n".join(lines)


def safe_query(sql, db_path=DB_PATH, allowed=None, max_rows=MAX_ROWS, time_limit=TIME_LIMIT):
    """The reusable tool: run model-written SQL through all four gates and return TEXT. Never raises."""
    result = run_guarded(sql, db_path, allowed, max_rows, time_limit)
    if not result.ok:
        return result.message
    return f"SQL used: {result.sql}\n{format_rows(result.columns, result.rows, result.truncated)}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_STATEMENTS = [
    "SELECT COUNT(*) AS orders FROM orders",
    "DROP TABLE orders",
    "DELETE FROM orders WHERE order_date < '2026-10-01'",
    "UPDATE menu_items SET price_inr = 1 WHERE name = 'Jain Thali'",
    "SELECT * FROM orders; DROP TABLE orders",
    "SELECT email FROM customers",
    "SELECT * FROM customers",
    "SELECT name FROM sqlite_master",
    "SELECT order_id, note FROM orders WHERE note LIKE '%drop%' LIMIT 500",
    "SELECT zone, COUNT(*) AS customers FROM customers GROUP BY zone -- ; DROP TABLE orders",
]


def show(sql, db_path=DB_PATH):
    """Print one statement and what the gates decided."""
    result = run_guarded(sql, db_path)
    print(f"SQL:     {sql}")
    if result.ok:
        print(f"Runs as: {result.sql}")
        print(f"Result:  {len(result.rows)} row(s), first row {result.rows[0] if result.rows else None}")
    else:
        print(f"Verdict: {result.message}")


def readonly_proof(db_path=DB_PATH):
    """Gate 2 on its own: skip gates 1, 3 and 4 and send a DELETE straight to the read-only connection."""
    conn = open_read_only(db_path)
    before = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    print(f"orders before: {before} rows")
    print("Sending  DELETE FROM orders  with the text rules switched OFF ...")
    try:
        conn.execute("DELETE FROM orders")
        conn.commit()
        print("The DELETE ran (this should never happen)")
    except sqlite3.OperationalError as error:
        print(f"SQLite refused: {error}")
    after = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    conn.close()
    print(f"orders after:  {after} rows")


def main():
    parser = argparse.ArgumentParser(description="Check SQL with the Dabbaloop guard.")
    parser.add_argument("sql", nargs="?", default="DROP TABLE orders")
    parser.add_argument("--demo", action="store_true", help="run ten safe and unsafe statements")
    parser.add_argument("--readonly-proof", action="store_true", help="a DELETE with only the read-only gate")
    args = parser.parse_args()

    if not Path(DB_PATH).exists():
        print(f"{DB_PATH.name} not found. Create it first with:  python seed_db.py")
        raise SystemExit(1)
    if args.readonly_proof:
        readonly_proof()
    elif args.demo:
        for number, sql in enumerate(DEMO_STATEMENTS, start=1):
            print(f"--- {number} ---")
            show(sql)
    else:
        show(args.sql)


if __name__ == "__main__":
    main()
