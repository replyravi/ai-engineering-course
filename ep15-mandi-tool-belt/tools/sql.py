"""
tools/sql.py - Step 6 (and the Step 9 fix): small, read-only SQL tools for the agent.

A new analyst first asks which tables exist, then reads the columns, then writes the query.
Our agent gets exactly those three tools:

    list_tables()              the table names
    describe_table(name)       the columns, the values they hold and 3 example rows
    run_select(query)          one SELECT, through Episode 12's four-gate guard (tools/ep12_sql_guard.py):
                               SELECT only + a LIMIT, a read-only connection, an allow-list of
                               tables and columns, and size and time limits

The Step 9 fixes for the AND/OR trap:
    make_sql_tools(careful=True)   the same run_select with a better DESCRIPTION (brackets, IN)
    make_price_tool()              get_prices(vegetable, cities, days): the model picks values,
                                   Python writes the SQL with bound parameters

    python -m tools.sql                          the tools run by hand, a DROP attempt included (no key)
    python -m tools.sql "SELECT COUNT(*) FROM daily_prices"     one query through run_select

Author: Ravi Shanker Singh (TechWithRavi)
"""

import sys

from langchain.tools import tool
from sqlalchemy import MetaData, Table, create_engine, func, inspect, select

from seed_db import DB_PATH, daily_prices
from tools.ep12_sql_guard import format_rows, open_read_only, safe_query

# The ONLY tables and columns a query may read (gate 3 of the guard).
ALLOWED = {
    "cities": {"city", "state"},
    "daily_prices": {"id", "day", "city", "vegetable", "price_rs_per_kg", "arrivals_quintals"},
}
MAX_ROWS = 20          # the guard adds LIMIT 20 to every query

# Step 8 uses the BASIC description. Step 9 (the fix) adds two rules the model must follow.
# Happy with the fix? Set this to True, and ask.py uses the careful description from then on.
USE_CAREFUL_DESCRIPTION = False

RUN_SELECT_BASIC = "Run one read-only SQL SELECT query on the mandi price database and return the rows."
RUN_SELECT_CAREFUL = (
    RUN_SELECT_BASIC + " Call list_tables and describe_table first and use the exact column names and "
    "values they show. SQL applies AND before OR, so when a WHERE clause mixes them, put brackets around "
    "the OR part or use IN, for example: WHERE vegetable = 'onion' AND day = 3 AND city IN ('Nashik', 'Pune')."
)


def read_only_engine(db_path=None):
    """A SQLAlchemy engine that can only READ: every connection is Episode 12's open_read_only (mode=ro)."""
    db_path = db_path or DB_PATH
    return create_engine("sqlite://", creator=lambda: open_read_only(db_path))


def describe(engine, table_name: str, db_path) -> str:
    """The text describe_table returns: columns + the values they hold + 3 example rows."""
    if table_name not in ALLOWED:
        return f"Error: there is no table called {table_name!r}. Tables: {', '.join(sorted(ALLOWED))}."
    table = Table(table_name, MetaData(), autoload_with=engine)      # read the columns from the file
    lines = []
    with engine.connect() as conn:
        rows = conn.execute(select(func.count()).select_from(table)).scalar_one()
        lines.append(f"Table {table_name}: {rows} rows (sample data, not real prices)")
        lines.append("Columns:")
        for column in table.columns:
            low, high, count = conn.execute(
                select(func.min(column), func.max(column), func.count(column.distinct()))).one()
            if isinstance(low, str) and count <= 10:
                values = conn.execute(select(column).distinct().order_by(column)).scalars().all()
                hint = "values: " + ", ".join(values)
            else:
                hint = f"from {low} to {high}"
            lines.append(f"  {column.name:<18} {str(column.type):<12} {hint}")
    example = safe_query(f"SELECT * FROM {table_name} LIMIT 3", db_path=db_path, allowed=ALLOWED)
    lines.append("3 example rows:")
    lines.append(example.split("\n", 1)[1])       # drop the "SQL used:" line
    return "\n".join(lines)


def make_sql_tools(db_path=None, careful=False) -> list:
    """The three SQL tools over ONE database file (default mandi.db; tests pass a temporary one)."""
    db_path = db_path or DB_PATH
    engine = read_only_engine(db_path)

    @tool
    def list_tables() -> str:
        """List the tables in the mandi price database."""
        try:
            return ", ".join(name for name in inspect(engine).get_table_names() if name in ALLOWED)
        except Exception as error:          # an exception inside a tool would stop the whole agent
            return f"Error: {error}"

    @tool
    def describe_table(table_name: str) -> str:
        """Show the columns of one table, the values each column holds, and 3 example rows."""
        try:
            return describe(engine, table_name.strip(), db_path)
        except Exception as error:
            return f"Error: {error}"

    def run_select(query: str) -> str:
        return safe_query(query, db_path=db_path, allowed=ALLOWED, max_rows=MAX_ROWS)   # never raises

    description = RUN_SELECT_CAREFUL if careful else RUN_SELECT_BASIC
    return [list_tables, describe_table, tool("run_select", description=description)(run_select)]


def make_price_tool(db_path=None):
    """Fix 2: a parameterised tool. The model never writes SQL, so it cannot get AND/OR wrong."""
    engine = read_only_engine(db_path)

    @tool
    def get_prices(vegetable: str, cities: list[str] | None = None, days: list[int] | None = None) -> str:
        """Look up the daily price (rupees per kg) and arrivals (quintals) of ONE vegetable.

        vegetable: onion, tomato, potato, cabbage or brinjal.
        cities: any of Nashik, Pune, Indore (leave empty for all three).
        days: sample days 1 to 10 (leave empty for all ten).
        Returns one row per city and day. Use the calculator for totals, averages and percentages.
        """
        c = daily_prices.c
        query = (select(c.day, c.city, c.vegetable, c.price_rs_per_kg, c.arrivals_quintals)
                 .where(c.vegetable == vegetable.strip().lower()))
        if cities:
            query = query.where(c.city.in_([city.strip().title() for city in cities]))
        if days:
            query = query.where(c.day.in_([int(day) for day in days]))
        query = query.order_by(c.day, c.city)
        try:
            with engine.connect() as conn:
                rows = conn.execute(query).fetchall()
        except Exception as error:
            return f"Error: {error}"
        sql = " ".join(str(query.compile(compile_kwargs={"literal_binds": True})).split())
        columns = ["day", "city", "vegetable", "price_rs_per_kg", "arrivals_quintals"]
        return f"SQL used (written by Python): {sql}\n{format_rows(columns, rows)}"

    return get_prices


# ---------------------------------------------------------------------------
# Run the tools by hand (no model, no key)
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "SELECT city, price_rs_per_kg FROM daily_prices WHERE vegetable = 'onion' AND day = 3 ORDER BY price_rs_per_kg",
    "DROP TABLE daily_prices",
    "SELECT * FROM daily_prices; DELETE FROM daily_prices",
    "SELECT name FROM sqlite_master",
]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not DB_PATH.exists():
        print(f"{DB_PATH.name} not found. Create it first with:  python seed_db.py")
        return 1
    list_tables, describe_table, run_select = make_sql_tools()
    if argv:
        print(run_select.invoke({"query": " ".join(argv)}))
        return 0

    print(f"list_tables() -> {list_tables.invoke({})}\n")
    print('describe_table("daily_prices") ->')
    print(describe_table.invoke({"table_name": "daily_prices"}))
    for query in DEMO_QUERIES:
        print(f'\nrun_select("{query}") ->')
        print(run_select.invoke({"query": query}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
