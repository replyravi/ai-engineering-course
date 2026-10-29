"""
tools/ - the agent's tool belt. One file per kind of tool:

    calc.py   the calculator (numexpr, Episode 11's safe check)
    sql.py    list_tables, describe_table, run_select (read-only, Episode 12's guard); get_prices
    docs.py   search_docs over the crawled rule book (Episode 07's index pattern)

tool_belt() hands the agent the kinds you ask for.

Author: Ravi Shanker Singh (TechWithRavi)
"""

KINDS = ("calc", "sql", "docs")


def tool_belt(kinds=KINDS, careful=None, price_tool=False, db_path=None, store=None) -> list:
    """The tools for the kinds asked for, in a fixed order: calc, sql, docs.

    careful     None = use USE_CAREFUL_DESCRIPTION from tools/sql.py; True/False to override (Step 9)
    price_tool  True = give get_prices instead of run_select (the parameterised fix)
    db_path     another database file (tests); store: another Chroma index (tests)
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"Unknown tool kind(s): {', '.join(sorted(unknown))}. Use: {', '.join(KINDS)}.")
    belt = []
    if "calc" in kinds:
        from tools.calc import calculator
        belt.append(calculator)
    if "sql" in kinds:
        from tools import sql
        options = {"db_path": db_path} if db_path else {}
        if careful is None:
            careful = sql.USE_CAREFUL_DESCRIPTION
        list_tables, describe_table, run_select = sql.make_sql_tools(careful=careful, **options)
        belt += [list_tables, describe_table, sql.make_price_tool(**options) if price_tool else run_select]
    if "docs" in kinds:
        from tools.docs import make_search_tool
        belt.append(make_search_tool(store))
    return belt
