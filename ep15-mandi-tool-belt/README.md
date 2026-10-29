# Episode 15: One Agent, Many Tools: Calculator, Read-Only SQL and Crawl Search With LangChain create_agent

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep15-mandi-tool-belt

## What you will build

**Mandi Tool Belt**: ONE LangChain agent with a belt of tools. It answers questions about vegetable prices in three markets (mandis) by choosing, on its own, between a **calculator**, **read-only SQL tools** over a small database of invented prices (5 vegetables, 3 cities, 10 days), and a **document-search tool** built from a capped Apify crawl of its rule book. Every run prints the trace, the answer, the sources and the tokens. Then `check_answers.py` grades the agent against answers we worked out with our own SQL, because an agent's number is only as good as your check of it.

```text
$ python ask.py full
Question: A hotel in Pune buys 3 bags of onions at the day 7 Pune price. Using the rule book, what is the total bill with commission, market fee and loading?
Tools: calculator, list_tables, describe_table, run_select, search_docs
Model: gemini-3.8-flash   Caps: 8 model calls, 30,000 tokens

Trace (<n> messages):
   1. human  A hotel in Pune buys 3 bags of onions at the day 7 Pune price. ...
   2. ai     calls search_docs({"query": "<what the model searches for>"})
   3. tool   search_docs -> [1] fees.html#2  source: https://.../site/fees.html ...
   ...       (more calls: run_select for the day 7 price, calculator for the bill)
  <n>. ai     <the answer, with the page it came from>
Tool calls: <how many> (<which tools, in order>)

Answer: <the answer>
Sources (what the tools returned):
  page: https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/fees.html
  SQL used: <the SELECT the model wrote> LIMIT 20
Tokens: <in> in + <out> out = <total> (thinking <t> of the out), <n> model calls
```

(This is the **shape** of a run, not a recorded one. Gemini chooses its own tool calls and words, so your trace will differ. The tool results are real: run the same tool with the same input and you get the same text. The right total for this question is **Rs 4,083.75**, and Step 15 shows how to check it by hand.)

**You will learn:**

- What a tool is to an agent: a **name**, a **description**, an **argument schema** and a **return value**. The model only ever reads the first three, so the description decides when a tool gets used.
- **Native tool calling**: the model returns a structured call like `{"name": "run_select", "args": {...}}`, and `create_agent` runs the loop. An agent may also answer with no tool at all.
- A SQL agent that first **discovers** the tables, then **describes** them, then **queries** them, over a read-only connection with Episode 12's guard.
- How to **grade an agent** against hand-written SQL, why `AND` and `OR` without brackets leak rows, and two ways to fix it.
- How to **crawl** a permitted site with an Apify Actor under a 3-page cap, index it like Episode 07 did, and give the agent a **search tool** with source URLs.
- How to **cap** every run (model calls, tokens, graph steps) and read real **token usage** from every message.

> **Seen an older tutorial?** It built three separate things: an agent with `initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)` and `load_tools(["llm-math"])`, a SQL agent with `create_sql_agent` + `SQLDatabaseToolkit` (an `AgentExecutor` underneath), and a crawl with `ApifyWrapper` + `VectorstoreIndexCreator`. In 2026 we build **one** agent with `langchain.agents.create_agent` and small `@tool` functions. The old names still exist in the legacy `langchain-classic` and the archived `langchain-community` packages, and in our course environment some of them no longer even start (Steps 3 and 11 show the real errors).

## Safety first

- **The prices are invented.** `mandi.db` is sample data made with a fixed random seed, **not real mandi rates**. Never use it to buy or sell anything.
- **SQL is read-only.** The agent can only run one `SELECT` at a time, on a read-only connection, through the four gates of Episode 12's guard. A `DROP` comes back as `Blocked:`. Optional: a human approves every query (Step 7).
- **Crawling only where it is allowed.** We crawl only the three rule-book pages written for this course (`site/`, published by Ravi), check `robots.txt` first, tell the Actor to respect it too, and cap the crawl at **3 pages**. No real business is crawled.
- **Secrets stay in `.env`.** The Apify token lives in the root `.env`, never in code, never on screen. No Apify account? The offline copy `fixtures/crawl.json` does the same job.

## Prerequisites

- Finished Episode 06 (`ep06-helpdesk-ladder`: tools and your first agent), Episode 07 (`ep07-ask-the-handbook`: load, split, embed, store), Episode 11 (`ep11-party-planner-chains`: the calculator tool) and Episode 12 (`ep12-dabbaloop-help-desk`: the SQL guard). This folder runs on its own: it carries copies of the Episode 11 calculator and the Episode 12 guard.
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- The shared `.venv` and root `.env` with your **free** Gemini key from Episode 1 ([SETUP.md](../SETUP.md)). For this episode we suggest the smarter model: add `LLM_MODEL=gemini-3.8-flash` to the root `.env` (also on the free tier when this was written; it thinks more, so it is slower and uses more of your free quota). The default `gemini-3.5-flash-lite` works too, but makes more mistakes in long tool chains.
- On the free tier, Google may use your prompts to improve its products. Our data is invented, so it is safe to send.
- Optional for Step 11: a free Apify account (Apify's terms say you must be 18 or older) and its API token.
- About 60 minutes. The only model is the local embeddings model [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (about 90 MB, **Apache-2.0** licence: free to use, also commercially; keep its licence notice if you share the model files). If you did Episode 01, 04, 06 or 07 it is already on your computer. (The optional final challenge also uses a local Ollama model; CHALLENGE.md gives its size and licence.)

## What is in this folder

```text
ep15-mandi-tool-belt/
├── tools/                    the tool belt (run each with python -m tools.<name>)
│   ├── calc.py               Step 2: the calculator tool, and why brackets matter
│   ├── sql.py                Steps 6 and 9: list_tables, describe_table, run_select, get_prices
│   ├── docs.py               Step 13: search_docs over the crawled rule book
│   ├── ep11_calculator.py    copied from Episode 11 so this folder runs on its own (do not edit)
│   ├── ep12_sql_guard.py     copied from Episode 12 so this folder runs on its own (do not edit)
│   └── __init__.py           tool_belt(): hands the agent the kinds of tools you ask for
├── agent.py                  Steps 3, 4, 7: ONE agent, any tools, caps and a readable trace
├── ask.py                    ask one question: trace, answer, sources, tokens
├── seed_db.py                Step 5: mandi.db with SQLAlchemy Core (invented prices)
├── check_answers.py          Steps 8 and 9: the scoreboard against our own SQL
├── data/questions.json       5 questions with their truth SQL (question 4 is the AND/OR trap)
├── crawl.py                  Step 11: robots.txt check + a capped Apify crawl -> crawl.json
├── site/                     the 3 rule-book pages we crawl (written for this course)
├── fixtures/crawl.json       offline copy of those pages, in the shape crawl.py saves
├── fixtures/make_fixture.py  how the offline copy was made (you do not need to run it)
├── ingest.py                 Step 12: split, embed, store (Episode 07's pipeline)
├── token_meter.py            Step 14: tokens per model call, and the token cap
├── llm.py, testing/          shared helpers (synced from _template)
├── tests/                    offline tests (scripted fake model, mocked Apify, no key, no internet)
├── requirements.txt          points to the shared ../requirements-base.txt
├── CHALLENGE.md
└── mandi.db, index/, runs/   made by the scripts (git-ignored)
```

Every file is already here. Type the code yourself as we go, or read along and run it.

## Step 0 - Get ready (about 2 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the repo root (`ai-engineering-course/`). You created them once in Episode 1 ([SETUP.md](../SETUP.md)). New here? Do SETUP.md first, then come back.

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd ep15-mandi-tool-belt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep15-mandi-tool-belt
```

Expected: your prompt starts with `(.venv)` and ends in `ep15-mandi-tool-belt`.

> [!WARNING]
> **Common errors**
> - Windows: `The term '.\.venv\Scripts\Activate.ps1' is not recognized` / macOS, Linux: `.venv/bin/activate` with `no such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

### 0.2 Make sure the libraries are up to date

Nothing new to install: SQLAlchemy, numexpr, apify-client, Chroma and the embeddings library are all in the shared `requirements-base.txt`. If you pulled a newer version of the repo, update from this folder:

```bash
pip install -r ../requirements-base.txt
pip check
```

Expected (last line):

```text
No broken requirements found.
```

> [!WARNING]
> **Common errors**
> - `No matching distribution found for torch==2.14.1` - the `.venv` was made with Python 3.14/3.15. Delete the `.venv` folder and recreate it with Python 3.13 ([SETUP.md](../SETUP.md)).
> - `ModuleNotFoundError: No module named 'sqlalchemy'` (or `numexpr`, `apify_client`) - the `.venv` is not active in this terminal. Activate it again (0.1).

---

## Step 1 - Check the setup

Make sure your key is found and Gemini answers. `llm.py` looks for the `.env` in this folder and then in each parent folder, so it finds the root file by itself. Do not copy `.env` into episode folders.

```bash
python llm.py
```

Expected output (your path will differ, and Gemini writes its own greeting):

```text
Settings file: /.../ai-engineering-course/.env
Provider: gemini   Model: gemini-3.8-flash
API key found. Asking the model a tiny question...
<a one-sentence hello written by Gemini>
```

(`Model:` shows `gemini-3.5-flash-lite` if you did not add `LLM_MODEL=gemini-3.8-flash`.)

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama`. Nothing else changes.

> [!WARNING]
> **Common errors**
> - `No API key found for LLM_PROVIDER=gemini` - every script in this folder prints this friendly help instead of crashing. The `.env` file is missing, sits in the wrong folder (it belongs in `ai-engineering-course/`), is named `.env.txt`, or still says `paste-your-key-here`. The last line tells you which `.env` file was found.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute (per-minute limit) or until the daily reset. Your limits: https://aistudio.google.com/rate-limit
> - `404 ... model not found` - the name in `LLM_MODEL` is wrong or retired. Leave `LLM_MODEL=` empty to use the default.
> - A long traceback that ends in `httpx.ConnectError: ...` - no internet connection, so Gemini cannot be reached. Connect and run again. (`ask.py` and `check_answers.py` print a one-line `Could not reach the model's server` instead.)

## Step 2 - The calculator tool, and why brackets matter

A toolbox where every tool has a label and instructions taped on: the worker reads the labels, not your mind. To an agent, a tool is exactly that: a **name**, a **description** and an **argument schema** the model reads, plus the **value** our Python returns. We reuse Episode 11's calculator (copied into `tools/ep11_calculator.py`): the model writes the expression, numexpr computes the number, and a check before numexpr lets in only numbers, `+ - * / **` and brackets. We never call `eval()` on text a model wrote.

`tools/calc.py` (the important lines):

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from tools.ep11_calculator import calculate, calculator     # Episode 11's tool, reused

# Same numbers, two answers: BODMAS says brackets first, then / and *, then + and -.
PRECEDENCE = [
    ("A 20 kg crate of tomatoes at Rs 28 per kg plus Rs 50 transport, shared by 2 shops",
     "(20 * 28 + 50) / 2", "20 * 28 + 50 / 2"),
    ...
]

def what_the_model_sees() -> dict:
    """The tool as the model receives it: name, description and a JSON schema of the arguments."""
    return convert_to_openai_tool(calculator)["function"]
```

`tools/` is a folder of Python files (a package), so we run its files with `-m`, from this folder:

```bash
python -m tools.calc
```

Expected output (exact, with Python 3.13):

```text
What the model sees (the tool's name, description and argument schema):
{
  "name": "calculator",
  "description": "Calculate an arithmetic expression exactly, like '12 * 154' or '(938.40 / 12) * 17'.\n\nUse only numbers, + - * / ** and brackets. Use brackets to make the order clear.\nWrite a percentage as / 100: 18% of 500 is '500 * 18 / 100'.\nUse this for EVERY sum, product, split or percentage. Never do maths in your head.",
  "parameters": {
    "properties": {
      "expression": {
        "type": "string"
      }
    },
    "required": [
      "expression"
    ],
    "type": "object"
  }
}

Brackets change the answer (the tool, run by hand):
  A 20 kg crate of tomatoes at Rs 28 per kg plus Rs 50 transport, shared by 2 shops
    (20 * 28 + 50) / 2           -> 305
    20 * 28 + 50 / 2             -> 585
  An 8.5 km auto ride: Rs 30 for the first 1.5 km, Rs 17 per km after that, Rs 20 waiting
    30 + (8.5 - 1.5) * 17 + 20   -> 169
    30 + 8.5 - 1.5 * 17 + 20     -> 33

Refused (returned as text, so an agent reads the reason and tries again):
  __import__('os').system('ls')    -> Error: only numbers, + - * / ** and brackets are allowed (found: ' _ e i l m o p r s t y).
  round(2.5)                       -> Error: only numbers, + - * / ** and brackets are allowed (found: d n o r u).
  2 ^ 10                           -> Error: write powers with **: 2 to the power 10 is 2**10.
  9**9**9                          -> Error: the result is too big.
```

The docstring became the `description`, and the type hint `expression: str` became the schema. (On Python 3.11 or 3.12 the description lines after the first start with 4 spaces: Python 3.13 is the first version that strips a docstring's indentation.) Its line "Use brackets to make the order clear" is there for a reason: the same numbers give 305 or 585 depending on two brackets. (numexpr on its own would happily compute `round(2.5)`; our check refuses every letter before numexpr sees it, so no name or function ever gets in.) Try your own: `python -m tools.calc "(20 * 28 + 50) / 2"`.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'tools'` (or `'llm'`) - run it with `-m` from the `ep15-mandi-tool-belt` folder: `python -m tools.calc`, not `python tools/calc.py`.
> - `No module named tools.calc` - you are in the repo root, which has its own `tools/` folder (course helpers). `cd ep15-mandi-tool-belt` first.
> - Why not let the model do the maths? A model reads and writes numbers as tokens (pieces of text), so long sums can come out wrong. Episode 11 put that to the test. The tool is exact every time.

## Step 3 - An agent with only the calculator

`create_agent` runs the loop: the model reads the question and the tool cards, and either answers or returns a **structured tool call**. Our Python runs the tool and sends the result back as a tool message. This is **native tool calling**: the provider's API carries the call as data, so nothing has to be fished out of a paragraph of text.

`agent.py` (the important lines):

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

MAX_MODEL_CALLS = 8          # the iteration cap (the old max_iterations)
RECURSION_LIMIT = 50         # LangGraph's own step limit, a last safety net

def build_agent(tools, llm=None, max_model_calls=MAX_MODEL_CALLS, max_tokens=TOKEN_BUDGET, ...):
    if llm is None:
        llm = get_llm()
    caps = [ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"), token_budget(max_tokens)]
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt_for(tools),
                        middleware=[*caps, ...])

def run_agent(question, tools, llm=None, **caps) -> list:
    agent = build_agent(tools, llm, **caps)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]},
                          {"recursion_limit": RECURSION_LIMIT})
    return result["messages"]
```

A **middleware** (Episode 10) runs before or after every model call. Ours are the caps: `ModelCallLimitMiddleware` ends the run politely after 8 model calls, and `token_budget` (Step 14) ends it when 30,000 tokens are used up. `system_prompt_for(tools)` adds one rule per tool the agent actually holds ("For every calculation, call the calculator..."), so the prompt never mentions a tool the agent does not have. `print_trace` prints every message of the run, numbered.

```bash
python ask.py fare
```

Expected shape (Gemini writes the expression and the last sentence; with a correct expression the tool returns exactly `169`):

```text
Question: An auto-rickshaw charges Rs 30 for the first 1.5 km and Rs 17 for every km after that. Waiting adds Rs 20. What is the fare for an 8.5 km ride with waiting?
Tools: calculator
Model: gemini-3.8-flash   Caps: 8 model calls, 30,000 tokens

Trace (4 messages):
   1. human  An auto-rickshaw charges Rs 30 for the first 1.5 km and Rs 17 for every km after that. Waiting ad...
   2. ai     calls calculator({"expression": "<the expression Gemini writes, e.g. 30 + (8.5 - 1.5) * 17 + 20>"})
   3. tool   calculator -> 169
   4. ai     <a short answer saying Rs 169>
Tool calls: 1 (calculator)

Answer: <a short answer saying Rs 169>
Tokens: <in> in + <out> out = <total> (thinking <t> of the out), 2 model calls
```

(The fare chart is invented for the example. Real auto fares differ by city.)

> **Classic vs 2026.** The course this series covers built this with `Tool(name, func, description)`, `load_tools(["llm-math"], llm=llm)` and `initialize_agent(..., agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, max_iterations=3)`. That old agent asked the model to write `Thought / Action / Action Input` lines as plain text and then **parsed** them, so one badly formatted line broke the run. In our 2026 environment `from langchain.agents import initialize_agent` fails (`cannot import name 'initialize_agent'`); it lives on only in `langchain_classic.agents`, and `load_tools(["llm-math"])` does not start at all: `ModuleNotFoundError: Module langchain_community.agent_toolkits.load_tools not found. Please install langchain-community ...` (the message is misleading: the package is installed, an extra library it needs is not). `create_agent` + `@tool` replaces all three.

> [!WARNING]
> **Common errors**
> - The trace shows the answer but `Tool calls: 0` - the model did the sum in its head. Run it again, or use `LLM_MODEL=gemini-3.8-flash`. The rule in the system prompt asks for the tool, but a model can still skip it: that is why we read traces.
> - `calculator -> Error: ...` and then a second call - the model wrote something the tool refused (for example `%` or `^`), read the reason and tried again. That is the tool doing its job.
> - `Answer: Model call limits exceeded: run limit (8/8)` - the call cap stopped a model that kept calling tools. See the Fix-It Corner.

## Step 4 - The same agent, a question that needs no tool

A common myth: "an agent must use a tool". It need not. The model decides, every time.

```bash
python ask.py general
```

Expected shape (Gemini's sentence will differ):

```text
Question: In one sentence, what is a mandi?
Tools: calculator
Model: gemini-3.8-flash   Caps: 8 model calls, 30,000 tokens

Trace (2 messages):
   1. human  In one sentence, what is a mandi?
   2. ai     <one sentence about a mandi, written by Gemini>
Tool calls: 0 (the model answered by itself)

Answer: <the same sentence>
Tokens: <in> in + <out> out = <total> (thinking <t> of the out), 1 model call
```

The course this series covers hit a crash here and added a second "general knowledge" tool (the same model wrapped as a tool) to get around it. A tool-calling agent simply answers.

> [!WARNING]
> **Common errors**
> - The agent calls the calculator for this question - unusual, but possible with a small model. Read the trace: the description of a tool is what invites the call.

## Step 5 - Seed the database with SQLAlchemy Core

SQLAlchemy lets us describe tables in Python. `MetaData` holds every `Table`, each `Table` holds its `Column`s with their types, the **engine** knows how to reach the database, and `engine.begin()` opens a **transaction**: all rows go in, or none do. We use SQLAlchemy **Core** (tables and rows), not its ORM (Python classes per row). `ForeignKey("cities.city")` says that the `city` of a price row points to a row of the `cities` table. Arrivals are counted in **quintals**, a unit used in Indian wholesale markets: 1 quintal = 100 kg.

`seed_db.py` (the important lines):

```python
metadata = MetaData()

daily_prices = Table(
    "daily_prices", metadata,
    Column("id", Integer, primary_key=True),
    Column("day", Integer, nullable=False),                      # 1 to 10, sample days
    Column("city", String(20), ForeignKey("cities.city"), nullable=False),
    Column("vegetable", String(20), nullable=False),             # lowercase: onion, tomato, ...
    Column("price_rs_per_kg", Float, nullable=False),            # invented modal price
    Column("arrivals_quintals", Integer, nullable=False),        # invented
)

engine = create_engine(sqlite_url(db_path))
metadata.create_all(engine)                            # CREATE TABLE for every Table in metadata
with engine.begin() as conn:                           # one transaction: all rows, or none
    conn.execute(insert(cities), [{"city": c, "state": s} for c, s in CITIES])
    conn.execute(insert(daily_prices), make_rows(seed))  # a list of dicts = many rows at once
```

`make_rows()` invents the prices with `random.Random(15)`: the same seed gives the same "random" numbers on every computer. Tomatoes drift up about 3 percent a day, so one question has a clear rise to measure.

```bash
python seed_db.py
```

Expected output (exact, every time):

```text
mandi.db, 150 rows
  daily_prices  150 rows  (5 vegetables x 3 cities x 10 days)
  cities        3 rows
Sample data, not real prices.
```

> [!WARNING]
> **Common errors**
> - `PermissionError` on Windows - another program (an open DB browser, a second terminal running the agent) still holds `mandi.db`. Close it, then run again. Running `seed_db.py` deletes the old file and builds it fresh.
> - Never commit `mandi.db`: it is in this folder's `.gitignore`, and anyone can rebuild it in a second.

## Step 6 - Read-only SQL tools

A new analyst on day one first asks which tables exist, then reads the columns, then writes the query. Our agent gets exactly those three tools. `run_select` sends every query through Episode 12's guard (`tools/ep12_sql_guard.py`): one `SELECT` only with a `LIMIT` added, a read-only connection, an allow-list of tables and columns, and size and time limits. Each tool **returns** problems as text instead of raising: an exception inside a tool would stop the whole agent, while a returned message lets the model fix its query.

`tools/sql.py` (the important lines):

```python
def read_only_engine(db_path=None):
    """A SQLAlchemy engine that can only READ: every connection is Episode 12's open_read_only (mode=ro)."""
    db_path = db_path or DB_PATH
    return create_engine("sqlite://", creator=lambda: open_read_only(db_path))

def make_sql_tools(db_path=None, careful=False) -> list:
    ...
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
        ...

    def run_select(query: str) -> str:
        return safe_query(query, db_path=db_path, allowed=ALLOWED, max_rows=MAX_ROWS)   # never raises

    description = RUN_SELECT_CAREFUL if careful else RUN_SELECT_BASIC
    return [list_tables, describe_table, tool("run_select", description=description)(run_select)]
```

Run the tools by hand (no model, no key):

```bash
python -m tools.sql
```

Expected output (exact):

```text
list_tables() -> cities, daily_prices

describe_table("daily_prices") ->
Table daily_prices: 150 rows (sample data, not real prices)
Columns:
  id                 INTEGER      from 1 to 150
  day                INTEGER      from 1 to 10
  city               VARCHAR(20)  values: Indore, Nashik, Pune
  vegetable          VARCHAR(20)  values: brinjal, cabbage, onion, potato, tomato
  price_rs_per_kg    FLOAT        from 11.5 to 38.1
  arrivals_quintals  INTEGER      from 103 to 1080
3 example rows:
id  day  city    vegetable  price_rs_per_kg  arrivals_quintals
--  ---  ------  ---------  ---------------  -----------------
1   1    Nashik  onion      22.3             724
2   2    Nashik  onion      23.0             777
3   3    Nashik  onion      24.6             726
(3 rows)

run_select("SELECT city, price_rs_per_kg FROM daily_prices WHERE vegetable = 'onion' AND day = 3 ORDER BY price_rs_per_kg") ->
SQL used: SELECT city, price_rs_per_kg FROM daily_prices WHERE vegetable = 'onion' AND day = 3 ORDER BY price_rs_per_kg LIMIT 20
city    price_rs_per_kg
------  ---------------
Indore  22.9
Pune    23.5
Nashik  24.6
(3 rows)

run_select("DROP TABLE daily_prices") ->
Blocked: only SELECT queries can run here (this one starts with DROP).

run_select("SELECT * FROM daily_prices; DELETE FROM daily_prices") ->
Blocked: only one statement at a time (found a second one after ';').

run_select("SELECT name FROM sqlite_master") ->
Blocked: table sqlite_master is not on the allow-list.
```

`describe_table` shows the exact values (`onion`, lowercase; `Indore`, capital I), because a model that guesses `'Onion'` gets zero rows. And look at the cheapest onions on day 3: **Indore**, not Nashik. If you "know" Nashik is famous for onions, so does the model. The data is invented, and only the query knows.

> [!WARNING]
> **Common errors**
> - `mandi.db not found. Create it first with:  python seed_db.py` - Step 5 first.
> - `Error: no such column: price` - the column is `price_rs_per_kg`. This is the most common model mistake too, which is why the agent should call `describe_table` first.

## Step 7 - Give the agent the SQL tools

Now the agent has to walk the same path: discover, describe, query, answer.

```bash
python ask.py onions
```

Expected shape (Gemini picks the calls and writes the SQL and the answer; the tool results below are what our tools return for those calls):

```text
Question: Which city had the cheapest onions on day 3?
Tools: list_tables, describe_table, run_select
Model: gemini-3.8-flash   Caps: 8 model calls, 30,000 tokens

Trace (8 messages):
   1. human  Which city had the cheapest onions on day 3?
   2. ai     calls list_tables({})
   3. tool   list_tables -> cities, daily_prices
   4. ai     calls describe_table({"table_name": "daily_prices"})
   5. tool   describe_table -> Table daily_prices: 150 rows (sample data, not real prices)
                               ...
   6. ai     calls run_select({"query": "<a SELECT written by Gemini>"})
   7. tool   run_select -> SQL used: <the same SELECT> LIMIT 20
                           ...
   8. ai     <an answer naming Indore>
Tool calls: 3 (list_tables, describe_table, run_select)

Answer: <an answer naming Indore>
Sources (what the tools returned):
  SQL used: <the same SELECT> LIMIT 20
Tokens: <in> in + <out> out = <total> (thinking <t> of the out), 4 model calls
```

The model may skip `list_tables` or make an extra call; any path is fine if the answer is right. The next step checks that.

**Want a human to approve every query?** LangChain's `HumanInTheLoopMiddleware` pauses the agent before a tool runs. `python ask.py onions --approve-sql` turns it on for `run_select`: the terminal shows the SQL and asks `Run this SQL? [y/N]`. Say no, and the model is told a human refused. (It is optional here, because the SQL is already read-only. For tools that change things, such as sending an email, it is the rule.)

```python
gate = HumanInTheLoopMiddleware(interrupt_on={"run_select": {"allowed_decisions": ["approve", "reject"]}})
agent = build_agent(tools, llm, extra_middleware=[gate], checkpointer=InMemorySaver())   # a pause needs a checkpointer
```

> **Classic vs 2026.** The course this series covers used `create_sql_agent(llm, toolkit=SQLDatabaseToolkit(db=db, llm=llm), agent_type=...)`, an `AgentExecutor` underneath. The toolkit still imports from the archived `langchain-community` package and gives four tools (`sql_db_query`, `sql_db_schema`, `sql_db_list_tables`, `sql_db_query_checker`). LangChain's own 2026 SQL agent tutorial (https://docs.langchain.com/oss/python/langchain/sql-agent) uses small `@tool` functions with `create_agent` instead, the pattern we use here, plus our guard.

> [!WARNING]
> **Common errors**
> - The answer says Nashik - the model answered from what it "knows" about Nashik without querying, or misread the rows. Read the trace. Step 8 catches this automatically.
> - `run_select -> Blocked: ...` followed by a second call - the guard said no and the model rewrote its query. Working as intended.

## Step 8 - Grade the agent against ground truth

Checking a bill against the receipt before you pay: `data/questions.json` holds 5 questions, and each has a **truth query we wrote by hand**. `check_answers.py` runs our query, asks the agent the same question (adding "End your reply with ANSWER: <value>"), reads the value and compares. Numbers pass within 1 percent (so `23.1` passes for `23.13`); city names must match.

First, the truths (no key needed):

```bash
python check_answers.py --truths
```

Expected output (exact):

```text
1. Which city had the cheapest onions on day 3?
   truth SQL: SELECT city FROM daily_prices WHERE vegetable = 'onion' AND day = 3 ORDER BY price_rs_per_kg ASC LIMIT 1
   right answer: Indore
2. What was the average brinjal price in Indore over the ten days, in rupees per kg?
   truth SQL: SELECT ROUND(AVG(price_rs_per_kg), 2) FROM daily_prices WHERE vegetable = 'brinjal' AND city = 'Indore'
   right answer: 33.86
3. On which day did Pune get the most potato arrivals?
   truth SQL: SELECT day FROM daily_prices WHERE vegetable = 'potato' AND city = 'Pune' ORDER BY arrivals_quintals DESC LIMIT 1
   right answer: 10
4. How many quintals of onions reached Nashik and Pune on day 3, taken together?
   truth SQL: SELECT SUM(arrivals_quintals) FROM daily_prices WHERE vegetable = 'onion' AND day = 3 AND (city = 'Nashik' OR city = 'Pune')
   right answer: 1075
5. By what percent did the tomato price in Pune rise from day 1 to day 10?
   truth SQL: SELECT ROUND((d10.price_rs_per_kg - d1.price_rs_per_kg) * 100.0 / d1.price_rs_per_kg, 2) FROM daily_prices AS d1 JOIN daily_prices AS d10 ON d10.vegetable = d1.vegetable AND d10.city = d1.city WHERE d1.vegetable = 'tomato' AND d1.city = 'Pune' AND d1.day = 1 AND d10.day = 10
   right answer: 23.13
```

(The tests check these five truth queries against plain Python over the seed rows, so the checker is checked too.)

Question 4 is the **trap**. We say "Nashik **and** Pune", but in SQL it is `city = 'Nashik' OR city = 'Pune'`. And SQL applies `AND` before `OR`, just like BODMAS applies `*` before `+`. Without brackets, `vegetable = 'onion' AND day = 3 AND city = 'Nashik' OR city = 'Pune'` means "(Nashik onions on day 3) OR (anything at all from Pune)". See it with a **scripted** model (no AI, no key; the tools, the database and the grader are real):

```bash
python check_answers.py --replay-trap
```

Expected output (exact; the `[SCRIPTED]` lines are written by our test script, not by an AI):

```text
Replay: a SCRIPTED model (no AI, no key) writes the SQL. The tools, the database and the grader are real.

Question 4: How many quintals of onions reached Nashik and Pune on day 3, taken together?
Truth (our SQL): 1075
Why it is a trap: In English we say 'Nashik and Pune', but in SQL it is city = 'Nashik' OR city = 'Pune'. SQL applies AND before OR, so without brackets around the OR part every Pune row (all days, all vegetables) leaks into the sum.

--- Without brackets ---
rows the WHERE clause lets through: 51
Trace (4 messages):
   1. human  How many quintals of onions reached Nashik and Pune on day 3, taken together?
             End your reply with one last line in this form: ANSWER: <only the number or the city name, no units>
   2. ai     calls run_select({"query": "SELECT SUM(arrivals_quintals) FROM daily_prices WHERE vegetable = 'onion' AND day = 3 AND city = 'Nashik' OR city = 'Pune'"})
   3. tool   run_select -> SQL used: SELECT SUM(arrivals_quintals) FROM daily_prices WHERE vegetable = 'onion' AND day = 3 A...
                           SUM(arrivals_quintals)
                           ----------------------
                           15862
                           (1 row)
   4. ai     [SCRIPTED] Nashik and Pune got 15862 quintals of onions on day 3.
             ANSWER: 15862
Tool calls: 1 (run_select)
Grade: FAIL (expected 1075 (within 1 percent), the agent said 15862)

--- With brackets ---
rows the WHERE clause lets through: 2
Trace (4 messages):
...
Grade: PASS
```

51 rows instead of 2, and 15,862 quintals instead of 1,075. Nothing crashed and the answer even sounds confident: only the check catches it. Now the real thing, your agent on all five questions:

```bash
python check_answers.py
```

Expected shape (your results are whatever your model does; that is the point):

```text
Scoreboard: 5 questions, run_select with its BASIC description
 #  result  truth    agent    tools called                             tokens
 1  <PASS or FAIL>  Indore   <its answer>  <the tools it called>   <tokens>
 2  ...     33.86    ...
 3  ...     10       ...
 4  ...     1075     ...
 5  ...     23.13    ...
Score: <passed>/5
Tokens: <in> in + <out> out = <total> (thinking <t> of the out), <n> model calls
```

Under every FAIL the scoreboard prints the question, what it expected and the SQL the agent ran, so you can see exactly where it went wrong. A strong model may well write `city IN ('Nashik', 'Pune')` and pass question 4 on the first try. Good: you know that because you checked, not because you hoped.

> [!WARNING]
> **Common errors**
> - `FAIL ... no 'ANSWER:' line in the reply` - the model forgot the last line. Run again; `gemini-3.8-flash` follows it more reliably.
> - A row says `ERROR` (its tokens say `-`) - that question's run stopped with an error: often `429 RESOURCE_EXHAUSTED` (5 questions make 10 or more model calls in a row), or `Could not reach the model's server` (no internet). The reason is printed under the scoreboard. Wait a minute and run again.
> - A different score on every run - normal for a model. Run it two or three times before you trust a fix.

## Step 9 - Fix the trap

Two fixes, both in **our** code, not in the model:

1. **A better description.** The model reads `run_select`'s description before every query. The careful version adds two rules: use the exact names from `describe_table`, and "put brackets around the OR part or use IN".
2. **A parameterised tool.** `get_prices(vegetable, cities, days)`: the model only picks values, and Python writes the SQL with bound parameters (`city IN ('Nashik', 'Pune')`). It cannot get `AND`/`OR` wrong because it never writes SQL.

```python
RUN_SELECT_CAREFUL = (
    RUN_SELECT_BASIC + " Call list_tables and describe_table first and use the exact column names and "
    "values they show. SQL applies AND before OR, so when a WHERE clause mixes them, put brackets around "
    "the OR part or use IN, for example: WHERE vegetable = 'onion' AND day = 3 AND city IN ('Nashik', 'Pune')."
)
```

```bash
python check_answers.py --fix description
python check_answers.py --fix price-tool
```

Expected shape (the goal is 5/5; your run decides):

```text
Scoreboard: 5 questions, run_select with the CAREFUL description (the fix)
 #  result  truth    agent    tools called                             tokens
 1  PASS    Indore   Indore   <tools>                                  <tokens>
 ...
Score: 5/5
Tokens: <in> in + <out> out = <total> (thinking <t> of the out), <n> model calls
```

Happy with the description fix? Make it permanent: in `tools/sql.py` set `USE_CAREFUL_DESCRIPTION = True`. From then on `ask.py` uses it too.

> [!WARNING]
> **Common errors**
> - Still 4/5 - read the FAIL lines. If the SQL is right but the number is wrong, the model misread the rows (try `--fix price-tool`); if the SQL is wrong in a new way, put that rule in the description. Re-run after every change: the scoreboard is how you know a fix worked.
> - 5/5 once is not 5/5 forever. Models change. Keep the scoreboard and run it again when you change the model, a prompt or a tool.

## Step 10 - Percent change: SQL, then the calculator

Some questions need two kinds of tool in a row: the database for the numbers, the calculator for the maths.

```bash
python ask.py percent
```

Expected shape (the right answer is 23.13 percent: 29.4 on day 1, 36.2 on day 10):

```text
Question: By what percent did the tomato price in Pune rise from day 1 to day 10?
Tools: calculator, list_tables, describe_table, run_select
Model: gemini-3.8-flash   Caps: 8 model calls, 30,000 tokens

Trace (<n> messages):
   1. human  By what percent did the tomato price in Pune rise from day 1 to day 10?
   ...    (maybe list_tables / describe_table first)
  <k>. ai     calls run_select({"query": "<a SELECT for day 1 and day 10>"})
  <k+1>. tool   run_select -> SQL used: ... (rows with 29.4 and 36.2)
  <k+2>. ai     calls calculator({"expression": "<e.g. (36.2 - 29.4) / 29.4 * 100>"})
  <k+3>. tool   calculator -> 23.129252
  <n>. ai     <an answer of about 23.13 percent>
Tool calls: <n> (..., run_select, calculator)
```

Look for the order in the trace: a SQL call **followed by** a calculator call. The course's SQL agent did its maths inside SQL (its math agent was a separate program); here one agent holds both, and the tool rules in the prompt send arithmetic to the calculator. (Doing the maths in SQL would also be correct. The trace tells you which one happened.)

> [!WARNING]
> **Common errors**
> - `calculator -> 23.129252` but the answer says `23.1` - fine: rounding in the sentence. The scoreboard allows 1 percent.
> - The answer says the price **fell** - the model swapped day 1 and day 10 in its expression. Read the calculator line in the trace.

## Step 11 - Crawl with a 3-page cap (online, optional)

Sending a courier to collect three named pages, not the whole library. An **Actor** is a ready-made program that runs on Apify's servers; the **Website Content Crawler** visits pages and returns their clean text in a dataset. We crawl our own rule book (`site/`, three pages written for this course), and cap everything: 3 pages, 1 link deep, 1 page at a time, the cheapest crawler type, 1 GB of memory, 5 minutes.

`crawl.py` (the important lines):

```python
def build_run_input(start_url: str, max_pages: int = MAX_PAGES) -> dict:
    return {
        "startUrls": [{"url": start_url}],
        "maxCrawlPages": max_pages,          # the page cap (the default is 9,999,999: effectively no cap)
        "maxCrawlDepth": 1,                  # only pages linked straight from the start page
        "maxResults": max_pages,             # never store more pages than that
        "maxConcurrency": 1,                 # one page at a time: polite to the site
        "crawlerType": "cheerio",            # plain HTTP, no browser: the cheapest; our pages need no JavaScript
        "respectRobotsTxtFile": True,        # the Actor checks robots.txt as well
        "proxyConfiguration": {"useApifyProxy": False},   # our own site does not need a proxy
    }

run = client.actor(ACTOR_ID).call(run_input=build_run_input(start_url, max_pages),
                                  memory_mbytes=MEMORY_MB, run_timeout=RUN_TIMEOUT)
if run is None:                                       # v3: call() can return None
    raise CrawlError("Apify returned no run (call() gave None). ...")
items = client.dataset(run.default_dataset_id).iterate_items()   # v3: an attribute, not run["defaultDatasetId"]
```

First, without any account, check permission and see exactly what would be sent:

```bash
python crawl.py --check-only
```

Expected output (exact on the day this README was written, 2026-10-26, before Ravi published the pages; once they are online the last-but-one line says `HTTP 200, online`):

```text
Start page:   https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/
robots.txt:   https://replyravi.github.io/robots.txt -> HTTP 404, allowed for us: True
Actor:        apify/website-content-crawler   memory 1024 MB, stopped after 5 minutes
Actor input (the caps are in here):
{
  "startUrls": [
    {
      "url": "https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/"
    }
  ],
  "maxCrawlPages": 3,
  "maxCrawlDepth": 1,
  "maxResults": 3,
  "maxConcurrency": 1,
  "crawlerType": "cheerio",
  "respectRobotsTxtFile": true,
  "proxyConfiguration": {
    "useApifyProxy": false
  }
}
Start page:   HTTP 404, NOT online yet (publish site/ first, see the README)
Check only: Apify was not called.
```

`robots.txt` always lives at the root of a site (here `replyravi.github.io/robots.txt`), never in a sub-folder. HTTP 404 means the site has no robots.txt, so it sets no rules; `crawl.py` still checks it on every run, because a site can add one any day. robots.txt is only half of the permission: the other half is the site's purpose and terms. These pages are ours, written to be crawled.

Then the real crawl. Apify's terms say an account holder must be **18 or older** (checked 2026-10-26), and you are responsible for what you crawl; younger viewers skip to Step 12, which uses the fixture. Add your token to the root `.env` (`APIFY_API_TOKEN=...`, from the API & Integrations page of the Apify console) and run:

```bash
python crawl.py
```

Expected shape (your run id, times and cost will differ; the console shows the same numbers):

```text
...
Running the Actor (at most 3 pages). Waiting for it to finish ...
  1. https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/  <n> characters  "Mandi Tool Belt rule book"
  2. https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/<fees.html or units.html>  ...
  3. ...
Saved crawl.json: 3 pages
Run time <s> s, <cu> compute units, cost $<usd>
See the run and its cost in the console: https://console.apify.com/actors/runs/<run id>
```

**Cost, roughly.** Apify bills compute units (1 CU = 1 GB of memory for 1 hour). Our caps allow at most 1 GB for 5 minutes, about 0.08 CU; what a run really used is printed after it and shown in the console. Apify's pricing page offered $5 of free platform usage per month on the free plan on 2026-10-26; prices and plans change, so check https://apify.com/pricing before you rely on it. Without the caps, the Actor's defaults were 8 GB of memory and no real page limit.

**Your own copy of the pages.** Forked the repo? Turn on GitHub Pages for your fork (Settings > Pages > Deploy from a branch > `main`, folder `/ (root)`), then add `CRAWL_START_URL=https://<your-user>.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/` to the root `.env`. Only ever point `--url` at a site whose robots.txt **and** terms allow it.

> **Classic vs 2026.** The course this series covers used `ApifyWrapper().call_actor(...)` with no page limit, then `VectorstoreIndexCreator().from_loaders([loader])` and `index.query_with_sources(...)`, which chunked and embedded everything silently and re-crawled on every run. In our 2026 environment both fail before they start: the community `ApifyWrapper(...)` prints a deprecation notice and then `AttributeError: 'ImpitHttpClient' object has no attribute 'httpx_client'` (it does not work with apify-client 3.3, the version we use), and `from langchain_classic.indexes import VectorstoreIndexCreator` raises `ImportError: The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed`. We call apify-client 3 directly, save the crawl once, and index it ourselves in Step 12, where you can see every chunk.

> [!WARNING]
> **Common errors**
> - `No APIFY_API_TOKEN found in your .env file.` - this step is optional. Skip it: Step 12 uses `fixtures/crawl.json`.
> - `Start page: HTTP 404, NOT online yet` / `Crawl failed: The crawl finished but found no text.` - the pages are not published at that address yet. Use the fixture, or publish your own copy (above).
> - `Apify call failed: ... 401` (or `user-or-token-not-found`) - the token is wrong or has spaces around it. Copy it again into `.env`.
> - `robots.txt says no. We do not crawl this site.` - then we don't. Pick a site that allows it.
> - `Crawl failed: The run ended with status FAILED` - open the run in the console (the link is printed) and read its log. If it ran out of memory, raise `MEMORY_MB` in `crawl.py` to 2048.

## Step 12 - Index the crawl (or the fixture)

Exactly Episode 07's indexing pipeline, now for web pages: load (one Document per page, labelled with its URL), split (400 characters, 60 overlap), embed (384 numbers per chunk, locally), store (Chroma in `index/`, with stable ids so a second run adds nothing twice). `ingest.py` uses your `crawl.json` if Step 11 made one, else the fixture.

`fixtures/crawl.json` is an offline copy of the same three pages, made on this computer from `site/` by `fixtures/make_fixture.py`, in the shape `crawl.py` saves. It is **not** a recorded Apify run: Apify's text extraction may break the lines a little differently, so your chunk count can differ slightly after a real crawl.

```bash
python ingest.py
python ingest.py
```

Expected output, first run (exact, apart from the model's loading lines):

```text
Read fixtures/crawl.json: 3 pages
  (Offline copy of the 3 rule-book pages in site/, made by fixtures/make_fixture.py in the same shape crawl.py saves. Not a recorded Apify run.)
  https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/  1,299 characters  "Mandi Tool Belt rule book"
  https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/fees.html  933 characters  "Buyer charges - Mandi Tool Belt"
  https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/units.html  643 characters  "Bags, crates and quintals - Mandi Tool Belt"

Split into 10 chunks (chunk_size=400, chunk_overlap=60)
  site/        5 chunks
  fees.html    3 chunks
  units.html   2 chunks
  Example chunk site/#2: 'What the price board shows\nThe board lists one price a day for five vegetables: onion, tom'...

Embedding new chunks and saving them to index/ ...
10 chunks indexed, dim 384 (new this run: 10, removed: 0)
```

Second run (last line):

```text
10 chunks indexed, dim 384 (new this run: 0, removed: 0)
```

> [!WARNING]
> **Common errors**
> - Extra lines such as `NOTE: Redirects are currently not supported in MacOs`, `Loading weights: 100%` or `You are sending unauthenticated requests to the HF Hub` - harmless, from the embeddings model.
> - First run without internet: `OSError: We couldn't connect to 'https://huggingface.co'` - the model downloads once (about 90 MB). Connect and run again.
> - An error about the embedding **dimension** after changing `EMBEDDINGS_PROVIDER` - vectors from different models cannot be mixed. Delete `index/` and run `python ingest.py` again.

## Step 13 - Retrieval as a tool, with source URLs

In Episode 07 our code always searched first and then asked the model. Now the model **decides** when to search and what to search for, like with any other tool. Every passage carries its URL, so the answer can say where a rule came from.

`tools/docs.py` (the important lines):

```python
@tool
def search_docs(query: str) -> str:
    """Search the Mandi Tool Belt rule book (crawled web pages) for rules about buyer charges,
    commission, market fee, loading, bag and crate sizes, quintals, grades and how to read the
    price board. Returns the 3 closest passages, each with its source URL. Prices are NOT here:
    use the SQL tools for prices and arrivals.
    """
    ...
    results = get_store().similarity_search_with_score(query, k=k)
    ...
        blocks.append(f"[{number}] {chunk_tag(doc)}  source: {doc.metadata['source']}  "
                      f"(distance {distance:.3f})\n{doc.page_content}")
```

The last sentence of the description matters: it tells the model what is **not** in the documents, so it does not search the rule book for a price.

```bash
python -m tools.docs "loading charge per bag"
```

Expected output (exact, apart from the model's loading lines):

```text
search_docs("loading charge per bag") ->
[1] fees.html#2  source: https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/fees.html  (distance 0.403)
Market fee
Every bill also carries a market fee of 1 percent of the goods value. The fee pays for the yard, the lights and the weighbridge.
Loading charge
Loading costs Rs 12 for each bag or crate that porters carry to the buyer's vehicle. Loose loads cost Rs 25 per quintal to load.
How a bill adds up
Total bill = goods value + commission + market fee + loading charge.

[2] fees.html#3  source: https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/fees.html  (distance 0.531)
Example: 2 bags of potatoes weigh 100 kg. At Rs 20 per kg the goods value is Rs 2,000. The commission is Rs 80, the market fee is Rs 20 and loading is Rs 24, so the total bill is Rs 2,124.

[3] site/#4  source: https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/  (distance 0.619)
Questions the agent answers
The agent answers price questions from the board, explains the charges a buyer pays on top of the price, and converts between bags, crates and kilograms. The charges are on the Buyer charges page. Bag and crate sizes are on the Bags, crates and quintals page.
Rounding
```

(Result 3 ends with a lonely heading, `Rounding`: the splitter cut between a heading and its paragraph. Real crawls are full of such seams; Episode 08 compares ways to split.)

Now the whole belt:

```bash
python ask.py --list-tools
```

Expected output (exact, no key needed):

```text
The agent holds 5 tools:
  1. calculator(expression)       Calculate an arithmetic expression exactly, like '12 * 154' or '(938.40 / 12) * 17'.
  2. list_tables()                List the tables in the mandi price database.
  3. describe_table(table_name)   Show the columns of one table, the values each column holds, and 3 example rows.
  4. run_select(query)            Run one read-only SQL SELECT query on the mandi price database and return the rows.
  5. search_docs(query)           Search the Mandi Tool Belt rule book (crawled web pages) for rules about buyer ch...
Caps: at most 8 model calls and 30,000 tokens per question (recursion_limit 50 as a last safety net)
```

Five tools of three kinds: the calculator, three SQL tools, and the document search.

> [!WARNING]
> **Common errors**
> - `Error: The document index is empty. Build it first with:  python ingest.py` - the tool returns this to the model instead of crashing. Run Step 12.
> - Each `python -m tools.docs` run takes a few seconds: it loads the embeddings model first. In `ask.py` that happens only on the first search of a run.

## Step 14 - The token meter

The course this series covers printed a token count of **zero** for its SQL agent, and nobody noticed. Our meter does not depend on a provider-specific counter: every AI message LangChain returns carries `usage_metadata`, so we add up the messages of the run. With Gemini, `output_tokens` already includes the **thinking** tokens (`output_token_details["reasoning"]`), which is why the meter shows them inside the "out" number. A model that reports nothing is shown as **not reported**, never as a silent 0.

`token_meter.py` (the important lines):

```python
def usage_of(message) -> dict | None:
    """The usage of one AI message, or None when the provider reported none."""
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    details = usage.get("output_token_details") or {}
    return {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0),
            "thinking": details.get("reasoning", 0), "total": usage.get("total_tokens", 0)}

def token_budget(max_tokens: int = TOKEN_BUDGET):
    """A middleware that stops the agent before a model call once max_tokens are used up."""
    @before_model(can_jump_to=["end"])
    def check_token_budget(state, runtime):
        used = meter(state["messages"])["total"]
        if used < max_tokens:
            return None
        notice = AIMessage(content=f"Stopped: the token budget of {max_tokens:,} was used up ({used:,} tokens).",
                           name="token_budget")
        return {"messages": [notice], "jump_to": "end"}
    return check_token_budget
```

Every `ask.py` run ends with the `Tokens:` line and saves itself in `runs/last_run.json`. See the last run call by call:

```bash
python token_meter.py
```

Before any successful run you see (exact):

```text
No saved run yet. Ask the agent something first, for example:  python ask.py onions
```

After a run, the expected shape is:

```text
Last run: Which city had the cheapest onions on day 3?
call  asked for                               in    out  thinking   total
   1  list_tables                          <in>  <out>      <t>  <total>
   2  describe_table                        ...
   3  run_select                            ...
   4  final answer                          ...
Tokens: <in> in + <out> out = <total> (thinking <t> of the out), 4 model calls
```

Look at the `in` column: it grows with every call, because each call re-sends the whole conversation so far (question, tool cards, every tool result). That is why caps matter: a model that keeps calling tools costs more on **every** extra step, not just a little.

> [!WARNING]
> **Common errors**
> - `Tokens: not reported by this model` - your provider or version sent no usage data. The meter says so instead of printing 0.
> - `Answer: Stopped: the token budget of 30,000 was used up (...)` - the budget cap ended the run. Raise `TOKEN_BUDGET` in `token_meter.py` only if you know why the run needs more.

## Step 15 - One question that needs docs, SQL and maths

A fixer who phones the warehouse, checks the rule book and uses a calculator before replying. The bill question needs all three: the rule book for the bag size and the charges, the database for the day 7 price, and the calculator for the total.

```bash
python ask.py full
```

Expected shape: the one at the top of this page. In the trace, look for `search_docs` (once or twice), `run_select` and `calculator`, in some order, and two kinds of source at the end: a page URL and the SQL used.

**Check it by hand.** Every tool is plain Python, so you can run the agent's steps yourself:

```bash
python -m tools.docs "how many kg in one onion bag"
python -m tools.sql "SELECT price_rs_per_kg FROM daily_prices WHERE city = 'Pune' AND vegetable = 'onion' AND day = 7"
python -m tools.calc "3 * 50 * 25.7 * (1 + 4 / 100 + 1 / 100) + 3 * 12"
```

The rule book says one onion bag holds 50 kg, commission is 4 percent, the market fee 1 percent and loading Rs 12 per bag. Expected output of the last two (exact):

```text
SQL used: SELECT price_rs_per_kg FROM daily_prices WHERE city = 'Pune' AND vegetable = 'onion' AND day = 7 LIMIT 20
price_rs_per_kg
---------------
25.7
(1 row)
```

```text
calculator("3 * 50 * 25.7 * (1 + 4 / 100 + 1 / 100) + 3 * 12") -> 4083.75
```

150 kg at Rs 25.70 is Rs 3,855 of goods; with 5 percent of charges and Rs 36 of loading, the bill is **Rs 4,083.75**. (The rule book also says the buyer pays it rounded to the nearest rupee, Rs 4,084, so an answer that gives both is right too.) If the agent says something else, its trace shows which step went wrong.

> [!WARNING]
> **Common errors**
> - The agent applies the bulk rate (3 percent) - only bills over 10 quintals get it; 3 bags are 1.5 quintals. It read the rule but not the condition: a good line for the system prompt.
> - `Answer: Model call limits exceeded: run limit (8/8)` - this question needs at least 3 tool calls plus the answer. If a model keeps searching, the cap stops it. Ask again, or try `gemini-3.8-flash`.

## Step 16 - Test it without internet

The tests use a **scripted fake chat model** (it "asks" for tools with `tool_call(...)` and answers with text we wrote), a **mocked Apify client** that returns a real apify-client 3 `Run` object, fake embeddings for the plumbing, and the real local MiniLM model for one search test. They need no key and block the internet on purpose.

```bash
python -m pytest
```

Expected output:

```text
........................................................................ [ 65%]
.............s........................                                   [100%]
109 passed, 1 skipped in 5.74s
```

Your time will differ. An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes.

How a test checks the whole SQL walk (from `tests/test_agent_fake.py`):

```python
def test_sql_walk_discover_describe_query_answer(temp_db):
    llm = fake_llm(tool_call("list_tables", {}),
                   tool_call("describe_table", {"table_name": "daily_prices"}),
                   tool_call("run_select", {"query": "SELECT city, price_rs_per_kg FROM daily_prices "
                                                     "WHERE vegetable = 'onion' AND day = 3 ORDER BY price_rs_per_kg"}),
                   "[SCRIPTED] Indore had the cheapest onions on day 3.")
    messages = run_agent(ask.PRESETS["onions"][1], tool_belt(["sql"], db_path=temp_db), llm)

    assert [name for name, _ in tool_calls(messages)] == ["list_tables", "describe_table", "run_select"]
```

The model is scripted; everything else is real: `create_agent`'s loop, our tools and a temporary copy of the database. `tests/test_checker.py` runs the whole scoreboard the same way, with one answer wrong on purpose, and expects `Score: 4/5`.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'agent'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called a real model. Pass `llm=fake_llm(...)`.
> - `108 passed, 2 skipped` - run `python -m pytest -rs`. "the local MiniLM model is not downloaded yet" means: run `python ingest.py` once with internet, then test again.

## Fix-It Corner: where agent answers go wrong

| Symptom | Check first |
|---|---|
| Wrong number, confident sentence | The scoreboard (Step 8), then the SQL under the FAIL line. |
| The agent never calls a tool | The tool's description: does it say when to use it? The system prompt's tool rules. |
| The agent calls the wrong tool | Two descriptions that sound alike. Say what each tool does NOT do (like `search_docs`: "Prices are NOT here"). |
| `Error: no such column` in the trace | Normal once: the model should call `describe_table` and try again. Every time: make `describe_table` part of the rules. |
| `AND`/`OR` rows leaking | Brackets or `IN` (Step 9), or a parameterised tool. |
| `is not a valid tool, try one of [...]` | The model asked for a tool this agent does not hold. Check `--tools` / `tool_belt()`. |
| `Model call limits exceeded` | The model kept calling tools. A clearer question, a clearer description, or a smarter model. |
| `Tokens: not reported` | The provider sent no usage data; the meter refuses to pretend it was 0. |

## Final challenge

Add a quintal-to-kilogram converter tool and a question that needs it, then reach a perfect score on the scoreboard with a local Ollama model (`LLM_PROVIDER=ollama`). Details in [CHALLENGE.md](CHALLENGE.md).

## What you learned

- A tool is a name, a description, an argument schema and a return value. The model reads the first three, so the description decides when a tool is used.
- `create_agent` runs the loop with native tool calls; an agent may also answer with no tool at all.
- A safe SQL agent discovers, describes, then queries, on a read-only connection through a guard, and gets errors back as text.
- Never trust an agent's number until you have checked it: a scoreboard with hand-written truths catches confident wrong answers, like rows leaking through `AND`/`OR` without brackets.
- Crawl only where it is allowed, cap every crawl, save it once, index it yourself, and give the agent search as a tool with source URLs.
- Cap every run (model calls, tokens, graph steps) and read the real token usage from every message.

## Next episode

**Episode 16: Hugging Face for Beginners (2026): Run AI Models on Your Own Laptop** - every model so far ran in someone else's data centre. Next we run small models on our own laptop: summarising, answering questions about an image, translating and embedding, with no key at all. -> `ep16-local-ai-toolbox` (coming soon)

---

The Mandi Tool Belt, its prices, charges and rules are invented for teaching. They are not the rates or rules of any real market, market committee or business. Any match is a coincidence.
