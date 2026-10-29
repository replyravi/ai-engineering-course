# Episode 13: AI Agents and Multi-Agent Crews Explained: Build Your First CrewAI Team

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep13-shelf-sifter-crew

## What you will build

`scout.py`, a team of three AI agents built with **CrewAI** that shops for books on [books.toscrape.com](https://books.toscrape.com), a practice bookshop made for learning web scraping:

- the **Scout** reads one shelf of the shop with a polite scraper tool (robots.txt checked, 2 seconds between requests),
- the **Analyst** finds titles that match what the reader likes with a free local embeddings model, and returns typed picks,
- the **Writer** turns the picks into a short Markdown buying guide, `reports/travel_guide.md`.

Then the program checks the **scorecard**: which tools really ran, did any tool fail, and does every pick match what was scraped?

```text
$ python scout.py
Shelf Sifter crew: shelf='travel'  interest='a long journey by train'  agents: scout, analyst, writer
Model: gemini/gemini-3.8-flash   Process: sequential   max_rpm: 10
  ... CrewAI's step-by-step log: every task, every tool call, every answer ...
The Analyst's typed answer: Picks with 3 pick(s)
  1. <a title from the shelf>  £<price>  rating <stars>/5  - <a reason written by Gemini>
  ...
The Writer saved reports/travel_guide.md (<size> characters):
  ...
Scorecard (who actually scored)
  scrape_shelf ............. 1 successful call(s), 11 books from https://books.toscrape.com/catalogue/category/books/travel_2/index.html
  find_similar_books ....... 1 successful call(s)
  tool failures ............ 0
  picks match the shelf .... 3 of 3
  pick links in the guide .. 3 of 3

Usage metrics (result.token_usage)
  requests 4   prompt tokens <n>   completion tokens <n>   reasoning tokens <n>   total <n>

All checks passed.
```

(This is the **shape** of a run. The lines in `<...>` come from Gemini and will differ every time. The 11 books, the URL and the scorecard lines come from our code and the real shop page.)

**You will learn:**

- The **agent loop** (decide, call a tool, read the result, decide again, stop), and how to point at each lap in a real log.
- When to use a plain **chain**, one **agent** or a **crew** of agents: more agents means more model calls, more waiting and more places for errors to pile up.
- CrewAI's building blocks: `Agent`, `Task`, `Crew`, `Process`, `kickoff()` and the output object, with the model named explicitly from `.env`.
- How a model "sees" a tool (a name, a description, typed arguments, a return value), how to write custom tools, and how to read a **red validation error** and fix it.
- How tasks hand work to each other with `context`, how `output_pydantic` gives you a typed Python object, and how `max_iter`, `max_rpm` and `tool_failure_policy` keep a crew honest.
- How to scrape **politely**: permission (robots.txt and the site's purpose), a pause between requests, a clear User-Agent, clean rows, and a saved page for offline tests.

> **Seen an older tutorial?** Five things changed (checked with CrewAI 1.15.28, 2026-10-24 to 2026-10-26):
> - `Agent(role=..., goal=..., backstory=...)` with **no** `llm=` falls back to an OpenAI model (`gpt-4.1-mini` in 1.15.28) and needs an OpenAI key. With our `.env` (its `OPENAI_API_KEY=` line is empty) it stops at once with `Error importing native provider: Missing credentials ... OPENAI_API_KEY`; without that line it fails later, at `kickoff()`, with `OPENAI_API_KEY is required`. We always pass `llm=get_crewai_llm()`, which reads `LLM_PROVIDER` from your `.env` (Gemini by default).
> - "CrewAI is built on LangChain" is **not true**. CrewAI's GitHub page calls it a standalone framework with its own agents, tasks, crews, flows and tools, and the installed `crewai` 1.15.28 needs no LangChain package at all. A LangChain tool must be wrapped with `Tool.from_langchain` (Step 6).
> - Pasting search results into the agent's `goal` with an f-string hides where the answer came from. We pass work between tasks with `Task(context=[...])` (Step 10).
> - There is no "parallel" process. `Process` has two values: `sequential` and `hierarchical` (Step 7).
> - A run whose every tool call failed can still print a confident answer. We make a failed tool call stop the crew (`tool_failure_policy="raise"`) and check the scorecard (Steps 8 and 12).

## Prerequisites

- Finished Episode 06, folder `ep06-helpdesk-ladder` (your first agent with `create_agent`), Episode 08, folder `ep08-chunk-lab` (embeddings) and Episode 11, folder `ep11-party-planner-chains` (tools and typed answers). New here? The one idea you need: an agent is a model in a loop that may call your Python functions (tools).
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15** - CrewAI 1.15 does not install on 3.14 or newer.
- The **CrewAI environment** `.venv-crewai` at the repo root (Step 0 shows how; [SETUP.md](../SETUP.md), part 8) and the root `.env` with your **free** Gemini key from Episode 1.
- Internet for Steps 1-3, 6-8 and 10-12, and once for Step 9 (the model download). The tests (Steps 5 and 13) run offline.
- About 60 minutes. Step 9 downloads a small local model once (about 80 MB).
- That model is [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), released under the **Apache-2.0** licence (free to use, also commercially; keep its licence notice if you share the model files). This episode runs the ONNX copy that Chroma publishes of the same model, so no PyTorch is needed. It is the only model this episode downloads.
- On the free tier, Google may use your prompts to improve its products. Our crew only sends public practice data from a demo shop, so that is fine here.

## Scraping rules for this episode

We read **one** website: [books.toscrape.com](https://books.toscrape.com), part of the [Web Scraping Sandbox](https://toscrape.com). Its home page invites beginners to practise scraping there, and the shop pages say that prices and ratings were assigned at random (checked 2026-10-24 and 2026-10-26). So our "best value" picks are practice, not shopping advice.

1. **Permission first.** The scraper reads robots.txt before the first page. On 2026-10-24 and again on 2026-10-26 `https://books.toscrape.com/robots.txt` answered HTTP 404: no robots.txt, so no rules. The code still checks it every run, because a site can add one any day.
2. **Slow down.** At least 2 seconds between requests (longer if robots.txt asks for it with `Crawl-delay`), a 10-second timeout, one page per tool call.
3. **Say who you are.** Every request sends a User-Agent with the project name and a link.
4. **Only this site.** The tool refuses every other address.
5. **Take only what is on display.** Titles, prices, ratings, stock and links of books. No personal data, no logins, no baskets.
6. **Keep their pages on your computer.** The page you save in Step 3 is git-ignored. The tests use `fixtures/handmade_shelf.html`, a page we wrote by hand with three invented books.

## Chain, agent or crew?

| | Who decides the next step? | Model calls per run | Use it when |
|---|---|---|---|
| **Chain** (Episodes 06, 11) | You, in code | fixed, often 1 | the steps never change |
| **Agent** (Episode 06, Step 7 here) | the model, in a loop with tools | 2 or more, it varies | the model must choose tools or how many laps |
| **Crew** (Steps 10-12 here) | each agent for its own task; you fix the order | several per agent | the job splits into roles that hand work on |

Every extra agent adds calls, waiting time and a new place for a mistake to spread to the next agent. Start with the simplest option that works, and put a human check before any action that matters (Episode 14 does this for emails).

**The agent loop**, the idea behind every log in this episode:

```text
decide  ->  call a tool  ->  read the result  ->  decide again  ->  ...  ->  final answer
                                                   (or stop when max_iter laps are used up)
```

## The agent framework landscape (as of 2026-10-24)

| Framework | One line, dated | Source |
|---|---|---|
| **CrewAI** (we use 1.15.28) | A standalone Python framework with its own agents, tasks, crews, flows and tools. **Crews** are teams of role-based agents; **Flows** are event-driven workflows that can run crews. Does not depend on LangChain. | [github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) README, checked 2026-10-24; installed package metadata |
| **LangGraph** | Built by LangChain Inc, the makers of LangChain, for agent workflows as graphs (one agent or many). LangChain's own agents, like `create_agent` from Episode 06, are built on top of it. | [docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/overview), checked 2026-10-24 |
| **AutoGen** (Microsoft) | Microsoft's README says it is archived and no longer maintained (wording of 2026-10-16) and points new users to **Microsoft Agent Framework**, a Python and .NET framework for agents and multi-agent workflows (1.0 in April 2026). | [github.com/microsoft/autogen](https://github.com/microsoft/autogen), [github.com/microsoft/agent-framework](https://github.com/microsoft/agent-framework), checked 2026-10-24 |

We use CrewAI here because roles and tasks read like a real team. These projects change fast: check the dates.

## What is in this folder

```text
ep13-shelf-sifter-crew/
├── check_setup.py          Step 1: is this the CrewAI environment, and does Gemini answer?
├── robots_check.py         Step 2: permission before scraping
├── save_fixture.py         Step 3: save one shelf page for offline work
├── scraper.py              Steps 3, 4, 6: parse_books() and the polite fetcher
├── tools.py                Steps 6, 9: the scrape_shelf and find_similar_books tools
├── wrap_langchain_tool.py  Step 6: a LangChain tool inside CrewAI
├── crew.py                 Steps 7, 10, 11: the Scout, the Analyst, the Writer and their tasks
├── scout.py                Steps 7, 10-12: run the crew, print the scorecard and usage metrics
├── broken_tool.py          Step 8: break the tool on purpose, then fix it
├── similarity.py           Step 9: local embeddings (all-MiniLM-L6-v2 through Chroma's ONNX copy)
├── models.py               Step 10: the Book, Pick and Picks shapes
├── privacy.py              turns off CrewAI's telemetry, traces and update check
├── fixtures/
│   ├── handmade_shelf.html     3 invented books, written by hand for the tests
│   ├── robots_sample.txt       a practice robots.txt, written by hand
│   └── travel.html             made by Step 3 on YOUR computer (git-ignored)
├── tests/                  offline tests (scripted model, no key, no internet)
├── llm.py, testing/        shared helpers (synced from _template)
├── requirements.txt        ../requirements-crewai.txt plus the versions we tested
├── CHALLENGE.md
└── reports/                made by scout.py (git-ignored)
```

Every file is already here. Type the code yourself as we go, or read along and run it.

## Step 0 - Get ready: the CrewAI environment (about 5 minutes)

Episodes 13 and 14 use a **second** virtual environment, `.venv-crewai`, next to the usual `.venv` at the repo root (`ai-engineering-course/`). Why a second one? CrewAI 1.15 pins older versions of libraries that the LangChain episodes need newer: for Gemini it needs `google-genai` 1.65, while `langchain-google-genai` 4.4 needs 2.20 or newer, and it needs `pydantic` below 2.13. pip cannot install both sets in one environment. The `.env` file stays the same: one file at the repo root.

### 0.1 Create (once) and activate `.venv-crewai`

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**). The first time only, create the environment (skip this if you did SETUP.md part 8):

**Windows (PowerShell):**

```powershell
py -3.13 -m venv .venv-crewai
.\.venv-crewai\Scripts\Activate.ps1
python -m pip install --upgrade pip
cd ep13-shelf-sifter-crew
```

**macOS / Linux:**

```bash
python3.13 -m venv .venv-crewai
source .venv-crewai/bin/activate
python -m pip install --upgrade pip
cd ep13-shelf-sifter-crew
```

Next time, only the activate line and `cd`. Expected: your prompt starts with `(.venv-crewai)` and ends in `ep13-shelf-sifter-crew`.

> [!WARNING]
> **Common errors**
> - `py -3.13` / `python3.13` not found - install Python 3.13 first ([SETUP.md](../SETUP.md), part 1).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`
> - Your prompt shows `(.venv)`, not `(.venv-crewai)` - that is the LangChain environment. Run `deactivate`, then activate `.venv-crewai`. In VS Code pick it with **Ctrl+Shift+P > Python: Select Interpreter**.

### 0.2 Install the CrewAI libraries

From the episode folder (this installs `../requirements-crewai.txt` plus the exact versions this episode was tested with):

```bash
pip install -r requirements.txt
pip check
```

Expected (last line):

```text
No broken requirements found.
```

> [!WARNING]
> **Common errors**
> - `ERROR: ... Requires-Python <3.14` or `No matching distribution found for crewai==1.15.28` - the environment was made with Python 3.14 or 3.15. Delete the `.venv-crewai` folder and make it again with Python 3.13.
> - Intel Mac: an error about `lancedb` - a CrewAI 1.15 dependency has no wheels for Intel Macs. Use Google Colab or a Windows/Linux machine.
> - Using Claude with CrewAI? Also run `pip install "crewai[anthropic]==1.15.28"`.

---

## Step 1 - Check the CrewAI environment

**Why its own environment?** You just saw it: CrewAI pins older shared libraries. This check proves you are in the right environment, that the privacy switches are on, and that Gemini answers through CrewAI's own `LLM` class, the one our agents use.

**Pick the model (optional but suggested).** Crews make several model calls and must use tools carefully. Google's smarter free model is worth it here. In the root `.env` set:

```text
LLM_MODEL=gemini-3.8-flash
```

It uses more tokens by design and is slower than the default `gemini-3.5-flash-lite`; empty the line again after Episode 14. Everything in this episode also runs with the default model.

```bash
python check_setup.py
```

Expected output with your key (your Python patch number, folder name and masked key will differ, and Gemini writes its own reply):

```text
AI Engineering From Zero - CrewAI setup check (Episodes 13-14)

Python .................. 3.13.x OK
Virtual env ............. active (.venv-crewai)
crewai .................. 1.15.28
crewai-tools ............ 1.15.28
langchain-core .......... 1.7.0
beautifulsoup4 .......... 4.15.0
requests ................ 2.34.2
chromadb ................ 1.5.9
onnxruntime ............. 1.31.0
numpy ................... 2.4.6
python-dotenv ........... 1.2.4
pytest .................. 9.1.1
CrewAI telemetry ........ off
CrewAI tracing .......... off
.env file ............... found (ai-engineering-course/.env)
Provider ................ gemini
Model ................... gemini-3.8-flash   (CrewAI name: gemini/gemini-3.8-flash)
API key ................. GEMINI_API_KEY found (AIza...xxxx)

Asking gemini/gemini-3.8-flash one tiny question through crewai.LLM ...
Reply: <one short sentence written by Gemini>

Gemini ready. Next: python robots_check.py
```

No key yet? Then the check stops before any call and tells you how to get the free key:

```text
.env file ............... NOT FOUND - copy .env.example to .env in the repo root
Provider ................ gemini
Model ................... gemini-3.5-flash-lite   (CrewAI name: gemini/gemini-3.5-flash-lite)
API key ................. MISSING - put GEMINI_API_KEY=... in the .env file

2 thing(s) to fix, see above.

Get a FREE Gemini key (no credit card needed):
  1. Open https://aistudio.google.com/apikey
  2. Sign in with your Google account and accept the terms
  3. Click 'Create API key' and copy it
  4. Put it in your .env file like this:  GEMINI_API_KEY=paste-your-key-here
```

`python check_setup.py --offline` runs the same checks without sending the question.

**Privacy.** Every program here that uses CrewAI starts with `import privacy`. It sets four switches before CrewAI loads, unless your `.env` says otherwise: `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` (no anonymous usage data), `CREWAI_TRACING_ENABLED=false` (no execution traces, and no question about them on your first run) and `CREWAI_DISABLE_VERSION_CHECK=true` (no "new version available" box: we pin versions on purpose). CrewAI also keeps the task outputs of your latest run in a small file on your own computer, `latest_kickoff_task_outputs.db`, in a folder named after the folder you run from (Mac: `~/Library/Application Support/ep13-shelf-sifter-crew/`, Windows: `C:\Users\<you>\AppData\Local\CrewAI\ep13-shelf-sifter-crew\`). Delete it whenever you like.

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` (also run `pip install "crewai[anthropic]==1.15.28"`) or `ollama`. `get_crewai_llm()` turns it into CrewAI's `"provider/model"` name.

> [!WARNING]
> **Common errors**
> - `crewai .................. MISSING` - you are in the LangChain `.venv`. Activate `.venv-crewai` (Step 0.1).
> - `ModuleNotFoundError: No module named 'langchain_google_genai'` when you run another episode's code here - expected: this environment has CrewAI's own Gemini support, not LangChain's. Use `.venv` for Episodes 1-12.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute, or until the daily reset (midnight Pacific time). Your limits: https://aistudio.google.com/rate-limit
> - `404 ... not found` for the model - a typo in `LLM_MODEL`. Leave it empty for the default.

## Step 2 - Check robots.txt

**Permission before scraping.** robots.txt is a small text file at the root of a site. It says which paths robots may visit. Python's standard library reads it for us.

`robots_check.py` (the important lines):

```python
def load_robots(page_url: str, get=requests.get, timeout: int = 10):
    response = get(robots_url(page_url), headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if response.status_code == 200:
        return parse_robots(response.text), 200       # the site has rules: read them

    parser = RobotFileParser()
    if response.status_code in (401, 403) or response.status_code >= 500:
        parser.disallow_all = True                     # "no robots" or the site has trouble: do not scrape
    else:
        parser.allow_all = True                        # 404: no robots.txt, so no rules
    return parser, response.status_code
```

```bash
python robots_check.py
```

Expected output (checked 2026-10-26):

```text
Robots file:  https://books.toscrape.com/robots.txt
HTTP status:  404 (the site has no robots.txt, so it sets no rules)
User-Agent:   shelf-sifter-crew/1.0 (learning project; https://github.com/replyravi/ai-engineering-course)
Allowed: True   https://books.toscrape.com/catalogue/category/books/travel_2/index.html

Practice rules from fixtures/robots_sample.txt (written by hand, not the real site):
  Allowed: True   /catalogue/category/books/travel_2/index.html
  Allowed: False  /basket/
  Allowed: False  /accounts/login/
  Wait between requests (Crawl-delay): 2 seconds
```

The real shop sets no rules, so the second half shows what rules look like, from a file we wrote. robots.txt is only half of the permission: the site's purpose and terms are the other half (see "Scraping rules" above).

> [!WARNING]
> **Common errors**
> - `Could not download robots.txt: ...` - no internet, or a firewall. No answer means no permission: fix the connection first.
> - `HTTP status: 200` and `Allowed: False` - the site added rules since 2026-10-26. Respect them: the scraper will refuse those pages.

## Step 3 - Save one page as a fixture

**Offline tests need saved data.** While you build the parser you will run it many times. A saved copy (a **fixture**) lets you do that without sending the shop a single extra request.

`save_fixture.py` downloads the page with the polite fetcher from `scraper.py` (the important lines):

```python
class PoliteFetcher:
    ...
    def wait_your_turn(self) -> None:
        """Sleep until `delay` seconds have passed since our previous request."""
        if self.last_request_at is not None:
            waited = self.clock() - self.last_request_at
            if waited < self.delay:
                self.log(f"  waiting {self.delay - waited:.1f} s (polite delay)")
                self.sleep(self.delay - waited)
        self.last_request_at = self.clock()

    def fetch(self, url: str) -> str:
        """Return the HTML of `url`, politely."""
        if urlsplit(url).netloc not in ALLOWED_HOSTS:
            raise ValueError(f"this scraper only visits {', '.join(sorted(ALLOWED_HOSTS))}, not {url!r}")
        self.check_robots(url)
        self.wait_your_turn()
        response = self.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()                 # 404 or 500 -> an error, not an empty page
        ...
```

Only one site, robots.txt first, a pause, a clear User-Agent and a timeout: the rules from the top of this page, in code. `check_robots` downloads robots.txt once per site with `load_robots` from Step 2, and obeys a `Crawl-delay` longer than 2 seconds.

```bash
python save_fixture.py
```

Expected output (the wait time depends on your connection):

```text
Fetching https://books.toscrape.com/catalogue/category/books/travel_2/index.html
  robots.txt: HTTP 404
  robots.txt: allowed
  waiting 1.2 s (polite delay)
  GET https://books.toscrape.com/catalogue/category/books/travel_2/index.html -> HTTP 200 (37,009 bytes)
Saved fixtures/travel.html (36,992 characters, 11 books on the page)
```

Two requests, 2 seconds apart: robots.txt, then the page. The polite fetcher waited 1.2 seconds because the first request had already taken 0.8 seconds. The saved file stays on your computer (`.gitignore` keeps it out of git): it is the shop's page, not ours.

> [!WARNING]
> **Common errors**
> - `Saved ... 0 books on the page` - the shop changed its layout. Open the page in a browser and compare it with `fixtures/handmade_shelf.html`.
> - Prices print as `Â£45.17` in your own experiments - the server sends `text/html` without a text encoding, so `requests` guesses ISO-8859-1. `scraper.py` sets UTF-8, which the page itself declares.

## Step 4 - Parse books with BeautifulSoup

**Cleaning HTML into rows.** One shelf page has one `<article class="product_pod">` per book. We keep five clean fields per book.

`scraper.py`:

```python
def parse_books(html: str, page_url: str) -> list[dict]:
    """Turn one shelf page into a list of book rows. Works offline."""
    soup = BeautifulSoup(html, "html.parser")
    books = []
    for card in soup.select("article.product_pod"):          # one card per book
        link = card.select_one("h3 a")
        stars = card.select_one("p.star-rating")["class"]     # ["star-rating", "Four"]
        price_text = card.select_one("p.price_color").get_text(strip=True)   # "£45.17"
        books.append({
            "title": link["title"].strip(),                   # the link text is cut short with "...", the title attribute is not
            "price": float(re.sub(r"[^0-9.]", "", price_text)),   # keep only digits and the dot
            "rating": next(RATING_WORDS[c] for c in stars if c in RATING_WORDS),
            "stock": card.select_one("p.availability").get_text(strip=True),   # "In stock"
            "url": urljoin(page_url, link["href"]),           # "../../../x/index.html" -> a full address
        })
    return books
```

```bash
python scraper.py
```

Expected output:

```text
Parsed 11 books from fixtures/travel.html. The first three:

{'title': "It's Only the Himalayas", 'price': 45.17, 'rating': 2, 'stock': 'In stock', 'url': 'https://books.toscrape.com/catalogue/its-only-the-himalayas_981/index.html'}
{'title': 'Full Moon over Noah’s Ark: An Odyssey to Mount Ararat and Beyond', 'price': 49.43, 'rating': 4, 'stock': 'In stock', 'url': 'https://books.toscrape.com/catalogue/full-moon-over-noahs-ark-an-odyssey-to-mount-ararat-and-beyond_811/index.html'}
{'title': 'See America: A Celebration of Our National Parks & Treasured Sites', 'price': 48.87, 'rating': 3, 'stock': 'In stock', 'url': 'https://books.toscrape.com/catalogue/see-america-a-celebration-of-our-national-parks-treasured-sites_732/index.html'}
```

Text became numbers (`"£45.17"` -> `45.17`, `class="star-rating Two"` -> `2`), short link texts became full titles, and relative links became full addresses.

> [!WARNING]
> **Common errors**
> - `No saved page yet. Run Step 3 first` - run `python save_fixture.py`.
> - `TypeError: 'NoneType' object is not subscriptable` - a selector found nothing (a typo in `"p.star-rating"`, or the page is not a shelf page).

## Step 5 - Test the parser offline

**Test tools before agents use them.** If the parser is wrong, every agent after it is wrong too, and the log will not tell you why. `tests/test_parse.py` runs `parse_books` on `fixtures/handmade_shelf.html`, our hand-made page with three invented books.

```python
def test_cleans_each_book_into_a_row():
    first = parse_books(handmade_page(), TRAVEL_URL)[0]

    assert first == {
        "title": SLOW_TRAIN,
        "price": 12.5,          # "£12.50" became a number
        "rating": 4,            # class="star-rating Four" became 4
        "stock": "In stock",    # spaces and line breaks removed
        "url": "https://books.toscrape.com/catalogue/the-slow-train-to-kanyakumari_901/index.html",
    }
```

```bash
python -m pytest tests/test_parse.py
```

Expected output (your time will differ):

```text
...                                                                      [100%]
3 passed in 0.15s
```

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'scraper'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test tried to download a page. Tests always use a saved page or a fake.

## Step 6 - Wrap fetch-and-parse as a tool

**Name, description, typed argument, delay.** The model never reads your Python code. It sees four things: the tool's **name**, its **description**, its **arguments** (names, types, descriptions) and what it **returns**. Write them like instructions for a new colleague.

`tools.py`:

```python
class ScrapeShelfInput(BaseModel):
    url: str = Field(
        description="The full address of ONE shelf page of https://books.toscrape.com, for example "
                    "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
    )


class ScrapeShelfTool(BaseTool):
    name: str = "scrape_shelf"
    description: str = (
        "Read one shelf (category page) of the practice bookshop books.toscrape.com and return its books "
        "as a JSON list. Each book has title, price (in pounds), rating (1 to 5 stars), stock and url. "
        "It checks robots.txt and waits 2 seconds between requests, so call it once per shelf."
    )
    args_schema: type[BaseModel] = ScrapeShelfInput
    shelf: Shelf                                    # where the scraped books are pinned
    fetch: Callable[[str], str] = fetch_politely    # tests pass a fake fetch that reads a saved page

    def _run(self, url: str):
        try:
            html = self.fetch(url)
        except Exception as error:   # wrong site, robots.txt says no, no internet, HTTP 404 ...
            # ToolFailure tells CrewAI "this call FAILED". A plain error string would look like a success.
            return ToolFailure(message=f"Could not read {url}: {error}")

        books = [Book(**row).model_dump() for row in parse_books(html, url)]   # Book checks every row
        if not books:
            return ToolFailure(message=f"No books found on {url}. Is it a shelf (category) page?")
        self.shelf.fill(books, url)
        self.shelf.successful_calls.append(self.name)
        return json.dumps(books, ensure_ascii=False)
```

Two details matter later. `ToolFailure` (part of `crewai.tools` in 1.15.28) marks a failed call as a failure, so the crew can stop instead of carrying on. The `Shelf` is a plain Python notice board shared by both tools: it remembers what was **really** scraped, so we can check the agents afterwards. CrewAI also has a `@tool` decorator for small tools; Step 8 uses it.

Run the tool by hand, with no agent and no model:

```bash
python tools.py scrape
```

Expected output (the wait time depends on your connection):

```text
Tool name:    scrape_shelf
Description:  Read one shelf (category page) of the practice bookshop books.toscrape.com and return its books as a JSON list. Each book has title, price (in pounds), rating (1 to 5 stars), stock and url. It checks robots.txt and waits 2 seconds between requests, so call it once per shelf.
Arguments:    {"url": {"description": "The full address of ONE shelf page of https://books.toscrape.com, for example https://books.toscrape.com/catalogue/category/books/travel_2/index.html", "title": "Url", "type": "string"}}

  robots.txt: HTTP 404
  robots.txt: allowed
  waiting 1.2 s (polite delay)
  GET https://books.toscrape.com/catalogue/category/books/travel_2/index.html -> HTTP 200 (37,009 bytes)

tool.run(url=...) returned 2,362 characters of JSON: a list of 11 books.
First book: {"title": "It's Only the Himalayas", "price": 45.17, "rating": 2, "stock": "In stock", "url": "https://books.toscrape.com/catalogue/its-only-the-himalayas_981/index.html"}
```

**Already have a LangChain tool?** CrewAI agents only accept CrewAI tools, so wrap it first:

```python
from crewai.tools.base_tool import Tool
from langchain_core.tools import tool as langchain_tool

@langchain_tool
def price_per_star(price: float, rating: int) -> float:
    """Pounds paid for each star of rating. Lower means better value."""
    return round(price / rating, 2)

crew_tool = Tool.from_langchain(price_per_star)    # now an Agent accepts it
```

Why a new little tool and not the `calculator` from Episode 11? That file also imports `langchain` and `numexpr`, which live in the LangChain `.venv`, not in `.venv-crewai` (`ModuleNotFoundError: No module named 'numexpr'`). The wrapping works the same for any LangChain tool.

```bash
python wrap_langchain_tool.py
```

Expected output (no key needed, nothing is sent):

```text
1. A LangChain tool, given to a CrewAI agent as it is:
   rejected: Input should be a valid dictionary or instance of BaseTool (field: tools.0, got a StructuredTool)

2. The same tool, wrapped with Tool.from_langchain:
   accepted: Tool named 'price_per_star'
   description: Pounds paid for each star of rating. Lower means better value.
   crew_tool.run(price=26.08, rating=5) -> 5.22
```

> **Seen an older tutorial?** It passed LangChain tools straight to `Agent(tools=[...])`. CrewAI 1.15 rejects that (above). Use `Tool.from_langchain(...)`. (`BaseTool.from_langchain` fails with "Can't instantiate abstract class".)

> [!WARNING]
> **Common errors**
> - `FAILED: Could not read ...` - read the end of the line. `Failed to establish a new connection`: no internet. `this scraper only visits books.toscrape.com`: by design, the tool only visits the practice shop.
> - `Unknown shelf 'fiction'` - `python tools.py scrape <shelf>` knows only the shelves in `scraper.CATEGORIES` (travel, mystery, poetry, ...).
> - `ValueError: URL ... resolves to private/reserved IP` from CrewAI's ready-made `ScrapeWebsiteTool` - crewai-tools blocks local and private addresses by default, a safety guard. Our own tool is stricter still: one site only.
> - `pydantic ... ValidationError ... rating` - a book row broke the `Book` shape (`models.py`). The page layout probably changed.

## Step 7 - A single-agent crew with the Scout

**Agent, Task, Crew, max_iter.** An agent is a job description (role, goal, backstory), a model and tools. A task is a work ticket with a clear `expected_output`. A crew is the team, and the **process** is how it works: `Process.sequential` runs the tasks one after the other, like a relay. (The other value, `Process.hierarchical`, adds a manager agent that hands out the tasks; CrewAI refuses it without `manager_llm=` or `manager_agent=`. We do not use it in this episode.)

`crew.py` (the Scout part, slightly shortened: in the file, `build_crew()` puts it together):

```python
scrape_tool = ScrapeShelfTool(
    shelf=shelf,
    result_as_answer=True,    # the tool's JSON IS the Scout's answer: no model retypes 11 prices
)

scout = Agent(
    role="Shelf Sifter",
    goal="Read the {category} shelf of the practice bookshop and report its books exactly as scraped",
    backstory="You are careful and polite. You only report what the scrape_shelf tool returned. "
              "You never invent a book, a price or a rating.",
    tools=[scrape_tool],
    llm=model(),              # Gemini by default, from get_crewai_llm(); never leave it out
    max_iter=3,               # at most 3 think-act-observe laps, then it must answer
    allow_delegation=False,   # the Scout does its own job (False is also CrewAI's default)
    verbose=verbose,          # print every step in the terminal
)

scout_task = Task(
    description="Call the scrape_shelf tool once, with this url: {shelf_url}",
    expected_output="The JSON list of books that scrape_shelf returned. "
                    "Each book has title, price, rating, stock and url.",
    agent=scout,
)

crew = Crew(agents=[scout], tasks=[scout_task], process=Process.sequential,
            max_rpm=10,                    # at most 10 model requests a minute (Step 12)
            tool_failure_policy="raise",   # a failed tool call stops the crew (Step 8)
            verbose=True)
result = crew.kickoff(inputs={"category": "travel", "shelf_url": "https://books.toscrape.com/...", ...})
```

`{category}` and `{shelf_url}` are filled in by `kickoff(inputs=...)`. CrewAI's default is `max_iter=25`; for a one-tool job, 3 laps is plenty.

```bash
python scout.py --stop-after scout
```

Expected shape (CrewAI draws boxes as wide as your terminal; the IDs and token numbers differ; we shortened the long JSON with `...`):

```text
Shelf Sifter crew: shelf='travel'  interest='a long journey by train'  agents: scout
Model: gemini/gemini-3.8-flash   Process: sequential   max_rpm: 10

╭───────────────────────── 🚀 Crew Execution Started ──────────────────────────╮
│  Crew Execution Started                                                      │
...
╭────────────────────────────── 🤖 Agent Started ──────────────────────────────╮
│  Agent: Shelf Sifter                                                         │
│  Task: Call the scrape_shelf tool once, with this url:                       │
│  https://books.toscrape.com/catalogue/category/books/travel_2/index.html     │
╰──────────────────────────────────────────────────────────────────────────────╯

  robots.txt: HTTP 404
  robots.txt: allowed
  waiting 1.2 s (polite delay)
  GET https://books.toscrape.com/catalogue/category/books/travel_2/index.html -> HTTP 200 (37,009 bytes)
╭─────────────────────── 🔧 Tool Execution Started (#1) ───────────────────────╮
│  Tool: scrape_shelf                                                          │
│  Args: {'url':                                                               │
│  'https://books.toscrape.com/catalogue/category/books/travel_2/index.html'}  │
╰──────────────────────────────────────────────────────────────────────────────╯

Tool scrape_shelf executed with result: [{"title": "It's Only the Himalayas", "price": 45.17, ...
╭────────────────────── ✅ Tool Execution Completed (#1) ──────────────────────╮
│  Tool Completed                                                              │
│  Tool: scrape_shelf                                                          │
│  Output: [{"title": "It's Only the Himalayas", "price": 45.17, "rating": 2,  │
...
╭─────────────────────────── ✅ Agent Final Answer ────────────────────────────╮
│  Agent: Shelf Sifter                                                         │
│  Final Answer:                                                               │
│  [{"title": "It's Only the Himalayas", "price": 45.17, "rating": 2,          │
...
======================================================================
The Scout's answer is the tool's JSON, untouched: a list of 11 books. The first one:
{"title": "It's Only the Himalayas", "price": 45.17, "rating": 2, "stock": "In stock", "url": "https://books.toscrape.com/catalogue/its-only-the-himalayas_981/index.html"}

Scorecard (who actually scored)
  scrape_shelf ............. 1 successful call(s), 11 books from https://books.toscrape.com/catalogue/category/books/travel_2/index.html
  find_similar_books ....... 0 successful call(s)
  tool failures ............ 0

Usage metrics (result.token_usage)
  requests 1   prompt tokens <n>   completion tokens <n>   reasoning tokens <n>   total <n>

All checks passed.
```

**Point at the loop in the log:** *decide* = the model's request (you see its result as **Tool Execution Started**: Gemini chose `scrape_shelf` and wrote the `url`), *act* = our polite fetcher's own lines, *observe* = **Tool Execution Completed**, *stop* = **Agent Final Answer**. (The fetcher's lines can print just before the Tool Execution Started box: CrewAI draws its boxes from a separate event handler, so the order on screen is not always the order of events.) One request to Gemini: because of `result_as_answer=True` the tool's JSON ends the task, so the model never has to retype it. (We checked this path with Google's reply replaced by a canned one; the books are real.)

> [!WARNING]
> **Common errors**
> - `No API key found for LLM_PROVIDER=gemini` - Step 1.
> - `The crew STOPPED because a tool call failed` - read the line under it: no internet, or the site said no. Step 8 shows how to read these.
> - `PROBLEM: the Scout's answer is NOT the tool's JSON` - the model answered without using the tool. Run it again, or try `LLM_MODEL=gemini-3.8-flash`.
> - Lines like `[Finalize] todos_count=0, todos_with_results=0` or `Warning: Event pairing mismatch` - CrewAI's own internal messages. Harmless.

## Step 8 - Break the tool on purpose

**Reading a validation error.** Most tutorials hide this part. We make the mistake on purpose: the broken tool has a **wrong type hint** (`url: int`, a web address is text) and a **lazy description** (`"Scrape."`). The `@tool` decorator turns the type hints and the docstring into what the model sees, so both mistakes go straight to the model.

`broken_tool.py`:

```python
@tool("scrape_shelf")
def broken_scrape_shelf(url: int) -> str:
    """Scrape."""
    return scrape(str(url))


@tool("scrape_shelf", result_as_answer=True)
def fixed_scrape_shelf(url: str) -> str:
    """Read one shelf (category page) of the practice bookshop books.toscrape.com and return its
    books as a JSON list (title, price, rating, stock, url). Pass the full page address as url, for
    example https://books.toscrape.com/catalogue/category/books/travel_2/index.html"""
    return scrape(url)
```

First see the error without a key: `--replay` runs the real CrewAI and the real tool, with a **scripted** model (from `testing/fakes.py`) that sends what any model would send here, the web address as text:

```bash
python broken_tool.py --replay
```

Expected output (the important parts; the IDs, the `#` box numbers and the `Iteration:` numbers change from run to run):

```text
REPLAY 1 of 2: the BROKEN tool (scripted model, no internet)

Tool 'scrape_shelf' (BROKEN) - what the model sees:
  description: Scrape.
  arguments:   {"url": {"title": "Url", "type": "integer"}}
...
╭───────────────────────────── 🔧 Tool Error (#1) ─────────────────────────────╮
│                                                                              │
│  Tool Failed                                                                 │
│  Tool: scrape_shelf                                                          │
│  Iteration: 1                                                                │
│  Attempt: 1                                                                  │
│  Error: Arguments validation failed: 1 validation error for Scrape_Shelf     │
│  url                                                                         │
│    Input should be a valid integer, unable to parse string as an integer     │
│  [type=int_parsing,                                                          │
│  input_value='https://books.toscrape.c...oks/travel_2/index.html',           │
│  input_type=str]                                                             │
...
The crew STOPPED because a tool call failed. The key lines of the error:
  Tool 'scrape_shelf' failed during 'Call the scrape_shelf tool once, with this url: https://books.toscrape.com/catalogue/category/books/travel_2/index.html': Arguments validation failed: 1 validation error for Scrape_Shelf
  url
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='https://books.toscrape.c...oks/travel_2/index.html', input_type=str]
  Expected arguments: {"url": {"title": "Url", "type": "integer"}}

How to read it: the tool asks for a NUMBER (url: int), but a web address is text.
Fix the type hint (url: str) and the description, then run:  python broken_tool.py --fixed

======================================================================
REPLAY 2 of 2: the FIXED tool, same scripted call (the page is fixtures/handmade_shelf.html)

Tool 'scrape_shelf' (FIXED) - what the model sees:
  description: Read one shelf (category page) of the practice bookshop books.toscrape.com and return its
    books as a JSON list (title, price, rating, stock, url). Pass the full page address as url, for
    example https://books.toscrape.com/catalogue/category/books/travel_2/index.html
  arguments:   {"url": {"title": "Url", "type": "string"}}
...
Clean run: one tool call, no errors. The Scout's answer is a JSON list of 3 books.
```

**How to read a red error, top to bottom:** which tool (`scrape_shelf`), which argument (`url`), what was expected (`a valid integer`), what came in (`input_value='https://...'`). In this replay CrewAI tries the same call 3 times (red **Tool Error** boxes with `Attempt: 1`, `2` and `3`; CrewAI draws its boxes from a separate thread, so their `#` numbers and their order can change from run to run) before it gives up, and then `tool_failure_policy="raise"` stops the crew. Without that setting, CrewAI's default is to note the failure and let the agent carry on, so a crew can finish with a confident answer that no tool ever produced (`tests/test_crew_fake.py` shows exactly that).

Now the live version with Gemini:

```bash
python broken_tool.py
python broken_tool.py --fixed
```

Expected: the first run stops with a red error and exit code 2. Which error depends on what Gemini sends: the address as text gives the `Input should be a valid integer` error above; a number (the schema asks for one) gives `this scraper only visits books.toscrape.com, not '2'`. With a real Gemini model CrewAI uses Gemini's own built-in tool calling (not the text replies our scripted model writes), and then it stops after the first error (we checked this path with canned replies). The fixed run ends with:

```text
Clean run: one tool call, no errors. The Scout's answer is a JSON list of 11 books.
```

> [!WARNING]
> **Common errors**
> - The broken run finishes without an error and prints `the answer is NOT the tool's JSON` - the model skipped the tool and answered on its own. That is the other face of the same bug: a vague tool is a tool the model avoids.
> - `Error executing listener ...` printed next to the red box (on the live run it starts with `ERROR:crewai.flow.runtime:`) - CrewAI logs the same failure twice. One failure, two messages.

## Step 9 - The similarity tool

**Local embeddings as an agent tool.** Same idea as Episodes 07 and 08: the all-MiniLM-L6-v2 model turns each title into 384 numbers, and titles with a similar meaning get similar numbers. It runs on your computer, for free. The CrewAI environment has no PyTorch, so we run the model's ONNX copy through Chroma, which CrewAI already installs. We compared both: same vectors as the sentence-transformers model of Episodes 07-08 (cosine 1.000000 for all 12 test texts).

`similarity.py`:

```python
def load_embedder():
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2   # imported here: it loads slowly
    return ONNXMiniLM_L6_V2()


def top_matches(query: str, books: list[dict], embed, k: int = 3) -> list[dict]:
    vectors = np.array(embed([query] + [book["title"] for book in books]), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)   # length 1, so a dot product = cosine
    scores = vectors[1:] @ vectors[0]
    best_first = np.argsort(-scores, kind="stable")[:k]
    ...
```

```bash
python similarity.py "a long journey by train"
```

The first run downloads the model once (a progress bar up to `79.3M`, 15 to 40 seconds on our connection, into `~/.cache/chroma/`, on Windows `C:\Users\<you>\.cache\chroma`; about 170 MB on disk once unpacked). Expected output:

```text
Query: 'a long journey by train'   (11 titles on the saved Travel shelf)

  0.464  The Great Railway Bazaar  (£30.54, rating 1/5)
  0.441  Vagabonding: An Uncommon Guide to the Art of Long-Term World Travel  (£36.94, rating 2/5)
  0.258  A Summer In Europe  (£44.34, rating 2/5)
```

"train" and "Railway" share no letters, yet that title wins: meaning, not keywords. Notice the catch too: the best match has 1 star. Matching what the reader likes and finding good value are two different jobs, which is why the Analyst gets both the list and this tool.

```bash
python similarity.py "mountain walks in the snow"
```

```text
Query: 'mountain walks in the snow'   (11 titles on the saved Travel shelf)

  0.337  It's Only the Himalayas  (£45.17, rating 2/5)
  0.187  Full Moon over Noah’s Ark: An Odyssey to Mount Ararat and Beyond  (£49.43, rating 4/5)
  0.138  The Road to Little Dribbling: Adventures of an American in Britain (Notes From a Small Island #2)  (£23.21, rating 1/5)
```

The same function as a CrewAI tool (`FindSimilarBooksTool` in `tools.py`), run by hand:

```bash
python tools.py similar "a long journey by train"
```

```text
Tool name:    find_similar_books
Description:  Find the 3 books on the shelf the Scout scraped whose titles are closest in meaning to the query. Returns a JSON list with title, score (higher means closer, 1.0 is the maximum), price, rating and url. Works only after scrape_shelf has run.
Arguments:    {"query": {"description": "What the reader likes, in plain words, for example 'a long journey by train'", "title": "Query", "type": "string"}}

tool.run(query='a long journey by train') returned:
[
  {
    "title": "The Great Railway Bazaar",
    "score": 0.464,
    "price": 30.54,
    "rating": 1,
    "url": "https://books.toscrape.com/catalogue/the-great-railway-bazaar_446/index.html"
  },
  {
    "title": "Vagabonding: An Uncommon Guide to the Art of Long-Term World Travel",
    "score": 0.441,
    "price": 36.94,
    "rating": 2,
    "url": "https://books.toscrape.com/catalogue/vagabonding-an-uncommon-guide-to-the-art-of-long-term-world-travel_552/index.html"
  },
  {
    "title": "A Summer In Europe",
    "score": 0.258,
    "price": 44.34,
    "rating": 2,
    "url": "https://books.toscrape.com/catalogue/a-summer-in-europe_458/index.html"
  }
]
```

> [!WARNING]
> **Common errors**
> - The first run hangs or fails without internet - the model downloads once (about 80 MB). Connect and run again; after that it works offline.
> - Scores around 0.3 to 0.5 "look low" - normal for this small model and short titles. Compare scores with each other, not with 1.0.
> - Why not `pip install sentence-transformers`? It would install here too (a dry run on 2026-10-24 changed no existing package), but it adds PyTorch and 12 more packages for the same vectors.

## Step 10 - The Analyst with typed output

**context and output_pydantic.** `context=[scout_task]` pastes the Scout's answer into the Analyst's prompt: that is the hand-off. `output_pydantic=Picks` makes CrewAI turn the Analyst's answer into a real `Picks` object, and Pydantic refuses anything that breaks the shape (a rating of 7, four picks, a price that is not a number).

`models.py`:

```python
class Pick(BaseModel):
    title: str = Field(description="the exact title from the Scout's list")
    price: float = Field(description="the exact price from the Scout's list")
    rating: int = Field(ge=1, le=5, description="the exact star rating from the Scout's list")
    url: str = Field(description="the exact url from the Scout's list")
    reason: str = Field(description="one short sentence: why this book is good value for this reader")


class Picks(BaseModel):
    category: str = Field(description="the shelf, e.g. travel")
    interest: str = Field(description="what the reader likes, copied from the task")
    picks: list[Pick] = Field(min_length=1, max_length=3, description="best value first")
```

`crew.py` (the Analyst's task):

```python
analyst_task = Task(
    description="The reader likes: {interest}.\n"
                "1. Call find_similar_books once with the reader's interest.\n"
                "2. From the Scout's list choose up to 3 best-value books for this reader. Best value "
                "means a high star rating for a low price: prefer 4 or 5 stars, and between equal "
                "stars the lower price wins. A book that find_similar_books ranked high wins a tie.\n"
                "Copy title, price, rating and url exactly from the Scout's list. Category: {category}.",
    expected_output="Up to 3 picks, best value first, each with title, price, rating, url and a "
                    "one-sentence reason, plus the category and the reader's interest.",
    agent=agents["analyst"],
    context=[scout_task],      # the Scout's answer is pasted into this task's prompt
    output_pydantic=Picks,     # CrewAI turns the answer into a Picks object (.pydantic)
)
```

```bash
python scout.py --stop-after analyst
```

Expected shape (Gemini chooses the books and writes the reasons):

```text
The Analyst's typed answer: Picks with 3 pick(s)
  1. <title>  £<price>  rating <stars>/5  - <reason>
  2. ...

Scorecard (who actually scored)
  scrape_shelf ............. 1 successful call(s), 11 books from https://books.toscrape.com/catalogue/category/books/travel_2/index.html
  find_similar_books ....... 1 successful call(s)
  tool failures ............ 0
  picks match the shelf .... 3 of 3

Usage metrics (result.token_usage)
  requests 3   prompt tokens <n>   completion tokens <n>   reasoning tokens <n>   total <n>
```

If Gemini follows the rule exactly, these are the top three on the Travel shelf (we sorted the real rows in Python: most stars first, then lowest price): **1,000 Places to See Before You Die** (5 stars, £26.08), **Full Moon over Noah's Ark** (4 stars, £49.43) and **A Year in Provence** (4 stars, £56.88). Compare with your run. `requests` is 3 (Scout 1, Analyst 2: the similarity call, then the answer), or 4 when CrewAI has to send one more request to turn a sentence into the `Picks` shape.

> **Seen an older tutorial?** It ran the search first and pasted the result into the agent's `goal` with an f-string, then gave the agent the same tool again. `context=[...]` keeps each task's input visible and testable, and the typed `Picks` object replaces "parse the answer text and hope".

> [!WARNING]
> **Common errors**
> - `PROBLEM: 'X': price is 49.4, the shelf says 49.43` - the Analyst changed a number. The scorecard caught it; the exit code is 3. Run again, or use `LLM_MODEL=gemini-3.8-flash`.
> - `PROBLEM: the Analyst's answer did not fit Picks` - the answer could not be turned into the shape (for example 4 picks). Read the Analyst's Final Answer box.

## Step 11 - The Writer produces the guide

**A task that writes a file.** The Writer has no tools. It receives the typed picks through `context=[analyst_task]`, and `output_file` makes CrewAI save its answer.

```python
writer_task = Task(
    description="Write a buying guide for the {category} shelf for a reader who likes {interest}.\n"
                "Start with the heading '# Best-value {category} books'. For each pick write a level-2 "
                "heading with the title as a Markdown link to its url, then one line with the price "
                "in pounds and the stars, then the reason. Use only the picks you received.\n"
                f"End with this line, exactly: {PRACTICE_NOTE}\n"
                "Answer with the Markdown only, without code fences.",
    expected_output="A Markdown guide: one heading, then one section per pick with a linked title, "
                    "price, stars and reason, then the practice-site note.",
    agent=agents["writer"],
    context=[analyst_task],    # the Writer receives the typed picks
    output_file=GUIDE_FILE,    # "reports/{category}_guide.md": CrewAI saves the Writer's answer here
)
```

```bash
python scout.py
```

Expected: the whole run from the top of this page, and a new file `reports/travel_guide.md` in this shape (Gemini writes the reasons):

```text
# Best-value travel books

## [<title>](https://books.toscrape.com/catalogue/<book>/index.html)
£<price>, <stars> stars. <reason>

## ...

_Prices and ratings come from books.toscrape.com, a practice site where they were assigned at random._
```

The scorecard line `pick links in the guide .. 3 of 3` checks that every pick's link really made it into the file.

> [!WARNING]
> **Common errors**
> - The guide's first line is three backticks and the word `markdown` - the model added a code fence anyway. The file is still fine to read; ask for no fences more firmly in the task text if it bothers you.
> - `pick links in the guide .. 2 of 3` - the Writer dropped or changed a link. That is why we check.

## Step 12 - The full sequential crew run

**Hand-offs and usage metrics.** Run the same crew on another shelf and another reader:

```bash
python scout.py --category poetry --interest "short poems about the sea"
```

Read the log from top to bottom like a cricket scorecard: who batted (each **Agent Started** box), who actually scored (each **Tool Execution Completed** box), and only then the result (the last **Agent Final Answer**). The Scout's JSON becomes the Analyst's input, and the Analyst's `Picks` becomes the Writer's input: each hand-off is a `context`.

Then the numbers at the end:

```text
Usage metrics (result.token_usage)
  requests <n>   prompt tokens <n>   completion tokens <n>   reasoning tokens <n>   total <n>
```

- `requests` counts calls to Gemini: **4** for one full run (Scout 1, Analyst 2, Writer 1), or 5 when CrewAI needs an extra call to shape the picks. A single chain step makes 1.
- `completion tokens` include Gemini's thinking tokens; `reasoning tokens` shows those on their own.
- `max_rpm=10` on the crew keeps it at 10 requests a minute or fewer, so a loop cannot burn your free quota.
- We give **each agent its own** `crewai.LLM` object. CrewAI adds up the usage of every agent's LLM object, so one object shared by three agents is counted three times: with canned replies we saw `requests 12` for 4 real requests.

(The request counts and the triple counting were measured on this machine with CrewAI's real Gemini class and Google's replies replaced by canned ones. Your token numbers come from your real run.)

`python scout.py --quiet` hides the step-by-step log and prints only the results, the scorecard and the usage.

> [!WARNING]
> **Common errors**
> - `429 RESOURCE_EXHAUSTED` in the middle of a run - free-tier limit. Wait a minute and run again; a full run is only about 4 requests.
> - `argument --category: invalid choice` - pick one of the shelves in `scraper.CATEGORIES` (travel, mystery, philosophy, music, science, poetry, humor, history, food-and-drink).

## Step 13 - The offline crew test

**FakeCrewAILLM scripts each step.** The test replaces Gemini with a scripted model from `testing/fakes.py`: `react_tool_use(name, args)` means "call this tool", `react_final_answer(text)` means "I am done". Everything else is real: CrewAI's loop, our tools on the hand-made page, the context hand-offs, the `Picks` object and the guide file. The scripted texts say `[scripted]`, so nobody mistakes them for AI output.

From `tests/test_crew_fake.py`:

```python
def scripted_full_run() -> list[str]:
    """The four model replies of one full run: Scout 1, Analyst 2, Writer 1."""
    return [
        react_tool_use("scrape_shelf", {"url": TRAVEL_URL}),                          # Scout
        react_tool_use("find_similar_books", {"query": "a long journey by train"}),   # Analyst, lap 1
        react_final_answer(PICKS_JSON),                                               # Analyst, lap 2
        react_final_answer(GUIDE),                                                    # Writer
    ]


def test_full_crew_runs_offline_with_a_scripted_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                            # the guide is written to reports/ here
    llm = FakeCrewAILLM(responses=scripted_full_run())
    shelf = Shelf()

    result = build_crew(llm=llm, shelf=shelf, fetch=handmade_fetch, embed=word_embed,
                        verbose=False).kickoff(inputs=INPUTS)

    picks = result.tasks_output[1].pydantic
    assert isinstance(picks, Picks)                        # typed, not text
    assert scout.check_picks(picks, shelf.books) == []     # every number matches the scrape
    ...
```

```bash
python -m pytest tests/test_crew_fake.py -vv
```

Expected output (the lines before `collecting` show your own paths; your time will differ):

```text
collecting ... collected 10 items

tests/test_crew_fake.py::test_one_agent_crew_makes_one_successful_tool_call PASSED [ 10%]
tests/test_crew_fake.py::test_full_crew_runs_offline_with_a_scripted_model PASSED [ 20%]
tests/test_crew_fake.py::test_context_hands_each_answer_to_the_next_task PASSED [ 30%]
tests/test_crew_fake.py::test_each_agent_gets_its_own_llm_object PASSED  [ 40%]
tests/test_crew_fake.py::test_crew_stops_loudly_when_the_scrape_fails PASSED [ 50%]
tests/test_crew_fake.py::test_without_raise_a_failed_tool_still_gives_a_confident_answer PASSED [ 60%]
tests/test_crew_fake.py::test_scout_run_prints_the_scorecard PASSED      [ 70%]
tests/test_crew_fake.py::test_scout_run_flags_a_made_up_price PASSED     [ 80%]
tests/test_crew_fake.py::test_scout_without_a_key_prints_the_free_key_help PASSED [ 90%]
tests/test_crew_fake.py::test_broken_tool_replay_shows_the_error_then_a_clean_run PASSED [100%]

============================== 10 passed in 1.62s ==============================
```

(Two `v`s, because `pytest.ini` already makes pytest quiet with `-q`: one `-v` only cancels it.) The crew test passed with no network: the tests block the internet on purpose. Now everything:

```bash
python -m pytest
```

```text
..........s..ssss.....................................                   [100%]
49 passed, 5 skipped in 2.33s
```

The 5 `s` are **skipped** tests, not failures: they check the LangChain model packages, which live in the other environment. If you see `6 skipped`, the local model of Step 9 is not downloaded yet: run `python similarity.py` once with internet.

> [!WARNING]
> **Common errors**
> - `Tests must not use the internet` - a test reached for a real model or page. Pass `llm=FakeCrewAILLM(...)` and a fake `fetch`.
> - `FakeCrewAILLM ran out of scripted responses` - CrewAI asked the model once more than you scripted (for example, you added a tool call). Add one more reply.

## Fix-It Corner

| You see | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'crewai'` | Wrong environment: activate `.venv-crewai` (Step 0.1). |
| CrewAI asks whether to share execution traces | Your `.env` sets `CREWAI_TRACING_ENABLED=true`, or you import crewai before `privacy`. Keep `import privacy` first. |
| `Error importing native provider: Missing credentials` or `OPENAI_API_KEY is required` although you use Gemini | An `Agent(...)` without `llm=` somewhere: CrewAI fell back to OpenAI. Pass `llm=get_crewai_llm()`. |
| `Attribute manager_llm or manager_agent is required` | You switched to `Process.hierarchical`. Give it a manager model, or go back to `Process.sequential`. |
| A crew that "worked" but the scorecard shows `0 successful call(s)` | The answer came from the model's imagination. Read the log for red Tool Error boxes. |
| `requests` in the usage line looks three times too high | Several agents share one `crewai.LLM` object. Create one per agent (`crew.py` does). |

## Final challenge

Add a **Fact-Checker** agent that compares every price and rating in the guide with the scraped JSON and flags mismatches, and make it fail loudly. Details, rules and a hint: [CHALLENGE.md](CHALLENGE.md).

## What you learned

- An agent is a loop: decide, call a tool, read the result, decide again, stop. A verbose log shows every lap, and `max_iter` caps them.
- A chain, an agent and a crew differ in who decides the next step. More agents cost more calls and give errors more places to spread, so use the simplest one that works.
- CrewAI is a standalone framework (not built on LangChain): `Agent`, `Task`, `Crew`, `Process.sequential`, `kickoff(inputs=...)`, and the output's `raw`, `pydantic`, `tasks_output` and `token_usage`. Always name the model.
- A tool is a name, a description, typed arguments and a return value. Test it alone first, return `ToolFailure` when it fails, and read red errors top to bottom.
- `context` hands one task's answer to the next, `output_pydantic` gives you a typed object, and a Python scorecard checks the agents' work against what the tools really returned.
- Polite scraping: a practice site, robots.txt, a pause between requests, a clear User-Agent, clean rows and saved pages for tests.

## Next episode

**Episode 14: CrewAI Projects: Email Drafts With Human Approval and an Education-Only Stock Research Desk** - two more crews in the same `.venv-crewai`: one writes emails that never leave without your click, one studies stocks for learning only. -> `ep14-crew-office` (coming soon)
