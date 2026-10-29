# Episode 10: Load Any File and Give Your Chatbot Memory: Document Loaders + LangChain Memory (2026)

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep10-counter-bot

## What you will build

A counter bot for **Kadak Kettle**, a small tea counter next to a bus stop that we invented for this course. It is built in two parts:

1. **Load any file.** `inspect_docs.py` reads the cafe's own files (a menu spreadsheet, an about-us web page and a two-page policy PDF) with LangChain loaders, prints what each loader made, and loads a whole folder through one function with a report.
2. **Give the bot a memory.** `counter_bot.py` is a terminal chat bot with three kinds of memory (`--memory buffer|window|summary`), a live token meter, one conversation per customer, menu answers looked up in the cafe's files with a citation, and chats that survive a restart.

<!-- not checked: written by hand -->
```text
[asha] You: Hi, I'm Asha. One masala chai, please. No nuts for me.
Bot: Hi Asha, one masala chai, no nuts, noted.

        (the bot runs with --save; you stop the program and start it again)

Welcome back, asha: 2 saved messages in this thread.

[asha] You: Who am I?
Bot: You are Asha, and you asked for nothing with nuts.
```

(The two `Bot:` lines are written by hand to show the idea; they are not a recorded Gemini reply, and Gemini will word its answers in its own way. The `Welcome back` line is printed by our program, so you will see exactly that line.)

**You will learn:**

- What a **Document** is (`page_content` + `metadata`) and what the CSV, HTML and PDF loaders put in each one.
- How to fix the usual loader problems: missing packages, the wrong text encoding and words glued together.
- How to load a whole folder with one function, and test it without a key.
- Why a chat model **forgets everything between calls**, and why "memory" means sending the chat again (with the token bill to prove it).
- Buffer, window and summary memory the 2026 way: `create_agent`, a **checkpointer**, a `thread_id`, `trim_messages` and `SummarizationMiddleware`.
- Where a fact belongs: the system prompt, retrieval, a tool, or the chat memory.

> **Seen an older tutorial?** It used the names on the left. In 2026 we use the right column. Same ideas, current and supported APIs.
>
> | Older tutorial | This episode (2026) |
> |---|---|
> | `from langchain.document_loaders import CSVLoader` | `from langchain_community.document_loaders import CSVLoader` (the old path no longer exists) |
> | `loader.load_and_split()` to "get the pages" | `loader.load()` (or `lazy_load()`), then the Episode 08 splitter |
> | `ConversationChain` + `ConversationBufferMemory` | `create_agent(..., checkpointer=InMemorySaver())` + a `thread_id` |
> | `ConversationBufferWindowMemory(k=2)` | `trim_messages(..., start_on="human")` in a `@before_model` middleware |
> | `ConversationSummaryMemory` | `SummarizationMiddleware(model, trigger, keep)` |
> | `RunnableWithMessageHistory` | also deprecated: LangGraph's checkpointer |
> | `memory.save_context(...)` to store fixed facts | the system prompt, retrieval or a tool (see "Where does a fact live?") |
>
> The old memory classes still run from the legacy `langchain-classic` package, but they are deprecated (`ConversationChain` since 0.2.7, the memory classes since 0.3.1).

## Prerequisites

- Finished Episode 06, folder `ep06-helpdesk-ladder` (messages, `create_agent`, tools) and Episode 08, folder `ep08-chunk-lab` (the splitter and the Chroma index). This folder runs on its own: the Episode 08 recipe is copied into `cafe_index.py`.
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- The shared `.venv` and root `.env` with your **free** Gemini key from Episode 1 ([SETUP.md](../SETUP.md)). Part 1 (Steps 1 to 8), the window demo, the token race and the index need no key; the chat steps (9 to 15) do.
- On the free tier, Google may use your prompts to improve its products. Chat with invented names only, never with real customer data.
- About 50 minutes. The only model this episode downloads is the local embeddings model (about 90 MB, Step 14); if you did Episode 07 or 08, it is already on your computer.
- That model, [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), is released under the **Apache-2.0** licence: free to use, also commercially (keep its licence notice if you share the model files).

## What is in this folder

<!-- not checked: folder tree -->
```text
ep10-counter-bot/
├── data/                     the cafe's own files (invented)
│   ├── menu.csv              12 items: item, price_rs, veg, contains_nuts, spice, description
│   ├── about.html            the about-us page
│   └── policy.pdf            2 pages: allergens, pickup, refunds, your chat (made with tools/make_policy_pdf.py)
├── kitchen/specials_today.csv   today's specials (live data, read by a tool in Step 14)
├── check_setup.py            Step 1: provider, model, key found?, library versions
├── loaders.py                Steps 3-7: the loaders and load_folder()
├── inspect_docs.py           Steps 2-8: print what each loader made
├── step09_two_calls.py       Step 9: two separate calls
├── step10_list_memory.py     Step 10: memory by hand, with the token meter
├── token_meter.py            the token meter (Steps 10-15)
├── memory_modes.py           Steps 11-13: buffer, window and summary memory
├── token_race.py             Step 13b: one scripted chat through all three memories (no key)
├── cafe_index.py             Step 14: split, embed and store the Documents (Episode 08's recipe)
├── cafe_tools.py             Step 14: the bot's two read-only tools
├── counter_bot.py            Steps 11-15: the chat bot
├── tools/make_policy_pdf.py  how policy.pdf was made (you do not need to run it)
├── llm.py, testing/          shared helpers (synced from _template)
├── tests/                    offline tests (fake chat model, no key, no internet)
├── requirements.txt          ../requirements-base.txt + ONE new package (Step 0.2)
├── CHALLENGE.md
└── index/  state/  outputs/  made when you run things (git-ignored)
```

Every file is already here. Type the code yourself as we go, or read along and run it.

## Step 0 - Get ready (about 3 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the repo root (`ai-engineering-course/`). You created them once in Episode 1 ([SETUP.md](../SETUP.md)). New here? Do SETUP.md first, then come back.

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd ep10-counter-bot
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep10-counter-bot
```

Expected: your prompt starts with `(.venv)` and ends in `ep10-counter-bot`.

> [!WARNING]
> **Common errors**
> - Windows: `The term '.\.venv\Scripts\Activate.ps1' is not recognized` / macOS, Linux: `.venv/bin/activate` with `no such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

### 0.2 Install this episode's one new package

This episode needs **one** package that is not in the shared list: `langgraph-checkpoint-sqlite`, which saves chats in a file (Step 15). This folder's `requirements.txt` installs the shared `../requirements-base.txt` plus that package (version 3.1.2, the newest on 2026-10-24, with the latest security fixes) and the two small packages it brings along:

```bash
pip install -r requirements.txt
pip check
```

Expected: the install ends with a line like `Successfully installed aiosqlite-0.22.1 langgraph-checkpoint-sqlite-3.1.2 sqlite-vec-0.1.9` (only on the first run; after that every line says `Requirement already satisfied`). The last command prints:

```text
No broken requirements found.
```

> [!WARNING]
> **Common errors**
> - `No matching distribution found for torch==2.14.1` - the `.venv` was made with Python 3.14/3.15. Delete the `.venv` folder and recreate it with Python 3.13 ([SETUP.md](../SETUP.md)).
> - `No matching distribution found for sqlite-vec==0.1.9` on a **Windows laptop with an ARM chip** (Snapdragon) - that small helper has no package for Windows on ARM, and our code does not use it, so install this episode's package without it: `pip install --no-deps langgraph-checkpoint-sqlite==3.1.2 aiosqlite==0.22.1`. Be warned: the shared list has the same gap for Chroma 1.5.9 (no Windows-on-ARM package either, checked on 2026-10-24), so `../requirements-base.txt` stops there too. Windows on ARM is not a tested setup for this course; a Windows x64 PC, a Mac with Apple Silicon or Linux works.

### 0.3 Check that your key is found from this folder

Your key lives in `ai-engineering-course/.env` (made in Episode 1). `llm.py` looks for it in this folder and then in each parent folder, so it finds the root file by itself. Do not copy `.env` into episode folders.

```bash
python llm.py
```

Expected output (your path will differ, and Gemini writes its own greeting):

<!-- not checked: needs a key -->
```text
Settings file: /.../ai-engineering-course/.env
Provider: gemini   Model: gemini-3.5-flash-lite
API key found. Asking the model a tiny question...
<a one-sentence hello written by Gemini>
```

With no key, `llm.py` prints its `Settings file` and `Provider` lines and then this friendly help, and every chat script in Steps 9 to 15 prints the same help and stops (real output; the last line names the `.env` file that was found):

```text
No API key found for LLM_PROVIDER=gemini.
Looked for: GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.

Get a FREE Gemini key (no credit card needed):
  1. Open https://aistudio.google.com/apikey
  2. Sign in with your Google account and accept the terms
  3. Click 'Create API key' and copy it
  4. Put it in your .env file like this:  GEMINI_API_KEY=paste-your-key-here

No .env file yet? In the ai-engineering-course folder (the repo root), copy .env.example to .env:
  macOS / Linux:       cp .env.example .env
  Windows PowerShell:  Copy-Item .env.example .env
(.env file in use right now: <the path of your .env file, or "none found">)
```

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama`. Nothing else changes.

> [!WARNING]
> **Common errors**
> - The `.env` file sits in the wrong folder (it belongs in `ai-engineering-course/`), is named `.env.txt` (Windows hides extensions), or still says `paste-your-key-here`. The last line of the help tells you which `.env` file was found.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute (per-minute limit) or until the daily reset. Your limits: https://aistudio.google.com/rate-limit

---

## Part 1 - Load any file

## Step 1 - Check the setup

One small script checks everything this episode needs, without calling the model: the provider and model from your `.env`, whether a key was found (it never prints the key), and the library versions.

```bash
python check_setup.py
```

Expected output while your `.env` has no key yet (real output; your Python patch number and your path will differ). With your key, the fourth line says `API key: found (Steps 9 to 15 can call the model)`:

```text
Python 3.13.15
Settings file: /.../ai-engineering-course/.env
Provider: gemini   Model: gemini-3.5-flash-lite
API key: NOT found. Steps 2 to 8 work without it; Steps 9 to 15 need it (see Step 0.3).

Libraries:
  langchain                    1.4.7     ok
  langgraph                    1.2.16    ok
  langchain-community          0.4.2     ok
  pypdf                        6.21.2    ok
  beautifulsoup4               4.15.0    ok
  lxml                         6.1.3     ok
  langchain-chroma             1.1.0     ok
  langgraph-checkpoint-sqlite  3.1.2     ok

Ready for Episode 10.
```

> [!WARNING]
> **Common errors**
> - `langgraph-checkpoint-sqlite  MISSING   <- install it` - you installed `../requirements-base.txt` instead of this folder's `requirements.txt`. Run Step 0.2.
> - `Settings file: no .env found` - the root `.env` from Episode 1 is missing. Make it as in [SETUP.md](../SETUP.md), step 6 (Steps 2 to 8 still work without it).
> - `ModuleNotFoundError: No module named 'dotenv'` - the `.venv` is not active in this terminal. Activate it again (0.1).

## Step 2 - Create the three cafe files

Real data lives in files of many shapes. Our cafe has three, each a different format, and you can make your own:

- `data/menu.csv` - a table. Type it in VS Code, or make it in Google Sheets (**File > Download > Comma-separated values**) or Excel (**Save As > CSV UTF-8**).
- `data/about.html` - a web page. Type it in VS Code.
- `data/policy.pdf` - a document with pages. Write it in Google Docs (**File > Download > PDF**) or LibreOffice (**File > Export as PDF**).

Ravi's copies are already in `data/`. Look at them:

```bash
python inspect_docs.py files
```

Expected output:

```text
Files in data/:
  about.html     1,271 bytes   a web page
  menu.csv       1,115 bytes   a table (rows and columns)
  policy.pdf    33,117 bytes   a document with pages
```

The first lines of the menu (open the file in VS Code to see all 12 items):

```text
item,price_rs,veg,contains_nuts,spice,description
Masala chai,20,yes,no,mild,"Strong tea boiled with milk, fresh ginger and green cardamom"
Cutting chai,12,yes,no,none,Half a glass of our everyday milky tea
Lemon ginger tea,25,yes,no,none,"Black tea with lemon, ginger and a spoon of honey, no milk"
```

> [!WARNING]
> **Common errors**
> - Your own CSV has a description with a comma in it - put that cell in double quotes, like the rows above. Spreadsheet apps do this for you.
> - Your own sizes differ - of course. Keep the same column names (`item`, `price_rs`, `veg`, `contains_nuts`, `spice`, `description`), because later steps use them.
> - On Windows, Ravi's files show `about.html  1,298 bytes` and `menu.csv  1,128 bytes` - git for Windows can save text files with Windows line endings (two characters at the end of each line instead of one). Python reads them the same way, so every later number in this README stays the same.

## Step 3 - Load the CSV: one Document per row

A **loader** reads one kind of file and gives back a list of **Documents**. Think of a sorting desk at a post office: parcels of every shape come in, and every item leaves in the same kind of labelled envelope. A Document has two parts:

- `page_content` - the text
- `metadata` - a small dictionary of labels about that text, like `{'source': 'data/menu.csv', 'row': 7}`

The metadata travels with the text all the way into the index, so an answer can cite it later (Step 14). `CSVLoader` makes **one Document per row**, written as one `column: value` line per column.

`loaders.py` (the important lines):

```python
from langchain_community.document_loaders import BSHTMLLoader, CSVLoader, PyPDFLoader, TextLoader


def load_menu(path=MENU_CSV):
    """Step 3: one Document per row. Its text is one 'column: value' line per column."""
    return CSVLoader(short_path(path), encoding="utf-8").load()
```

```bash
python inspect_docs.py csv
```

Expected output:

```text
langchain-community says: `langchain-community` is being sunset and is no longer actively maintained.
(It still works. We pin version 0.4.2. More in the README, Step 3.)

CSVLoader made 12 Documents from data/menu.csv (one per row)

Document 1:
  page_content:
    item: Masala chai
    price_rs: 20
    veg: yes
    contains_nuts: no
    spice: mild
    description: Strong tea boiled with milk, fresh ginger and green cardamom
  metadata: {'source': 'data/menu.csv', 'row': 0}

Document 8 (row 7: rows count from 0):
  page_content:
    item: Khasta kachori
    price_rs: 35
    veg: yes
    contains_nuts: no
    spice: medium
    description: Flaky fried pastry stuffed with spiced moong dal, served with green chutney
  metadata: {'source': 'data/menu.csv', 'row': 7}
```

Twelve rows, twelve Documents; the header line is not a Document. `row` counts from 0, so **row 7 is the 8th item** on the menu (the khasta kachori). Remember that row: it comes back as a citation in Step 14.

**About that first line.** The loaders live in a package called `langchain-community`. LangChain announced on 22 May 2026 that it is being sunset, and its GitHub repository was archived in June 2026. Every import prints that warning (our `loaders.py` catches it and shows it to you once). The loaders still work, and there is no standalone LangChain package for CSV and HTML loaders yet, so we **pin the last version, 0.4.2**, and import the loaders in **one** file, `loaders.py`. If they move one day, only that file changes. (Episode 07 read its PDF with plain `pypdf` to avoid the warning. This episode is the loaders episode, so we use them on purpose: one way of working for every format, and they fill in the metadata for us.)

> **Classic vs 2026.** Older tutorials import `from langchain.document_loaders import CSVLoader`. That path is gone in LangChain 1.x (real output):
>
> ```text
> ModuleNotFoundError: No module named 'langchain.document_loaders'
> ```

> [!WARNING]
> **Common errors**
> - A CSV saved from Excel as **CSV UTF-8** starts with an invisible marker (a BOM). Read with `encoding="utf-8"`, the first column is called `'\ufeffitem'` instead of `item`, and `source_column="item"` (Step 4) fails. Real output for the test file `tests/fixtures/tiny_menu_excel.csv`:
>
>   ```text
>   '\ufeffitem: Tea'
>   RuntimeError: Error loading tests/fixtures/tiny_menu_excel.csv
>     caused by ValueError: Source column 'item' not found in CSV file.
>   {'source': 'Tea', 'row': 0}
>   ```
>
>   The last line is the fix: `encoding="utf-8-sig"` reads the file with or without the marker.
> - Leave `encoding` out and Python uses your computer's default, which on Windows is often not UTF-8: the rupee sign `₹` then turns into junk letters. Always name the encoding.

## Step 4 - Name rows and add metadata

Two settings change what goes where:

- `source_column="item"` puts the item name in `metadata["source"]` **instead of** the file path.
- `metadata_columns=["veg"]` copies the `veg` column into the metadata **and takes it out of the text**.

```python
def load_menu_named(path=MENU_CSV):
    """Step 4: metadata["source"] becomes the item name, and veg is copied into the metadata.

    Careful: a metadata column is also taken OUT of the text.
    """
    return CSVLoader(short_path(path), encoding="utf-8", source_column="item", metadata_columns=["veg"]).load()
```

```bash
python inspect_docs.py csv-named
```

Expected output:

```text
CSVLoader(..., source_column="item", metadata_columns=["veg"])

Document 1:
  page_content:
    item: Masala chai
    price_rs: 20
    contains_nuts: no
    spice: mild
    description: Strong tea boiled with milk, fresh ginger and green cardamom
  metadata: {'source': 'Masala chai', 'row': 0, 'veg': 'yes'}

Notice: source is now the item name (the file path is gone), and veg moved OUT of the text.

CSVLoader(..., source_column="item", metadata_columns=["veg"], content_columns=[all six columns])

Document 1:
  page_content:
    item: Masala chai
    price_rs: 20
    veg: yes
    contains_nuts: no
    spice: mild
    description: Strong tea boiled with milk, fresh ginger and green cardamom
  metadata: {'source': 'Masala chai', 'row': 0, 'veg': 'yes'}

A filter on metadata: items with veg = no -> ['Egg bhurji pav']
```

The metadata now names the item and its veg flag, so a program can filter on it without reading the text. `content_columns` lists the text columns yourself, so `veg` can be in both places.

> [!WARNING]
> **Common errors**
> - Your citations stop saying `menu.csv` - that is what `source_column` does: it replaces the file path. Our folder loader (Step 7) adds its own `file` key, so citations keep the file name.
> - Never put `item` in `metadata_columns`: the dish name would leave the text, and a search for "kachori" could no longer find it.

## Step 5 - Load the HTML two ways

`BSHTMLLoader` uses BeautifulSoup to strip the tags and keep the visible text: **one Document per page**, with the page `<title>` copied into the metadata. Two settings matter:

- `open_encoding` - how to read the bytes of the file. Leave it out and Python uses your computer's default: UTF-8 on macOS and Linux, but usually `cp1252` on Windows.
- `get_text_separator` - what to put between the text of neighbouring tags. The default is `""`, nothing at all.

We load the page twice: first with the settings a typical Windows laptop gets (we name `cp1252`, so everyone sees the problem), then the fixed way.

```python
def load_about_like_windows(path=ABOUT_HTML):
    ...
    return BSHTMLLoader(short_path(path), open_encoding="cp1252").load()


def load_about(path=ABOUT_HTML):
    """Step 5, way 2: read the file as UTF-8 and put a space between the text of neighbouring tags."""
    return BSHTMLLoader(short_path(path), open_encoding="utf-8", get_text_separator=" ").load()
```

```bash
python inspect_docs.py html
```

Expected output (on Windows the first line says `cp1252`):

```text
Your computer's default text encoding: utf-8
(Python uses it when you give no encoding. On Windows it is usually cp1252.)

Way 1: open_encoding="cp1252" (Windows default), get_text_separator="" (the default)
  apostrophe:  bus stop. Itâ€™s a standing
  rupee sign:  less than â‚¹50.
  list items:  Ginger crushed fresh every morningMilk delivered twice a day
  menu bar:    HomeMenuAbout us
  metadata:    {'source': 'data/about.html', 'title': 'About Kadak Kettle'}
  characters:  926

Way 2: open_encoding="utf-8", get_text_separator=" "
  apostrophe:  bus stop. It’s a standing
  rupee sign:  less than ₹50.
  list items:  Ginger crushed fresh every morning Milk delivered twice a day
  menu bar:    Home Menu About us
  metadata:    {'source': 'data/about.html', 'title': 'About Kadak Kettle'}
  characters:  956

After tidy() (trim every line, drop empty lines): 916 characters. The first 4 lines:
    About Kadak Kettle
    Home Menu About us
    About Kadak Kettle
    Kadak Kettle is a small tea counter next to a busy city bus stop. It’s a standing counter with six stools, one big steel kettle and a glass case full of snacks.
```

Way 1 turns the curly apostrophe and the rupee sign into junk letters and glues the menu bar into `HomeMenuAbout us`. Way 2 fixes both. The `title` is there both times: the loader always copies it. The space separator leaves some empty lines behind, so `tidy()` trims them.

> [!WARNING]
> **Common errors**
> - ``ImportError: beautifulsoup4 package not found, please install it with `pip install beautifulsoup4` `` or ``ImportError: By default BSHTMLLoader uses the 'lxml' package. Please either install it with `pip install -U lxml` ...`` - the HTML loader needs both packages. They are in the course requirements, so seeing this means the `.venv` is not active (Step 0.1).
> - Junk letters like `â€™` in your own data - that is not a LangChain bug. The file was read with the wrong encoding. Pass `open_encoding="utf-8"`.

## Step 6 - Load the PDF: one Document per page

`PyPDFLoader` (it uses the `pypdf` package) makes **one Document per page**. Its page number starts at **0**; `page_label` is the number printed for people.

```python
def load_policy(path=POLICY_PDF):
    """Step 6: one Document per page. metadata["page"] counts from 0, metadata["page_label"] is the printed number."""
    return PyPDFLoader(short_path(path)).load()
```

```bash
python inspect_docs.py pdf
```

Expected output:

```text
PyPDFLoader made 2 Documents from data/policy.pdf (one per page)

  page=0  page_label='1'  total_pages=2  1,118 characters  starts: Kadak Kettle: Allergens, Pickup and Refunds
  page=1  page_label='2'  total_pages=2  886 characters  starts: 4. Pickup orders

All metadata keys: ['author', 'creationdate', 'creator', 'page', 'page_label', 'producer', 'source', 'title', 'total_pages']

Document 1 (page 0):
  page_content:
    Kadak Kettle: Allergens, Pickup and Refunds
    Version 1.0, October 2026. Kadak Kettle is an invented cafe used for teaching. Every
    rule on these two pages is made up.
    1. Allergens on our menu
    ... (15 more lines)
  metadata: {'producer': 'Matplotlib pdf backend v3.11.2', 'creator': 'Matplotlib v3.11.2, https://matplotlib.org', 'creationdate': '', 'title': 'Kadak Kettle: Allergens, Pickup and Refunds', 'author': 'Kadak Kettle (invented)', 'source': 'data/policy.pdf', 'total_pages': 2, 'page': 0, 'page_label': '1'}

load_and_split() gave 2 pieces: one per page here only because each page is
shorter than its hidden splitter's chunk_size (4,000 characters).
load_and_split(RecursiveCharacterTextSplitter(chunk_size=400)) gave 7 pieces.
```

Two pages, two Documents, and page 0 has `page_label` `'1'`. The loader also copies the PDF's own information (producer, author, title) into the metadata. Your own PDF from Google Docs will show different producer and creator names.

> **Classic vs 2026.** Older tutorials call `loader.load_and_split()` "get the pages". It is not: it loads the pages and then runs a text splitter with 4,000-character chunks, so it only matches the pages when every page is short. LangChain's own code calls it deprecated. Use `load()`, then the splitter you chose in Episode 08 (Step 14 does exactly that).

> [!WARNING]
> **Common errors**
> - ``ImportError: `pypdf` package not found, please install it with `pip install pypdf` `` - the `.venv` is not active (Step 0.1).
> - A scanned PDF gives empty text. A scan is a picture of a page, with no letters inside. Real output for a PDF that holds only an image: `1 Document, '' as text`. Reading pictures needs OCR, which is outside this episode.
> - Odd gaps inside words (like `T ake` in Episode 07's PDF) - PDF text is placed letter by letter and the reader guesses the spaces. Always print what a loader read before you trust it.

## Step 7 - Load the whole folder

A real cafe has more than three files. Think of a receptionist who checks every delivery against a list before stocking the shelves: `load_folder()` looks at each file's extension, picks the right loader, uses `lazy_load()` to read **one Document at a time**, and writes a report. A file with no loader is skipped and listed; a file that cannot be read is listed as failed, and the rest still load.

```python
# file extension -> (loader class, its settings)
LOADERS = {
    ".csv": (CSVLoader, {"encoding": "utf-8-sig"}),        # utf-8-sig also reads Excel's "CSV UTF-8" files
    ".html": (BSHTMLLoader, {"open_encoding": "utf-8", "get_text_separator": " "}),
    ".htm": (BSHTMLLoader, {"open_encoding": "utf-8", "get_text_separator": " "}),
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
}
```

and the heart of `lazy_load_folder()`:

```python
        loader = loader_for(path)
        try:
            for doc in loader.lazy_load():
                doc.metadata["file"] = path.name       # our own key: a short file name for citations
                if isinstance(loader, BSHTMLLoader):
                    doc.page_content = tidy(doc.page_content)
                if report is not None:
                    report.add(path.name, type(loader).__name__, doc)
                yield doc
        except Exception as error:                     # one broken file must not stop the whole folder
            if report is None:
                raise
            report.failed.append(f"{path.name} ({type(error).__name__})")
```

`yield` makes it a **generator**: it hands over one Document, waits until you ask for the next, then reads on. `load()` is simply `list(lazy_load())`, everything at once. For three files there is no difference; for ten thousand pages, lazy loading never needs them all in memory together.

```bash
python inspect_docs.py folder --export
```

Expected output:

```text
Loaders by file type: {'.csv': 'CSVLoader', '.html': 'BSHTMLLoader', '.htm': 'BSHTMLLoader', '.pdf': 'PyPDFLoader', '.txt': 'TextLoader', '.md': 'TextLoader'}

Folder: data/
file         loader        documents  characters
about.html   BSHTMLLoader          1         916
menu.csv     CSVLoader            12       1,709
policy.pdf   PyPDFLoader           2       2,004
total                             15       4,629
skipped (no loader for these): none
failed (the loader raised an error): none

Metadata keys on each file's Documents ('file' is our own key, the loader added the rest):
  about.html           file, source, title
  menu.csv             file, row, source
  policy.pdf           author, creationdate, creator, file, page, page_label, producer, source, title, total_pages

lazy_load_folder() is a generator. next() read one Document: about.html, 916 characters.
Saved 15 Documents to outputs/documents.jsonl (one JSON object per line).
```

`outputs/documents.jsonl` holds one Document per line, so you can open everything that was loaded in any editor. Line 2 (the first menu row):

```text
{"page_content": "item: Masala chai\nprice_rs: 20\nveg: yes\ncontains_nuts: no\nspice: mild\ndescription: Strong tea boiled with milk, fresh ginger and green cardamom", "metadata": {"source": "data/menu.csv", "row": 0, "file": "menu.csv"}}
```

> [!WARNING]
> **Common errors**
> - `No folder called ... Run this from the ep10-counter-bot folder.` - you are in another folder, or the `--folder` name has a typo.
> - Want another format, like `.json` or `.docx`? Add one line to `LOADERS` with a loader for it. Heavy all-in-one loaders (such as `unstructured`) pull in many packages; add them only when you need them.

## Step 8 - Run the loader tests

Tests read tiny files in `tests/fixtures/` whose answers you can check by eye: a 3-row CSV, the same CSV saved by Excel (with the marker), a one-line HTML page, a two-page PDF, a text note, a `.xyz` file with no loader and a `.pdf` that is not a PDF. No key and no internet are needed.

```bash
python -m pytest tests/test_loaders.py
```

Expected output (your time will differ):

```text
...............                                                          [100%]
15 passed in 0.25s
```

A test looks like this (from `tests/test_loaders.py`):

```python
def test_pdf_gives_one_document_per_page_counting_from_zero():
    docs = PyPDFLoader(TWO_PAGES).load()

    assert len(docs) == 2
    assert [doc.metadata["page"] for doc in docs] == [0, 1]
    assert [doc.metadata["page_label"] for doc in docs] == ["1", "2"]
```

Now point the folder loader at that messy test folder:

```bash
python inspect_docs.py folder --folder tests/fixtures
```

Expected output (the `invalid pdf header` and `EOF marker not found` lines are pypdf's own warnings about `broken.pdf`; they can appear in a slightly different place):

```text
Loaders by file type: {'.csv': 'CSVLoader', '.html': 'BSHTMLLoader', '.htm': 'BSHTMLLoader', '.pdf': 'PyPDFLoader', '.txt': 'TextLoader', '.md': 'TextLoader'}

invalid pdf header: b'This '
EOF marker not found
Folder: tests/fixtures/
file                loader        documents  characters
notes.txt           TextLoader            1          32
tiny_menu.csv       CSVLoader             3          99
tiny_menu_excel.csv CSVLoader             3          99
tiny_page.html      BSHTMLLoader          1          29
two_pages.pdf       PyPDFLoader           2          54
total                                    10         313
skipped (no loader for these): unknown.xyz
failed (the loader raised an error): broken.pdf (PdfStreamError)

Metadata keys on each file's Documents ('file' is our own key, the loader added the rest):
  notes.txt            file, source
  tiny_menu.csv        file, row, source
  tiny_menu_excel.csv  file, row, source
  tiny_page.html       file, source, title
  two_pages.pdf        author, creationdate, creator, file, page, page_label, producer, source, title, total_pages
invalid pdf header: b'This '
EOF marker not found

lazy_load_folder() is a generator. next() read one Document: notes.txt, 32 characters.
```

The Excel CSV loads like the plain one (thanks to `utf-8-sig`), the `.xyz` file is skipped, and the broken PDF is reported instead of stopping the run. pypdf's two warnings come **twice**: files are read in name order, so `broken.pdf` is the first file, and the `next()` test at the end tries it again before it reaches `notes.txt`.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'loaders'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test tried to connect somewhere. Our tests block the internet on purpose.

---

## Part 2 - Give the bot a memory

## Step 9 - Two bare calls: models are stateless

Phone a call centre where every agent is new: unless someone keeps notes, you repeat your whole story on every call. A chat model is like that. Each call is a brand-new request, and the model keeps **nothing** between calls.

`step09_two_calls.py` (the important lines):

```python
FIRST = "Hi, I'm Asha. Please remember my name."
SECOND = "What is my name?"


def ask_once(llm, text):
    """One call. The model receives exactly ONE message: this text. Nothing else."""
    return llm.invoke(text)
```

```bash
python step09_two_calls.py
```

Expected shape (Gemini writes its own words; the input-token numbers come from Gemini):

```text
Call 1  You:   Hi, I'm Asha. Please remember my name.
        Model: <a friendly hello from Gemini>
        (this call sent 1 message; input tokens: <N>)

Call 2  You:   What is my name?
        Model: <Gemini's reply: it has no way to know your name>
        (this call sent 1 message; input tokens: <N>)

Call 2 carried only its own message, so the model had no way to know your name.
```

"Memory" in a chat app means: **the app** keeps the conversation and sends it again on every call. The model itself remembers nothing. (Some providers can also keep a chat on their own servers for you, but you still pay for the earlier tokens on every turn.)

> [!WARNING]
> **Common errors**
> - You see the key help - finish Step 0.3 first.
> - The model "guesses" a name - it cannot know it. Run it again; a different guess proves the point.

## Step 10 - Memory by hand: a list and a token meter

The simplest memory is a Python list. We add every message to it and send the **whole list** on every call. A small token meter shows the bill: an offline estimate before sending (LangChain's `count_tokens_approximately`, about 4 characters per token), and Gemini's own count from `reply.usage_metadata` after it answers.

```python
def chat_with_list_memory(llm, lines, write=print):
    """Talk through `lines`. Returns the history list (our whole 'memory')."""
    history = [SystemMessage(CAFE_RULES)]         # the memory: one plain Python list
    for turn, line in enumerate(lines, start=1):
        history.append(HumanMessage(line))
        sent = len(history)
        estimate = estimate_tokens(history)        # counted before sending
        reply = llm.invoke(history)                # the WHOLE list goes to the model, every time
        history.append(reply)                      # remember the answer too, or the next call cannot see it
```

```bash
python step10_list_memory.py
```

Expected shape (4 calls; Gemini's words and all token numbers will differ, but the message counts are exactly these):

```text
Turn 1  You: Hi, I'm Asha.
        Bot: <Gemini's reply>
        meter: sent 2 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

Turn 2  You: One masala chai, please. No nuts for me, I'm allergic.
        Bot: <Gemini's reply>
        meter: sent 4 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

Turn 3  You: Add one khasta kachori too.
        Bot: <Gemini's reply>
        meter: sent 6 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

Turn 4  You: What is my name, and what have I ordered so far?
        Bot: <Gemini's reply: Asha, a masala chai and a khasta kachori>
        meter: sent 8 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

The list now holds 9 messages. Every new turn sends all of them again.
```

The input grows on every turn, because every earlier message is sent again. Our Gemini model can read about a million tokens, so a cafe chat will never fill it; the real price of re-sending is **money and waiting time**. If a request ever is too long, the API answers with an error. Starting a new chat is something chat apps do on screen, not the API.

> [!WARNING]
> **Common errors**
> - Gemini's count is higher than the estimate - normal. The estimate is a rough rule; Gemini counts its own tokens (and, later, the tool descriptions too).
> - `model counted: not reported` - some providers or local models do not report usage. The estimate still works.

## Step 11 - Buffer memory: a checkpointer and two threads

Doing that list by hand gets messy: one list per customer, saving it, loading it. LangGraph's **checkpointer** does it for us. We give `create_agent` (Episode 06) a checkpointer, and on every call we say **which conversation** this is with a `thread_id`. The checkpointer loads that thread's messages before the call and saves the new ones after it. One thread per customer keeps customers apart.

`memory_modes.py` builds all three memories with one function (the important lines):

```python
    return create_agent(
        model=llm,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=middleware,          # buffer: none. Window and summary add one (Steps 12 and 13)
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
```

and `counter_bot.py` passes the thread on every call:

```python
def config_for(customer, meter=None):
    """The thread_id tells the checkpointer WHICH conversation to load and save."""
    config = {"configurable": {"thread_id": thread_id_for(customer)}, "recursion_limit": MAX_AGENT_STEPS}
```

Run the bot and type the lines after `You:` (the `/customer` command switches to another customer):

```bash
python counter_bot.py
```

Expected shape (Gemini's words and the token numbers will differ; every other line is printed by our program exactly like this):

```text
Kadak Kettle counter bot | model: gemini-3.5-flash-lite (gemini) | memory: buffer | tools: off | saved in: RAM (gone when you exit)
Commands: /history  /customer NAME  /forget  /help  exit

[asha] You: Hi, I'm Asha. One masala chai, please. No nuts for me, I'm allergic.
Bot: <Gemini's reply>
  meter | model call: sent 2 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: /customer vikram
Now serving vikram (thread_id 'vikram', 0 saved messages).

[vikram] You: Who am I, and what did I order?
Bot: <Gemini's reply: it does not know yet>
  meter | model call: sent 2 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[vikram] You: /customer asha
Now serving asha (thread_id 'asha', 2 saved messages).

[asha] You: Who am I, and what did I order?
Bot: <Gemini's reply: Asha, one masala chai, no nuts>
  meter | model call: sent 4 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: /history
Saved for asha (thread_id 'asha'): 4 messages
   1. customer           Hi, I'm Asha. One masala chai, please. No nuts for me, I'm allergic.
   2. bot                <Gemini's first reply>
   3. customer           Who am I, and what did I order?
   4. bot                <Gemini's second reply>

[asha] You: exit
Bye!
```

Vikram's call sent only **2** messages (the system prompt and his question), so his thread never saw Asha. Asha's second call sent **4**. The system prompt is added fresh on every call and is never saved in the thread, so no memory setting can ever trim it away.

> **Classic vs 2026.** Older tutorials use `ConversationChain(llm=..., memory=ConversationBufferMemory())` and `.predict()`. Both are deprecated (the chain since 0.2.7, the memory class since 0.3.1) and `from langchain.memory import ...` no longer exists in LangChain 1.x. `RunnableWithMessageHistory`, which the old warning recommended, is deprecated too. Today: `create_agent` + a checkpointer + a `thread_id`.

> [!WARNING]
> **Common errors**
> - `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id` - you called a bot that has a checkpointer without a `thread_id`.
> - Everything is forgotten after `exit` - correct for now: `InMemorySaver` keeps chats in RAM only. Step 15 saves them to a file.

## Step 12 - Window memory: keep only the last two exchanges

A waiter who remembers only your last two orders. Before **every** model call, a small **middleware** cuts the saved conversation down to the new question plus the 2 exchanges before it. An exchange is one customer message and the bot's reply. `trim_messages` does the cutting; our "token counter" counts customer messages instead of tokens, so `max_tokens=k + 1` means "at most 3 customer messages": the 2 old ones plus the new question.

```python
def last_exchanges(messages, k=WINDOW_EXCHANGES, start_on="human"):
    """The new question plus the k exchanges before it.

    start_on="human" makes the window begin with a customer message. Without it, the window can
    begin with an old REPLY whose question was cut off (see python memory_modes.py --window-demo).
    """
    return trim_messages(messages, max_tokens=k + 1, token_counter=count_turns,
                         strategy="last", start_on=start_on)


@before_model
def keep_recent_turns(state: AgentState, runtime) -> dict | None:
    """Runs before every model call: forget everything older than the window."""
    messages = state["messages"]
    recent = last_exchanges(messages)
    if len(recent) == len(messages):
        return None                                 # nothing to forget yet
    # Replace the saved messages with the recent ones (the old ones are deleted, not just hidden).
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent]}
```

**Why `start_on="human"` matters.** See the cut without and with it (no key, no model):

```bash
python memory_modes.py --window-demo
```

Expected output:

```text
The whole chat (7 messages, the last one is the new question):
    human Hi, I'm Asha.
    ai    Hello Asha! What can I get you?
    human One masala chai, please.
    ai    One masala chai coming up.
    human Add a vada pav.
    ai    Done: one masala chai and one vada pav.
    human Who am I?

Window of 2 exchanges WITHOUT start_on (6 messages):
    ai    Hello Asha! What can I get you?
    human One masala chai, please.
    ai    One masala chai coming up.
    human Add a vada pav.
    ai    Done: one masala chai and one vada pav.
    human Who am I?
    -> it starts with a reply that uses the name Asha, but the message that gave the name is gone.

Window of 2 exchanges WITH start_on="human" (5 messages):
    human One masala chai, please.
    ai    One masala chai coming up.
    human Add a vada pav.
    ai    Done: one masala chai and one vada pav.
    human Who am I?
    -> it starts on a customer message. The name is simply gone, so the model can say it does not know.
```

Without the rule, the window keeps a reply that **uses** the name but drops the line that **gave** it, so the model reads a chat that contradicts itself. With the rule, the context is consistent. The name is gone either way: that is what a window is.

Now chat in window mode and type these five lines:

```bash
python counter_bot.py --memory window
```

Expected shape (Gemini's words and the token numbers will differ; the message counts and the `/history` lines are exactly these):

```text
[asha] You: Hi, I'm Asha.
Bot: <Gemini's reply>
  meter | model call: sent 2 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: One masala chai, please.
Bot: <Gemini's reply>
  meter | model call: sent 4 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: Add one vada pav.
Bot: <Gemini's reply>
  meter | model call: sent 6 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: Add one samosa plate too.
Bot: <Gemini's reply>
  meter | model call: sent 6 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: What was my first order, and what is my name?
Bot: <Gemini's reply: it cannot see the chai or your name any more>
  meter | model call: sent 6 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: /history
Saved for asha (thread_id 'asha'): 6 messages
   1. customer           Add one vada pav.
   2. bot                <Gemini's reply>
   3. customer           Add one samosa plate too.
   4. bot                <Gemini's reply>
   5. customer           What was my first order, and what is my name?
   6. bot                <Gemini's reply>
```

From turn 3 on, every call sends 6 messages (the system prompt + 2 exchanges + the new question): the meter flattens. By turn five the first order (turn 2) is gone. The rule: a fact from turn *n* leaves the window at turn *n* + 2 + 1. And because our middleware deletes the old messages, `/history` shows only the window.

To see which turns the model gets on every turn of a longer, scripted chat (no key), run `python token_race.py --show window`. The first lines:

```text
What the model sees on every turn in window memory (replies written by hand, played by a fake model):

turn  1  model call: system prompt + customer turns 1  (2 messages, about 117 tokens)
turn  2  model call: system prompt + customer turns 1, 2  (4 messages, about 151 tokens)
turn  3  model call: system prompt + customer turns 1, 2, 3  (6 messages, about 184 tokens)
turn  4  model call: system prompt + customer turns 2, 3, 4  (6 messages, about 178 tokens)
turn  5  model call: system prompt + customer turns 3, 4, 5  (6 messages, about 174 tokens)
```

> **Classic vs 2026.** The old `ConversationBufferWindowMemory(k=...)` is deprecated since 0.3.1. Its `k` counts **exchanges**, not messages: with `k=1` it returns the last 2 messages (real output from the legacy class: `k=1 returns 2 messages: [('human', 'Add a vada pav.'), ('ai', 'Added.')]`), while it still stores all 6. Today: `trim_messages` in a `@before_model` middleware.

> [!WARNING]
> **Common errors**
> - The bot says something like "as I said, Asha..." without the name being in the window - you removed `start_on="human"`. Put it back.
> - You want the full chat kept on disk but only the window sent to the model - then trimming must change only the model's input, not the saved state. That is a different middleware (`wrap_model_call`); try it after the challenge.

## Step 13 - Summary memory: one running note

A waiter who keeps a short note of the table. When the saved chat reaches 8 messages, `SummarizationMiddleware` asks a model to squeeze everything except the newest 4 messages into **one note**, and replaces them with it. The note costs **one extra model call** each time it squeezes, and a note can drop exact details.

```python
def summary_memory(llm, summary_llm=None):
    """Squeeze old messages into one summary message. Each squeeze is one EXTRA model call.

    summary_llm: the model that writes the note. None = the bot's own model.
    """
    return SummarizationMiddleware(
        model=llm,
        trigger=("messages", SUMMARY_TRIGGER_MESSAGES),     # when to squeeze (no trigger = never)
        keep=("messages", SUMMARY_KEEP_MESSAGES),           # how many recent messages stay word for word
        summary_prompt=SUMMARY_PROMPT,
        summarizer=summary_llm,
    )
```

Our `SUMMARY_PROMPT` tells the note-writer what a cafe must never lose: the name, every allergy, the order and the pickup time. (The default prompt is written for coding agents.) Chat in summary mode and type these lines:

```bash
python counter_bot.py --memory summary
```

Expected shape (Gemini writes the replies and the note; the counts and labels are exactly these):

```text
[asha] You: Hi, I'm Asha. I'm allergic to nuts.
Bot: <Gemini's reply>
  meter | model call: sent 2 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: One masala chai, please.
...
[asha] You: Add one vada pav.
...
[asha] You: I'll pick it up at 6:15 pm.
Bot: <Gemini's reply>
  meter | model call: sent 8 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: Do you take UPI?
Bot: <Gemini's reply>
  meter | summary call: sent 1 message, about <N> tokens (estimate) | model counted <N> in, <N> out
  meter | model call: sent 6 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: /history
Saved for asha (thread_id 'asha'): 6 messages
   1. summary            Here is a summary of the conversation to date: <the note Gemini wrote>
   2. bot                <Gemini's reply to "Add one vada pav.">
   3. customer           I'll pick it up at 6:15 pm.
   4. bot                <Gemini's reply>
   5. customer           Do you take UPI?
   6. bot                <Gemini's reply>

[asha] You: Can I add a peanut chikki?
Bot: <Gemini's reply: it should remember the nut allergy from the note>
  meter | model call: sent 8 messages, about <N> tokens (estimate) | model counted <N> in, <N> out
```

On turn 5 the saved chat reached 9 messages, so the middleware squeezed first (the `summary call`) and then the bot answered from the note plus the 4 newest messages. `/history` shows **one** summary message. It is saved with the human role, so we label it `summary` ourselves. Read the note Gemini wrote: is the allergy in it? Is the pickup time? Whether a detail survives depends on the model and the prompt, so always check.

> **Classic vs 2026.** The old `ConversationSummaryMemory` (deprecated since 0.3.1) wrote a new summary after **every** turn: one extra call per turn. Some tutorials pass it `max_token_limit=100`; that setting belongs to `ConversationSummaryBufferMemory`, and the plain summary memory silently ignores it. `SummarizationMiddleware` squeezes only when its `trigger` fires and keeps the newest messages word for word, so it is really the old "summary buffer": a note plus a recent window.

> [!WARNING]
> **Common errors**
> - The summary never happens - `trigger` has no useful default: with no trigger the middleware never squeezes. Ours is `("messages", 8)`.
> - `ValueError: keep thresholds must be greater than 0, got 0.` - keep at least one message.

## Step 13b - Race the three memories (no key)

How many tokens does each memory send? `token_race.py` plays one 20-turn chat through all three. No model is called: the replies and the summary notes were written by hand and are played back by the fake model from `testing/fakes.py`, so you get exactly these numbers. The agent, the middleware and the meter are the real ones, and the meter counts what each memory would send, with the offline estimate.

```bash
python token_race.py
```

Expected output:

```text
The same 20-turn chat in all three memories. Tokens sent to the model on each turn
(an offline estimate: count_tokens_approximately, about 4 characters per token).
The replies are written by hand and played back by a fake model; no AI is called.

turn  buffer  window  summary   extra summary call
   1     117     117      117
   2     151     151      151
   3     184     184      184
   4     215     178      215
   5     245     174      196   + 168
   6     278     177      229
   7     307     177      201   + 180
   8     339     176      233
   9     365     170      204   + 185
  10     394     172      233
  11     427     177      212   + 187
  12     457     175      242
  13     478     166      204   + 195
  14     510     169      236
  15     543     179      214   + 187
  16     570     174      241
  17     604     177      218   + 199
  18     629     175      243
  19     667     179      217   + 201
  20     699     186      249
 all   8,179   3,433    4,239   + 1,502

Model calls: 20 in every memory, plus 8 summary calls in summary memory.
Summary memory in all, with its summary calls: 5,741 tokens (buffer 8,179, window 3,433).
```

Buffer climbs like stairs. Window goes flat after turn 3. Summary saws up and down: it grows, squeezes, grows again. Count the summary calls too, or summary looks cheaper than it is. And remember what each one forgets: on turn 20 the scripted customer asks "Who am I, and what am I allergic to?", and the window memory no longer has turn 1 (see `python token_race.py --show window`).

| Memory | Tokens per turn | What it forgets | Extra calls | Good for |
|---|---|---|---|---|
| Buffer | grow every turn | nothing | none | short chats |
| Window | flat | everything older than the window | none | long chats where only the last few turns matter |
| Summary | small saw-tooth | exact details the note leaves out | one per squeeze | long chats with a few facts to keep |

> [!WARNING]
> **Common errors**
> - Your numbers differ - you changed the cafe rules in `common.py`, the script or a memory setting. That is the point: change one thing and race again.
> - These are estimates for **our** scripted replies. Gemini's replies are longer or shorter, and Gemini counts its own tokens.

## Step 14 - Ground menu answers: facts in retrieval, not memory

"Is the kachori nut-free?" should not be answered from memory or from the model's general knowledge: the answer lives in the cafe's files. So the bot gets two **read-only tools**:

- `search_cafe_docs` - **retrieval**: searches the menu, the about page and the policy.
- `todays_specials` - **live data**: reads `kitchen/specials_today.csv` at the moment it is asked.

First the index. This is Episode 08's recipe, copied into `cafe_index.py`: `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)`, the free local MiniLM model and a Chroma collection with cosine distance. New here: the chunks come from our three loaders, and their metadata becomes the **citation tag**.

```python
def citation(doc):
    """The tag a human can check: 'menu.csv row 7', 'policy.pdf page 1' or 'about.html'."""
    meta = doc.metadata
    if "row" in meta:                       # CSVLoader: row counts from 0, like the loader does
        return f"{meta['file']} row {meta['row']}"
    if "page_label" in meta:                # PyPDFLoader: the page number printed on the page
        return f"{meta['file']} page {meta['page_label']}"
    return meta["file"]
```

```bash
python cafe_index.py
```

Expected output (the first run on a new computer also downloads the embeddings model):

```text
Loaded 15 Documents from data/ (about.html 1, menu.csv 12, policy.pdf 2)
Split with RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60) -> 22 chunks
  about.html    1 Documents ->  3 chunks (longest 398 characters)
  menu.csv     12 Documents -> 12 chunks (longest 163 characters)
  policy.pdf    2 Documents ->  7 chunks (longest 395 characters)
Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 numbers per chunk)
Saved 22 chunks to index/ (Chroma collection 'cafe', cosine distance)
Example: id 'menu.csv row 7#1', metadata {'source': 'data/menu.csv', 'row': 7, 'file': 'menu.csv', 'start_index': 0, 'chunk': 1}
```

Every menu row is shorter than one chunk, so each row stays whole and keeps its `row` number. Search it the way the tool will (the tool asks for 4 chunks):

```bash
python cafe_index.py --search -k 4
```

Expected output:

```text
Question: Is the kachori nut-free?

1. distance 0.425  [menu.csv row 7]  item: Khasta kachori price_rs: 35 veg: yes contains_nuts: no spice: medium description:...
2. distance 0.436  [policy.pdf page 1]  (roasted peanuts) and peanut chikki (peanuts). - Milk: every chai except lemon ginger t...
3. distance 0.483  [menu.csv row 3]  item: Kesar badam doodh price_rs: 45 veg: yes contains_nuts: yes spice: none descriptio...
4. distance 0.499  [policy.pdf page 1]  2. Our kitchen is not nut-free Everything is cooked in one small kitchen. Peanuts, almo...
```

The kachori row comes first, and it says `contains_nuts: no`. But the 4th chunk matters just as much: **our kitchen is not nut-free**. That is why the tool returns 4 chunks, not 1. A correct answer says both: the recipe has no nuts, the kitchen cannot promise there are no traces, tell the staff. (Kadak Kettle's policy is invented for teaching. It is not food-safety advice.)

The search tool, from `cafe_tools.py`:

```python
@tool
def search_cafe_docs(question: str) -> str:
    """Search Kadak Kettle's menu, about page and policy.

    Use it for menu items, prices, ingredients, nuts and other allergens, pickup orders and refunds.
    Returns the best matching passages, each starting with its citation tag in square brackets.
    """
    try:
        return format_hits(search(_store(), question, k=TOOL_K))
    except Exception as error:      # return the problem as text: an exception would stop the whole agent
        return f"Error: could not search the cafe files ({error})."
```

Now chat with the tools on:

```bash
python counter_bot.py --tools
```

Expected shape (Gemini decides which tool to call and writes the answer; the `tool |` lines show what it called, and with tools each turn usually has two model calls: one that asks for the tool, one that answers):

```text
Kadak Kettle counter bot | model: gemini-3.5-flash-lite (gemini) | memory: buffer | tools: on | saved in: RAM (gone when you exit)
Commands: /history  /customer NAME  /forget  /help  exit

[asha] You: Is the kachori nut-free?
  tool  | search_cafe_docs({'question': '<the search words Gemini chose>'})
Bot: <Gemini's answer: no nuts in the recipe [menu.csv row 7], but the kitchen is not nut-free [policy.pdf page 1]>
  meter | model call 1: sent 2 messages, about <N> tokens (estimate) | model counted <N> in, <N> out
  meter | model call 2: sent 4 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: What are today's specials?
  tool  | todays_specials({})
Bot: <Gemini's answer: methi thepla with curd and a ginger lemon cooler, with prices>
  meter | model call 1: sent 6 messages, about <N> tokens (estimate) | model counted <N> in, <N> out
  meter | model call 2: sent 8 messages, about <N> tokens (estimate) | model counted <N> in, <N> out
```

`/history` now also shows the tool calls and the tool results the model read, for example `3. tool result        [menu.csv row 7] item: Khasta kachori price_rs: 35 ...`.

**Where does a fact live?** Four homes, like the notice board, the filing cabinet, the phone to the kitchen and the order pad at a real counter (our rule of thumb):

| Kind of fact | Example | Where it goes |
|---|---|---|
| Fixed rules that rarely change | opening hours, "never promise an allergy is safe" | the **system prompt** (`CAFE_RULES` in `common.py`) |
| Documents | the menu, the policy | **retrieval** (`search_cafe_docs`) |
| Live data | today's specials | a **tool** (`todays_specials`) |
| The conversation | "I'm Asha, I'm allergic to nuts" | the **chat memory** (the checkpointer) |

> **Classic vs 2026.** Older tutorials use `memory.save_context(...)` to store fixed facts such as a company name or a link. That adds a fake exchange to the chat, which a window or a summary later cuts away. Fixed facts belong in the system prompt, documents in retrieval and live data in tools.

> [!WARNING]
> **Common errors**
> - `The cafe index is not built yet. Build it first with:  python cafe_index.py` - run `python cafe_index.py` first.
> - The answer has no `[...]` tag - small models sometimes skip the citation rule. Ask again, or try `LLM_MODEL=gemini-3.8-flash` in `.env` (smarter, slower, uses more free quota).
> - Extra lines like `Loading weights: 100%`, `NOTE: Redirects are currently not supported in MacOs` or a Hugging Face `unauthenticated requests` warning - harmless. They come from the embeddings model.
> - You edited `data/` - run `python cafe_index.py` again. It empties the collection and rebuilds it, so there are never duplicates.

## Step 15 - Survive a restart: the SQLite checkpointer

`InMemorySaver` lives in RAM, so every chat is gone when the program stops. `SqliteSaver`, from the package we installed in Step 0.2, writes every thread to one file, `state/counter_bot.sqlite`. Same bot, same `thread_id`s, one different checkpointer.

```python
def open_checkpointer(save=False, path=STATE_FILE):
    """InMemorySaver keeps chats in RAM (gone when the program stops). SqliteSaver keeps them in a file."""
    if not save:
        return InMemorySaver()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)    # LangGraph may use it from another thread
    return SqliteSaver(connection)
```

A stored chat is **personal data**: keep it small and let people delete it. `/forget` deletes the thread, then rewrites the file, because SQLite only marks deleted rows as free space:

```python
def forget(checkpointer, thread_id):
    """Delete one customer's conversation. A chat is personal data: deleting it must really delete it."""
    checkpointer.delete_thread(thread_id)
    if isinstance(checkpointer, SqliteSaver):
        # SQLite only marks deleted rows as free space, so the old text can stay inside the file.
        # VACUUM rewrites the file without it, and the checkpoint empties the write-ahead log (-wal file).
        checkpointer.conn.execute("VACUUM")
        checkpointer.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
```

Chat, stop the program with `exit` (or close the terminal), start it again and ask "Who am I?". We leave the tools off here, so the message counts are easy to follow:

```bash
python counter_bot.py --save
```

Expected shape (Gemini's words will differ; the `saved in`, `Welcome back` and message-count lines are exactly these):

```text
Kadak Kettle counter bot | model: gemini-3.5-flash-lite (gemini) | memory: buffer | tools: off | saved in: state/counter_bot.sqlite
Commands: /history  /customer NAME  /forget  /help  exit

[asha] You: Hi, I'm Asha. One masala chai, please. No nuts for me.
Bot: <Gemini's reply>
  meter | model call: sent 2 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: exit
Bye!
```

Run the same command again:

```bash
python counter_bot.py --save
```

```text
Kadak Kettle counter bot | model: gemini-3.5-flash-lite (gemini) | memory: buffer | tools: off | saved in: state/counter_bot.sqlite
Commands: /history  /customer NAME  /forget  /help  exit
Welcome back, asha: 2 saved messages in this thread.

[asha] You: Who am I?
Bot: <Gemini's reply: you are Asha, and you asked for nothing with nuts>
  meter | model call: sent 4 messages, about <N> tokens (estimate) | model counted <N> in, <N> out

[asha] You: /forget
Deleted every saved message of asha.

[asha] You: /history
Saved for asha (thread_id 'asha'): 0 messages
  (nothing saved for this customer)
```

The new process found Asha's 2 messages in the file and sent them with her question (4 messages). The whole bot is `python counter_bot.py --tools --save`: it saves the same way, and its threads also hold the tool calls and tool results, so its counts are higher. The tests prove the same thing with a fake model and a real second Python process:

```bash
python -m pytest tests/test_threads.py
```

```text
.....                                                                    [100%]
5 passed in 0.38s
```

(Your time will differ.) One of these tests counts how often the name "Asha" appears in the SQLite files after `/forget`: zero.

`memory_modes.py` also switches on `LANGGRAPH_STRICT_MSGPACK=true` before LangGraph loads. With it, LangGraph only rebuilds known, safe kinds of objects when it reads a saved chat back from the file, as its security notes recommend.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'langgraph.checkpoint.sqlite'` - the new package is missing. Run Step 0.2.
> - SQLite is a file on one computer: fine for one counter and for learning. LangGraph's docs call it the local option for development; a busy app with many users uses a server database such as Postgres (`langgraph-checkpoint-postgres`).
> - Never commit `state/`: it holds people's chats. It is in this folder's `.gitignore`.

## Step 16 - Test everything without internet

Every part has tests with a **fake chat model**, fake replies you write yourself, so they run without a key, for free, and give the same result every time. They check what the model **sees** on every turn: `llm.received` keeps every list of messages the fake model was sent.

```bash
python -m pytest
```

Expected output:

```text
......................s....................................              [100%]
58 passed, 1 skipped in 5.78s
```

Your time will differ. An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes.

How a test checks the window (from `tests/test_memory.py`):

```python
def test_window_memory_forgets_the_first_order_by_turn_five():
    lines = ["Hi, I'm Asha.", "One masala chai, please.", "Add one vada pav.", "Add one samosa plate too.",
             "What was my first order, and what is my name?"]
    llm = fake_llm("Hi Asha!", "One masala chai.", "Added a vada pav.", "Added a samosa plate.", "I'm not sure.")
    bot = build_bot("window", llm=llm)
    for text in lines:
        say(bot, text)

    customer_lines = [[text for kind, text in seen(llm, turn) if kind == "human"] for turn in range(5)]
    assert "One masala chai, please." in customer_lines[3]            # turn 4: still in the window
    assert "One masala chai, please." not in customer_lines[4]        # turn 5: gone
```

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'counter_bot'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called a real model. Pass `llm=fake_llm(...)`.
> - More than one `s` (for example `52 passed, 7 skipped`) - run `python -m pytest -rs` to see why. "Local model not downloaded yet" means: run `python cafe_index.py` once with internet, then test again.

## Final challenge

Give the bot a **customer profile**: after each turn, extract the name, allergies and favourite item with structured output, keep them in a small pinned block that every call sees, and pair it with window memory. Then race four memories over a 20-turn chat. Details in [CHALLENGE.md](CHALLENGE.md).

## What you learned

- Every loader gives the same thing: a list of Documents, text plus metadata. CSV gives one per row, HTML one per page, PDF one per page counting from 0.
- Name the encoding, set the HTML separator, and print what a loader read before you trust it.
- One folder loader with a report handles many formats, and its metadata becomes the citation in the answer.
- The model is stateless. Memory is the app sending the chat again, and the token meter shows the cost.
- Buffer keeps everything, window keeps the last exchanges (start on a human message), summary keeps a note and costs extra calls. Pick by trade-off.
- One `thread_id` per customer, a SQLite checkpointer for restarts, and a `/forget` that really deletes.

## Next episode

**Episode 11: LangChain Chains the Modern Way** - pipes, parallel steps, and letting Python do the maths. -> `ep11-party-planner-chains`

---

Kadak Kettle, its menu, prices, rules and customers are invented for teaching. Any match with a real business or person is a coincidence.
