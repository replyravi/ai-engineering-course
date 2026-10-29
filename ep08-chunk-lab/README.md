# Episode 08: Vector Databases and Chunking, Tested: Which Way of Splitting Documents Finds Answers Best?

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep08-chunk-lab

## What you will build

`chunklab`, a small command-line lab. It cuts the **Dabbaloop handbook** from Episode 07 (an invented meal-box
delivery service in Pune) four different ways, shows you every chunk in colour, stores each way in its own
Chroma collection, and then **measures** which way finds the answers to ten test questions. It also filters a
search by file, races FAISS against Chroma, and answers with Gemini while citing the section the answer came from.

In Episode 07, the main refund rule (`refunds.md#4`, "Report a missed lunch or dinner within 6 hours...") came
only **8th** for the question "What is the refund window for a missed lunch?". Here is the scoreboard you will
build (real output, Step 11):

```text
   hit@3                                               10/10      10/10      8/10                10/10
   found first (hit@1)                                  9/10       9/10      7/10                10/10
   context per question (top 3, MiniLM tokens)           244        236       293                  305
   rank of the main refund rule (Episode 07)               8          8         1                    3

Winner: headers + recursive (10/10 at k=3, 10/10 found first). Tie rule: most hits at k=3, then most found first, then the least context.
```

The four columns are `character`, `recursive`, `tokens` and `headers + recursive`.

**You will learn:**

- Three reasons to chunk: one vector for a long text blurs many topics, long prompts cost more, and the embedder silently cuts long text (MiniLM reads 256 tokens).
- What a vector store row holds (id, vector, text, metadata), and how a search compares the question with the rows: exactly (every row) or approximately (HNSW hops).
- How to choose between FAISS, Chroma, a database extension such as pgvector and a managed cloud store.
- How the four LangChain splitters really cut: `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `TokenTextSplitter` and `MarkdownHeaderTextSplitter`, so you can predict their boundaries.
- How to measure chunk size in the embedder's own tokens with `from_huggingface_tokenizer`.
- How to split Markdown by headings, keep the heading path as metadata, and split long sections again.
- How to compare strategies with a hit@3 scoreboard instead of opinion, and how to filter a search by metadata.

> **Seen an older tutorial?** (checked 2026-10-23)
> - It imported `from langchain.text_splitter import ...`. In 2026 the splitters live in their own package, `langchain-text-splitters` (plural): `from langchain_text_splitters import RecursiveCharacterTextSplitter`.
> - It used `from langchain_community.vectorstores import Chroma` and then called `db.persist()`. In 2026 we use `from langchain_chroma import Chroma`: it saves to disk by itself as you add chunks, and there is no `persist()` any more.
> - It read the heading from a metadata key the splitter never writes (such as `header_type`), so every chunk looked as if it had no heading. `MarkdownHeaderTextSplitter` stores each heading under the name **you** pass in `headers_to_split_on` (Step 5).
> - It called `TokenTextSplitter(chunk_size=100)` "100 characters". It is 100 **tokens**, counted with OpenAI's tiktoken, and the default tokenizer is the old `gpt2` one, so we name `encoding_name="cl100k_base"` (Step 4).

## Prerequisites

- Finished Episode 07, folder `ep07-ask-the-handbook` (the RAG pipeline, the Chroma index and the grounded prompt). Episode 02 helps with tokens.
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- The shared `.venv` and root `.env` with your **free** Gemini key from Episode 1 ([SETUP.md](../SETUP.md)). Only Step 15 needs the key; Steps 1-14 run without it.
- About 50 minutes.
- Downloads, once: the local embeddings model [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (about 90 MB; already on your computer if you did Episode 01, 04, 06 or 07), released under the **Apache-2.0** licence: free to use, also commercially (keep its licence notice if you share the model files). And tiktoken's `cl100k_base` encoding file (1.7 MB, Step 4). The `tiktoken` library is MIT-licensed; the encoding file comes with no separate licence.
- On the free tier, Google may use your prompts to improve its products. Step 15 pastes handbook text into the prompt. Our handbook is invented, so it is safe.

## What is in this folder

```text
ep08-chunk-lab/
├── check_env.py              Step 1: splitters, stores and the embedder ready?
├── chunklab/                 the lab: python -m chunklab <command>
│   ├── strategies.py         the handbook, the four splitters, MiniLM token counting
│   ├── show.py               Steps 2-6: see where each splitter cuts
│   ├── measure.py            Step 7: chunk counts and sizes
│   ├── index.py              Steps 8-9: one Chroma collection per strategy, reopened
│   ├── score.py              Steps 10-12: the hit@3 scoreboard (and --fake, --sweep)
│   ├── search.py             Step 13: search with a metadata filter
│   ├── faiss_compare.py      Step 14: FAISS next to Chroma
│   ├── ask.py                Step 15: answer with Gemini, citing the section
│   └── __main__.py           the command list
├── data/
│   ├── handbook/             refunds.md delivery.md plans.md support.md food_safety.md
│   ├── pdf/food_safety.pdf   the original PDF of the food safety guide
│   └── questions.json        10 test questions, each with its section and answer phrase
├── tools/pdf_to_markdown.py  how food_safety.md was made from the PDF
├── llm.py, testing/          shared helpers (synced from _template)
├── tests/                    offline tests (fake models, no key, no internet)
├── requirements.txt          ../requirements-base.txt plus one pin (rich)
├── CHALLENGE.md
└── collections/, faiss_index/, tiktoken_cache/   made when you run the lab (git-ignored)
```

**Where the handbook comes from.** `refunds.md`, `delivery.md`, `plans.md` and `support.md` are copied unchanged
from Episode 07's `handbook/` folder, and `data/pdf/food_safety.pdf` is the same PDF (same MD5,
`7fc8adc063227512e7a25a83ea11c80b`). The header splitter only understands Markdown, so `food_safety.md` was made
from that PDF by `tools/pdf_to_markdown.py` (more in Step 5). The ten questions start with Episode 07's eight
golden questions; questions 9 and 10 are new.

Every file is already here. Type the code yourself as we go, or read along and run it.

## Step 0 - Get ready (about 2 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the repo root
(`ai-engineering-course/`). You created them once in Episode 1 ([SETUP.md](../SETUP.md)). New here? Do SETUP.md
first, then come back.

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd ep08-chunk-lab
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep08-chunk-lab
```

Expected: your prompt starts with `(.venv)` and ends in `ep08-chunk-lab`.

> [!WARNING]
> **Common errors**
> - `.venv\Scripts\Activate.ps1 cannot be loaded` / `No such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. Run `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again:
>   `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`
> - VS Code already shows `(.venv)` in new terminals once you pick the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.

### 0.2 Make sure the libraries are up to date

This episode uses the shared `requirements-base.txt` (the splitters, Chroma, FAISS, tiktoken and the local
embeddings model are all in it). `requirements.txt` in this folder adds one pin: `rich==15.0.0`, the library that
prints the coloured chunks. It is already installed with the base libraries (Chroma needs it), so normally
nothing new is downloaded:

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
> - `No matching distribution found for torch==2.14.1` - the `.venv` was made with Python 3.14/3.15. Delete the `.venv` folder and recreate it with Python 3.13 ([SETUP.md](../SETUP.md)).
> - Linux without an NVIDIA GPU: install the CPU-only PyTorch first (SETUP.md), otherwise pip downloads several GB of CUDA files.

### 0.3 Check that your key is found from this folder

Your key lives in `ai-engineering-course/.env` (made in Episode 1). `llm.py` looks for it in this folder and then in
each parent folder, so it finds the root file by itself. Do not copy `.env` into episode folders.

```bash
python llm.py
```

Expected output (your path will differ, and Gemini writes its own greeting):

```text
Settings file: /.../ai-engineering-course/.env
Provider: gemini   Model: gemini-3.5-flash-lite
API key found. Asking the model a tiny question...
<a one-sentence hello written by Gemini>
```

Without a key you get this help instead (real output), and Steps 1-14 still work:

```text
Settings file: no .env found
Provider: gemini   Model: gemini-3.5-flash-lite


No API key found for LLM_PROVIDER=gemini.
Looked for: GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.
...
```

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama` (free, local - install from https://ollama.com and run `ollama pull qwen3.5:4b`). Nothing else changes.

> [!WARNING]
> **Common errors**
> - `Settings file: no .env found` and `No API key found` - the root `.env` is missing or is named `.env.txt` (Windows hides extensions!). Create it from the repo root: `Copy-Item .env.example .env` (Windows) or `cp .env.example .env` (macOS/Linux), then paste your key.
> - `429 RESOURCE_EXHAUSTED` - you hit the free-tier limit. Wait a minute (per-minute limit) or until tomorrow (daily limit, resets at midnight Pacific time). See your limits at https://aistudio.google.com/rate-limit
> - `404 ... model not found` - the model name in `LLM_MODEL` is wrong or retired. Leave `LLM_MODEL=` empty to use the default.

---

## Step 1 - Check the environment

**Goal:** make sure the four splitters, Chroma, FAISS and the local embedder are ready.
**New concept:** the embedder has its own input limit. Text longer than that limit is cut off, and nobody tells you.

`check_env.py` (the important lines):

```python
model = SentenceTransformer(EMBEDDER, device="cpu")
dim = len(model.encode("Is the kitchen nut-free?"))
raw_limit = AutoTokenizer.from_pretrained(EMBEDDER).model_max_length
print(f"Embedder: {EMBEDDER} -> {dim} numbers per text")
print(f"  max length: {model.max_seq_length} tokens. Longer text is cut off, with no warning.")
```

```bash
python check_env.py
```

Expected output (recorded with Python 3.11; with the course's Python 3.13 the first line says `Python 3.13.x`, and every other line is the same):

```text
Python 3.11.17
Packages: langchain-text-splitters 1.1.3, langchain-chroma 1.1.0, chromadb 1.5.9, faiss-cpu 1.15.1, tiktoken 0.14.0, sentence-transformers 6.2.0, rich 15.0.0
Splitters: CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter imported
Handbook: 5 Markdown files in data/handbook/ (14,716 characters), 10 test questions
Embedder: sentence-transformers/all-MiniLM-L6-v2 -> 384 numbers per text
  max length: 256 tokens. Longer text is cut off, with no warning.
  (the tokenizer file on its own says 512; the embedder uses 256)
splitters OK, embedder OK (dim 384, max length 256)
```

The last two lines are the surprise of this episode: the tokenizer's own settings file says 512, but
sentence-transformers feeds MiniLM at most 256 tokens. If you trust the 512, half of a long chunk is silently thrown away.

> [!WARNING]
> **Common errors**
> - `can't open file '.../check_env.py': [Errno 2] No such file or directory` - you are not in the `ep08-chunk-lab` folder. `cd ep08-chunk-lab` first.
> - Extra lines such as `Loading weights: 100%`, `NOTE: Redirects are currently not supported in MacOs` or `Warning: You are sending unauthenticated requests to the HF Hub` - harmless. They come from loading the embeddings model; you will see them in most steps. They are left out of the expected outputs below.
> - `OSError: We couldn't connect to 'https://huggingface.co'` - the first run must download MiniLM. Connect to the internet and run it again.

## Step 2 - The character splitter

**Goal:** cut the handbook with `CharacterTextSplitter` and see where it cuts.
**New concept:** it cuts **only** at one separator (here a blank line, `"\n\n"`), then glues neighbouring pieces back
together while they fit in `chunk_size`. It never cuts inside a piece, so a long piece stays long, and that is what
its warnings mean.

`chunklab/strategies.py`:

```python
def character_splitter(size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cuts ONLY at the separator (a blank line), then glues pieces together up to chunk_size."""
    return CharacterTextSplitter(separator="\n\n", chunk_size=size, chunk_overlap=overlap, add_start_index=True)
```

**Predict first:** `refunds.md` has short paragraphs and `##` headings, each separated by a blank line. Where will the
400-character chunks end?

```bash
python -m chunklab show char
```

`-m` runs the `chunklab` folder as a package, so its files can import `llm.py` from this folder.
Expected output (the start, then the first 5 of the 12 `refunds.md` chunks; in your terminal they alternate in
colour, and the size warnings are printed by LangChain itself):

```text
Splitter: CharacterTextSplitter(separator='\n\n', chunk_size=400, chunk_overlap=60)
It cuts ONLY at blank lines, then glues neighbouring pieces together while they fit in 400 characters.
Created a chunk of size 613, which is longer than the specified 400
Created a chunk of size 456, which is longer than the specified 400
Created a chunk of size 445, which is longer than the specified 400
Whole handbook: 51 chunks (delivery.md 10, food_safety.md 10, plans.md 9, refunds.md 12, support.md 10)
3 chunks are longer than 400 characters: that is what each 'Created a chunk of size ...' warning means.
  food_safety.md#3 613 characters, starts with: - Every cook and packer wears a hair cap, gloves and an apro
  food_safety.md#8 456 characters, starts with: - Eat your meal within 3 hours of delivery. If you will eat 
  plans.md#2       445 characters, starts with: | Thali | What is inside | One meal | Monthly plan (22 meals
A piece with no blank line inside it cannot be cut by this splitter, so it stays whole.
Overlap: 24 of 46 neighbouring pairs share text (24 headings, 0 other short pieces, at most 36 characters).
chunk_overlap=60 can only repeat WHOLE pieces that fit in 60 characters, and most paragraphs are longer.

refunds.md, chunk by chunk (12 chunks, overlap with the previous chunk in amber between « »):
── refunds.md#1 · 230 characters ──
# Dabbaloop Refunds and Credits Policy

Version 3.2, effective 1 August 2026. Dabbaloop is an invented meal-box delivery service used for teaching; every name, price and rule in this handbook is made up.

## Who this policy covers
── refunds.md#2 · 290 characters ──
«## Who this policy covers»

This policy applies to every Dabbaloop customer in Pune, whether you pay for a weekly plan, a monthly plan, a quarterly plan or a single one-off thali. Corporate office plans follow the same rules unless the office contract says otherwise.

## Missed or late meal
── refunds.md#3 · 230 characters ──
«## Missed or late meal»

A meal counts as missed when it has not reached you 45 minutes after your delivery slot ends. For example, if your lunch slot is 12:45 pm to 1:30 pm and nothing has arrived by 2:15 pm, that lunch is missed.
── refunds.md#4 · 311 characters ──
Report a missed lunch or dinner within 6 hours of the end of your slot. Use Orders, then Report a problem, in the Dabbaloop app. We add a full credit for that meal to your Dabbaloop Wallet, usually within 2 hours of your report. Reports made after 6 hours are reviewed one by one and a credit is not guaranteed.
── refunds.md#5 · 213 characters ──
If the same address has three missed meals in one calendar month, the operations team calls you to fix the cause, for example a wrong gate number or a society that does not let riders in.

## Wrong or damaged meal
```

Read three things in it:

1. **The warnings.** The three oversized chunks are two bullet lists and a price table. Lists and tables have no blank
   lines inside, so this splitter has nowhere to cut them.
2. **The overlap.** Overlap means the end of one chunk is repeated at the start of the next, so a sentence cut at a
   boundary is not lost (Step 3 shows it properly). We asked for 60 characters of overlap, but the only thing ever
   repeated is a heading line. The splitter repeats whole pieces only, and a paragraph does not fit in 60 characters.
3. **`refunds.md#4`.** This is Episode 07's missing refund rule. Its heading, "Missed or late meal", sits in the
   chunks before it (`#2` and `#3`), so `#4` is a rule with no title.

> [!WARNING]
> **Common errors**
> - `Created a chunk of size 613, which is longer than the specified 400` - a warning, not an error. The splitter kept a piece whole because it had no separator inside it.
> - `ModuleNotFoundError: No module named 'chunklab'` after `python chunklab/show.py char` - run it as a package from the episode folder: `python -m chunklab show char`.
> - `No module named chunklab` after `python -m chunklab ...` - you are in the wrong folder (for example the repo root). `cd ep08-chunk-lab` first.
> - `No file called 'refund.md'. Try one of: ...` - the `--file` name must match a file in `data/handbook/` exactly.

## Step 3 - The recursive splitter and the overlap

**Goal:** cut with `RecursiveCharacterTextSplitter`, the usual default, and watch the overlap.
**New concept:** the separator ladder. It tries a blank line first, then a line break, then a space, then any
character, and it goes down a step **only** for a piece that is still too big. Then it glues pieces up to
`chunk_size` and repeats the end of one chunk at the start of the next.

```python
def recursive_splitter(size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Tries "\\n\\n", then "\\n", then " ", then "" - a smaller separator only for pieces still too big."""
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
```

```bash
python -m chunklab show recursive
```

Expected output (the start):

```text
Splitter: RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
Separators it tries, biggest first: "\n\n" (blank line), "\n" (line break), " " (space), "" (any character).
A smaller separator is used only for a piece that is still too big. Then pieces are glued up to 400 characters, and the end of one chunk is repeated at the start of the next (the overlap).
Whole handbook: 54 chunks (delivery.md 10, food_safety.md 12, plans.md 10, refunds.md 12, support.md 10), 94 to 398 characters
Overlap: 24 of 49 neighbouring pairs share text (24 of them share just a heading line), 561 characters stored twice.
```

No warnings now, and no chunk is over 400: the two lists and the table were cut at their line breaks
(`food_safety.md` went from 10 to 12 chunks, `plans.md` from 9 to 10). In `refunds.md` every paragraph is already
under 400 characters, so its 12 chunks are exactly the ones from Step 2, including the rule without a heading in
`refunds.md#4`. These are also Episode 07's settings (400 and 60).

To see the overlap carry real words, make the chunks small enough that the cuts land inside sentences:

```bash
python -m chunklab show recursive --size 150 --overlap 40
```

Expected output (part):

```text
Whole handbook: 165 chunks (delivery.md 35, food_safety.md 35, plans.md 30, refunds.md 35, support.md 30), 15 to 149 characters
Overlap: 48 of 160 neighbouring pairs share text (1 of them share just a heading line), 1,688 characters stored twice.
...
── refunds.md#10 · 149 characters ──
Report a missed lunch or dinner within 6 hours of the end of your slot. Use Orders, then Report a problem, in the Dabbaloop app. We add a full credit
── refunds.md#11 · 149 characters ──
«the Dabbaloop app. We add a full credit» for that meal to your Dabbaloop Wallet, usually within 2 hours of your report. Reports made after 6 hours are
── refunds.md#12 · 90 characters ──
«report. Reports made after 6 hours are» reviewed one by one and a credit is not guaranteed.
```

The amber text between « » is stored twice. That is the price of overlap: more storage and more tokens in every
prompt, paid so that a sentence cut at a boundary still appears whole in at least one chunk. Step 7 counts the cost.

**Where does the full stop go?** If you add `". "` to the separators, the setting `keep_separator` decides which
chunk gets it. Try this in `python` from the episode folder:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text = open('data/handbook/refunds.md', encoding='utf-8').read()
for keep in (True, 'end'):
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0,
                                              separators=['\n\n', '\n', '. ', ' '], keep_separator=keep)
    print(keep, splitter.split_text(text)[5:7])
```

Expected output:

```text
True ['thali', '. Corporate office plans follow the same rules unless the office contract says otherwise.']
end ['thali.', 'Corporate office plans follow the same rules unless the office contract says otherwise.']
```

With the default (`True`, which means "start"), the next chunk starts with a stray `". "`. With `"end"`, the full
stop stays with its sentence.

> [!WARNING]
> **Common errors**
> - `ValueError: Got a larger chunk overlap (60) than chunk size (50), should be smaller.` - the overlap must be smaller than the chunk size.
> - A chunk starts with `. ` - you added `". "` to the separators. Use `keep_separator="end"`.

## Step 4 - Measure in the embedder's tokens

**Goal:** stop counting characters and count what the models count: tokens.
**New concept:** characters, words and tokens are different rulers, and every tokenizer gives a different number.
`TokenTextSplitter` counts **tiktoken** tokens (OpenAI's tokenizer), which have nothing to do with MiniLM. For
retrieval, the ruler that matters is the **embedder's own tokenizer**.

### 4a - TokenTextSplitter, and two kinds of tokens

```python
def token_splitter():
    """Cuts every 100 tiktoken tokens, wherever that lands (even in the middle of a word)."""
    return TokenTextSplitter(encoding_name=TOKEN_ENCODING, chunk_size=TOKEN_CHUNK_SIZE,
                             chunk_overlap=TOKEN_OVERLAP, add_start_index=True)
```

The first run downloads tiktoken's `cl100k_base` file (1.7 MB). `strategies.py` keeps it in this folder, in
`tiktoken_cache/` (git-ignored), so the offline tests find it later:

```python
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(HERE / "tiktoken_cache"))
```

```bash
python -m chunklab show tokens
```

Expected output (the start, two chunks, and the end):

```text
Splitter: TokenTextSplitter(encoding_name='cl100k_base', chunk_size=100, chunk_overlap=15)
It counts tiktoken tokens (OpenAI's tokenizer). Without encoding_name it would use 'gpt2'.

One sentence, four rulers: "Report a missed lunch or dinner within 6 hours of the end of your slot."
  characters 71 · words 15 · cl100k_base tokens 17 · MiniLM tokens 16

Whole handbook: 44 chunks (delivery.md 9, food_safety.md 10, plans.md 9, refunds.md 8, support.md 8), each at most 100 cl100k_base tokens.
The same chunks in MiniLM tokens: 24 to 110. MiniLM reads 254 tokens of text (256 minus 2 markers), so no chunk is cut off.
...
── refunds.md#2 · 100 cl100k_base tokens · 95 MiniLM tokens ──
« quarterly plan or a single one-off thali. Corporate office plans follow the» same rules unless the office contract says otherwise.

## Missed or late meal

A meal counts as missed when it has not reached you 45 minutes after your delivery slot ends. For example, if your lunch slot is 12:45 pm to 1:30 pm and nothing has arrived by 2:15 pm, that lunch is missed.

Report a missed lunch or dinner within 6 hours of the end of
── refunds.md#3 · 100 cl100k_base tokens · 97 MiniLM tokens ──
«.

Report a missed lunch or dinner within 6 hours of the end of» your slot. Use Orders, then Report a problem, in the Dabbaloop app. We add a full credit for that meal to your Dabbaloop Wallet, usually within 2 hours of your report. Reports made after 6 hours are reviewed one by one and a credit is not guaranteed.

If the same address has three missed meals in one calendar month, the operations team calls you to fix the cause,
...
Without splitting (one vector per whole file):
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (769 > 512). Running this sequence through the model will result in indexing errors
  delivery.md      769 MiniLM tokens -> MiniLM reads the first 254 and ignores 515 (67%)
  food_safety.md   797 MiniLM tokens -> MiniLM reads the first 254 and ignores 543 (68%)
  plans.md         710 MiniLM tokens -> MiniLM reads the first 254 and ignores 456 (64%)
  refunds.md       687 MiniLM tokens -> MiniLM reads the first 254 and ignores 433 (63%)
  support.md       623 MiniLM tokens -> MiniLM reads the first 254 and ignores 369 (59%)
Proof with refunds.md: the vector of the whole file and the vector of only its first 254 tokens have cosine similarity 1.0000: the rest of the file changed nothing.
(The 'Token indices sequence length' warning above came from COUNTING whole files with the tokenizer, whose own file says 512. When MiniLM embeds a long text, it cuts it at 256 without a word.)
```

What to notice:

- The same sentence measures 71, 15, 17 or 16 depending on the ruler. A "100-token chunk" is roughly 400 characters of English, not 100.
- `TokenTextSplitter` cuts wherever token 100 lands, even in the middle of a sentence ("...within 6 hours of the end of" | "your slot").
- MiniLM reads 256 tokens, and two of them are its own start and end markers, so 254 are left for your text.
- Without chunking, MiniLM would ignore 59 to 68 percent of every file, and the proof shows it: the vector of the whole `refunds.md` is identical to the vector of its first 254 tokens. That is the third reason to chunk.

### 4b - The recursive splitter, measured in MiniLM tokens

Any LangChain splitter can count with a Hugging Face tokenizer instead of `len`:

```python
def section_splitter():
    """The recursive splitter, but chunk_size is counted in MiniLM tokens instead of characters."""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer(), chunk_size=SECTION_MAX_TOKENS, chunk_overlap=SECTION_OVERLAP,
    )
```

`tokenizer()` is `AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")`: counting tokens needs only the tokenizer files, not the model.

```bash
python -m chunklab show minilm
```

Expected output (the start):

```text
Splitter: RecursiveCharacterTextSplitter.from_huggingface_tokenizer(MiniLM tokenizer, chunk_size=200, chunk_overlap=30)
The same recursive cutting as Step 3, but chunk_size is now counted in MiniLM tokens, the units the embedder reads.
Whole handbook: 25 chunks (delivery.md 5, food_safety.md 6, plans.md 5, refunds.md 4, support.md 5), 27 to 198 MiniLM tokens: every chunk at or under 200, and MiniLM reads 254.
```

Every chunk is now guaranteed to fit the embedder. We use exactly this splitter in Step 6.

> [!WARNING]
> **Common errors**
> - A long `requests.exceptions.ConnectionError` mentioning `openaipublic.blob.core.windows.net` - tiktoken could not download `cl100k_base` on its first run. Connect to the internet and run `python -m chunklab show tokens` again; after that it works offline.
> - `Token indices sequence length is longer than the specified maximum sequence length` - a warning from **counting** a long text, as explained in the output. It is not an error.
> - `ValueError: Tokenizer received was not an instance of PreTrainedTokenizerBase` - you passed the `SentenceTransformer` model to `from_huggingface_tokenizer`. Pass its tokenizer (`AutoTokenizer.from_pretrained(...)`).

## Step 5 - The Markdown header splitter

**Goal:** cut at the headings and keep each heading as metadata.
**New concept:** `MarkdownHeaderTextSplitter` writes every heading into `doc.metadata`, under the key **you** choose.
That heading path is a breadcrumb, like "Home > Staff rules > Night shift" on a shopping site.

```python
HEADERS = [("#", "h1"), ("##", "h2")]   # (Markdown marker, the metadata key it becomes)

def header_splitter():
    """Cuts at # and ## headings and puts the heading text into metadata["h1"] and ["h2"]."""
    # strip_headers=False keeps the heading line in the chunk text too (the default, True, removes it).
    return MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
```

```bash
python -m chunklab show headers
```

Expected output (the start and the refund section):

```text
Splitter: MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'h1'), ('##', 'h2')], strip_headers=False)
It cuts at every # and ## line and writes the heading into metadata, under the key you chose.
Whole handbook: 42 sections (delivery.md 9, food_safety.md 9, plans.md 9, refunds.md 8, support.md 7). Metadata keys: ['h1', 'h2', 'source']
...
── refunds.md#3 · 169 MiniLM tokens · Dabbaloop Refunds and Credits Policy > Missed or late meal ──
metadata: {'source': 'refunds.md', 'h1': 'Dabbaloop Refunds and Credits Policy', 'h2': 'Missed or late meal'}
## Missed or late meal  
A meal counts as missed when it has not reached you 45 minutes after your delivery slot ends. For example, if your lunch slot is 12:45 pm to 1:30 pm and nothing has arrived by 2:15 pm, that lunch is missed.  
Report a missed lunch or dinner within 6 hours of the end of your slot. Use Orders, then Report a problem, in the Dabbaloop app. We add a full credit for that meal to your Dabbaloop Wallet, usually within 2 hours of your report. Reports made after 6 hours are reviewed one by one and a credit is not guaranteed.  
If the same address has three missed meals in one calendar month, the operations team calls you to fix the cause, for example a wrong gate number or a society that does not let riders in.
...
Sections longer than 200 MiniLM tokens (Step 6 splits them again): 1
  plans.md > The thalis: 235 MiniLM tokens
```

Now the refund rule sits with its heading, in one chunk that knows its breadcrumb. Three details you can see:

- The metadata keys are exactly the names we chose (`h1`, `h2`) plus `source`, which we add ourselves. Read them with `doc.metadata["h2"]`.
- `strip_headers=False` keeps `## Missed or late meal` in the text. With the default (`True`) the heading would be only in the metadata.
- The splitter joins the lines of a section with two spaces and a line break (a Markdown line break), and drops the blank lines.

**PDFs and Word files first become Markdown.** The header splitter only sees `#` headings. Our food safety guide was a
PDF, so `tools/pdf_to_markdown.py` rebuilt its headings with pypdf: the title becomes `#`, numbered lines like
"3. Allergens in our food" become `##`, and page footers are dropped. For your own PDFs and Office files, converter
tools such as Microsoft's MarkItDown or Docling do this job. They are not used or installed in this episode; check their
licences and supported formats before you use them.

```bash
python tools/pdf_to_markdown.py
```

Expected output (it writes the same `food_safety.md` again):

```text
Wrote data/handbook/food_safety.md: 3,225 characters, 9 headings
  # Dabbaloop Kitchen Food Safety and Allergen Guide
  ## 1. Our kitchen
  ## 2. Hygiene rules we follow every day
  ## 3. Allergens in our food
  ## 4. Allergy notes and red stickers
  ## 5. Labels on the lid
  ## 6. Storing and reheating at home
  ## 7. If something seems wrong
  ## 8. Kitchen visits
```

PDF text is never perfect: section 7 of the guide comes out as `T ake a photo`, exactly as Episode 07's loader read it.

> [!WARNING]
> **Common errors**
> - **Common mix-up:** every chunk seems to have no heading. You read a key that does not exist, such as `header_type`. Print `doc.metadata` once and use the keys you put in `headers_to_split_on`.
> - The headings are missing from the chunk text - that is the default `strip_headers=True`. Pass `strip_headers=False` to keep them.
> - Lines end with two spaces - that is how this splitter joins lines (a Markdown line break), not a bug in your file.

## Step 6 - Headers first, then a recursive split of long sections

**Goal:** keep the heading boundaries, but never let a section grow past what the embedder reads.
**New concept:** a two-stage split. Stage 1 cuts at headings. Stage 2 runs the token-measured recursive splitter
(Step 4b) on each section. A section that fits stays whole; a long one is split, and **every part keeps the
section's metadata**.

```python
def split_headers_then_recursive(docs):
    """Stage 2: sections longer than SECTION_MAX_TOKENS are split again. Metadata is copied to every part."""
    chunks = []
    splitter = section_splitter()
    for section in split_by_headers(docs):
        parts = splitter.split_documents([section])
        for number, part in enumerate(parts, start=1):
            if len(parts) > 1:
                part.metadata["part"] = f"{number} of {len(parts)}"
            chunks.append(part)
    return chunks
```

```bash
python -m chunklab show headers+recursive
```

Expected output:

```text
Stage 1: MarkdownHeaderTextSplitter -> 42 sections
Stage 2: RecursiveCharacterTextSplitter.from_huggingface_tokenizer(MiniLM, chunk_size=200, chunk_overlap=30) -> 43 chunks
Whole handbook: 43 chunks (delivery.md 9, food_safety.md 9, plans.md 10, refunds.md 8, support.md 7), 27 to 192 MiniLM tokens
Sections that were too long and got split: 1

plans.md > Dabbaloop Meal Plans and Prices > The thalis -> 2 parts
── plans.md#2 · 192 MiniLM tokens ──
metadata: {'source': 'plans.md', 'h1': 'Dabbaloop Meal Plans and Prices', 'h2': 'The thalis', 'part': '1 of 2'}
## The thalis  
Every thali is vegetarian and cooked fresh in our central kitchen each morning.  
| Thali | What is inside | One meal | Monthly plan (22 meals) |
|---|---|---|---|
| Ghar Ka Khana | 3 phulkas, 1 sabzi, dal, rice, salad | ₹120 | ₹2,490 |
| Jain Thali | No onion, garlic or root vegetables, 3 phulkas, sabzi, dal, rice | ₹130 | ₹2,690 |
| Protein Plus | Paneer or soya sabzi, sprouts, extra dal, less rice, 2 phulkas | ₹160 | ₹3,290 |
| Mini Meal | 2 phulkas, sabzi, small dal and rice, for kids and light eaters | ₹90 | ₹1,850 |
── plans.md#3 · 43 MiniLM tokens ──
metadata: {'source': 'plans.md', 'h1': 'Dabbaloop Meal Plans and Prices', 'h2': 'The thalis', 'part': '2 of 2'}
Add-ons for any thali: an extra phulka for ₹10, a bowl of curd for ₹20, buttermilk for ₹20 and the sweet of the day for ₹30.

Every part keeps its section's metadata, so a chunk from the middle of a section still knows its file and its headings.
```

Only one section of our handbook is long enough to be split: the price table. Notice that the whole table stayed in
part 1, with its header row. Add `--file plans.md` to print every chunk of a file.

> [!WARNING]
> **Common errors**
> - A part lost its breadcrumb - you split `doc.page_content` with `split_text`, which returns plain strings. Use `split_documents([...])`: it copies the metadata to every part.

## Step 7 - Compare the sizes

**Goal:** one table to compare all four strategies before we search anything.
**New concept:** chunk count, average and largest size, all in MiniLM tokens, and what overlap costs in storage.

```bash
python -m chunklab measure
```

Expected output:

```text
Chunk sizes in MiniLM tokens (MiniLM reads at most 254 tokens of text per chunk)

strategy               chunks   average   largest   over 254   stored tokens vs handbook
character                  51        73       172          0           3,733        +4%
recursive                  54        69       139          0           3,733        +4%
tokens                     44        95       110          0           4,179       +17%
headers + recursive        43        83       192          0           3,586        +0%
(no splitting)              5       717       797          5           3,586        +0%

The handbook itself: 5 files, 3,586 MiniLM tokens. 'vs handbook' = extra tokens stored because of overlap and cut words.

What overlap costs (recursive splitter, chunk_size=150 characters):

chunk_overlap          chunks   average   largest   over 254   stored tokens vs handbook
0 characters              159        23        47          0           3,586        +0%
20 characters             162        23        49          0           3,770        +5%
40 characters             165        24        50          0           4,000       +12%
60 characters             171        25        50          0           4,316       +20%
```

All four strategies fit the embedder (0 chunks over 254). The token splitter stores 17 percent extra, because its
15-token overlap repeats real text in every chunk. The second table is the overlap bill: with 60 characters of overlap
on small chunks, you store a fifth more tokens than the handbook has.

> [!WARNING]
> **Common errors**
> - Your numbers differ - you edited a handbook file or a setting in `strategies.py`. Re-run this step, then Step 8.

## Step 8 - Index every strategy in its own Chroma collection

**Goal:** embed every chunk once and save it, one collection per strategy, so the strategies can be compared fairly.
**New concept:** what a vector store row holds: an **id**, a **vector** (384 numbers), the **text**, and **metadata**.

`chunklab/index.py`:

```python
def open_collection(strategy, embeddings=None, folder=COLLECTIONS_DIR):
    """Open (or create) the Chroma collection of one strategy. Its name is the strategy name."""
    ...
    return Chroma(
        collection_name=strategy,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(folder),
        collection_configuration=SPACE,
    )


def build_collection(strategy, chunks, embeddings, folder=COLLECTIONS_DIR):
    """Empty the collection, then embed and add every chunk. The id of a chunk is its tag, e.g. refunds.md#4."""
    store = open_collection(strategy, embeddings, folder)
    store.reset_collection()                       # start empty: no old chunks, no duplicates
    store.add_documents(chunks, ids=[tag(chunk) for chunk in chunks])
    return store
```

`SPACE = {"hnsw": {"space": "cosine"}}` asks for cosine distance, as in Episode 07. In a lab we change settings
often, so `index` rebuilds every collection from scratch (Episode 07 showed how stable ids update an index without a rebuild).

```bash
python -m chunklab index
```

Expected output (your times will differ):

```text
Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 numbers per chunk)
Rebuilding 4 collections in collections/ (one per strategy) ...
  character             51 chunks   embedded and stored in 0.38 s
  recursive             54 chunks   embedded and stored in 0.08 s
  tokens                44 chunks   embedded and stored in 0.07 s
  headers + recursive   43 chunks   embedded and stored in 0.10 s
Saved 192 chunks. collections/ is 2.1 MB on disk.
Index type: HNSW (approximate nearest neighbours), distance: cosine, ef_search=100, max_neighbors=16

One row of the 'headers' collection:
  id:        refunds.md#3
  vector:    384 numbers, starting [-0.0002, 0.0566, -0.0092, 0.0249]
  text:      ## Missed or late meal   A meal counts as missed when it has not reach...
  metadata:  {'source': 'refunds.md', 'h1': 'Dabbaloop Refunds and Credits Policy', 'h2': 'Missed or late meal', 'chunk': 3}
```

The HNSW line comes from the settings Chroma saved for the collection. HNSW is a graph of neighbours: a search hops
from node to node towards the question instead of comparing it with every row. The first collection takes longest
because it also pays for starting Chroma.

> [!WARNING]
> **Common errors**
> - You changed `EMBEDDINGS_PROVIDER` or `EMBEDDINGS_MODEL` in `.env` - run `index` again. Vectors from different models cannot be mixed, and the token counts in this lab always use MiniLM's tokenizer.
> - `collections/` is in the episode folder and git-ignored. Delete it any time; `index` rebuilds it.

## Step 9 - Reopen in a new process

**Goal:** prove that the collections live on disk and come back without embedding anything again.
**New concept:** persistence. The vectors were computed once, in Step 8.

```python
def saved_counts(folder=COLLECTIONS_DIR):
    """{strategy: number of chunks} read straight from disk with the chromadb client (no embedding model)."""
    import chromadb
    ...
    client = chromadb.PersistentClient(path=str(folder))
    ...
    return {strategy: client.get_collection(strategy).count() for strategy in STRATEGIES}
```

```bash
python -m chunklab index --reopen
```

Expected output (your time will differ):

```text
Opening collections/ in this new process, with no embedding model loaded ...
  character             51 chunks
  recursive             54 chunks
  tokens                44 chunks
  headers + recursive   43 chunks
Opened and counted in 0.25 s. Embedding model loaded: no. Texts embedded: 0 (the vectors were read from disk).
```

LangChain's `Chroma` class is a thin layer over this `chromadb` client, so you can always open your data without LangChain.

> [!WARNING]
> **Common errors**
> - `No collections yet. Build them first with:  python -m chunklab index` - Step 8 was not run in this folder, or `collections/` was deleted.

## Step 10 - The question file

**Goal:** write down the right answers **before** looking at any score.
**New concept:** ground truth for retrieval. For each question we name the file(s) that answer it, the section where
the rule lives, and a short phrase that a retrieved chunk must contain.

`data/questions.json` (the first question):

```json
{
  "question": "What is the refund window for a missed lunch?",
  "sources": ["refunds.md", "support.md", "delivery.md"],
  "section": "Missed or late meal",
  "answer_contains": "6 hours"
}
```

Check that every answer phrase is really where we say it is:

```bash
python -m chunklab score --questions
```

Expected output (the start and the end):

```text
10 questions in data/questions.json. Each one names the section that answers it and a phrase a chunk must contain:

 1  What is the refund window for a missed lunch?
    answer in refunds.md > Missed or late meal   (also accepted: support.md, delivery.md)
    phrase "6 hours": found in that section
 2  Which areas are in delivery zone C?
    answer in delivery.md > Delivery zones
    phrase "Hinjewadi Phase 1": found in that section
...
10  Do my wallet credits expire?
    answer in refunds.md > Dabbaloop Wallet credits
    phrase "90 days": found in that section

10 of 10 answer phrases were found in their sections.
```

> [!WARNING]
> **Common errors**
> - `NOT FOUND - fix the file` - a typo in `section` or `answer_contains`, or the handbook text changed. A wrong answer key makes every score after it wrong.

## Step 11 - Score hit@3

**Goal:** stop arguing about splitters and measure them.
**New concept:** **hit@3**: a question is a hit when one of the 3 closest chunks comes from a right file **and**
contains the answer phrase. We also count **found first** (hit@1) and the **context** that the 3 chunks would put in a
prompt.

```python
def holds_answer(doc, question):
    """True when the chunk comes from a right file and contains the answer phrase."""
    return (doc.metadata["source"] in question["sources"]
            and question["answer_contains"].lower() in doc.page_content.lower())


def rank_of_answer(docs, question):
    """1 for the closest chunk, 2 for the next, ...: the first chunk that holds the answer (None if none)."""
    for rank, doc in enumerate(docs, start=1):
        if holds_answer(doc, question):
            return rank
    return None
```

```bash
python -m chunklab score
```

Expected output (in your terminal the hits are lime, the misses magenta and the winner bold):

```text
Scoring the 4 saved collections (character 51, recursive 54, tokens 44, headers 43 chunks) with 10 questions, k=3.

Rank of the first chunk that holds the answer (MISS = not in the top 3):
#  question                                        character  recursive    tokens  headers + recursive
1  What is the refund window for a missed lu...            1          1         1                    1
2  Which areas are in delivery zone C?                     1          1         1                    1
3  How much does the monthly Jain thali plan...            3          3    6 MISS                    1
4  How many days a month can I pause my plan?              1          1         1                    1
5  What are the chat support hours on Sunday?              1          1    5 MISS                    1
6  Is the kitchen nut-free?                                1          1         1                    1
7  How long can I keep the delivered food be...            1          1         2                    1
8  What is the deposit for the steel meal box?             1          1         1                    1
9  How much is the fee if I cancel my monthl...            1          1         1                    1
10 Do my wallet credits expire?                            1          1         1                    1

   hit@3                                               10/10      10/10      8/10                10/10
   found first (hit@1)                                  9/10       9/10      7/10                10/10
   context per question (top 3, MiniLM tokens)           244        236       293                  305
   rank of the main refund rule (Episode 07)               8          8         1                    3

Winner: headers + recursive (10/10 at k=3, 10/10 found first). Tie rule: most hits at k=3, then most found first, then the least context.
```

How to read it honestly:

- On this tidy handbook, three strategies find all ten answers in the top 3. Only the token splitter misses two. It cut the price table just after its header row, so the chunk with the Jain thali row has no column names (no "Monthly plan") and came only 6th. And it packed the Sunday chat hours into one chunk with the file's introduction, which blurred that chunk down to 5th place.
- **headers + recursive** is the only strategy that puts a right chunk **first** for every question, and every chunk it returns carries a breadcrumb. The price: about 305 tokens of context per question, against 236 for recursive.
- **Episode 07's miss:** the main refund rule moves from 8th place to 3rd once it sits with its heading. The token splitter even puts it 1st, but only because one of its cuts happened to start a chunk with that sentence. Question 1 was a hit for every strategy anyway, because another chunk with "6 hours" came first (for three strategies, the short summary in `delivery.md`). Look at the ranks, not only the hit count.

**Extra: the size slider.** Is 400 a good size? Try the recursive splitter at five sizes (in-memory collections, your
saved ones are not touched):

```bash
python -m chunklab score --sweep
```

Expected output:

```text
Recursive splitter at 5 sizes (overlap = 15 percent), in-memory collections (your saved collections are not touched):

chunk_size  chunks   hit@3  found first  context per question
       100     227    7/10         5/10             58 tokens
       200     119    6/10         5/10             98 tokens
       400      54   10/10         9/10            236 tokens   <- the size we use
       800      25   10/10         7/10            482 tokens
      1600      12    9/10         8/10            944 tokens
```

Tiny chunks lose the words that match the question; huge chunks blur many topics into one vector and send four
times the context. On this handbook, 400 characters is the sweet spot. On your documents, run the sweep again.

> [!WARNING]
> **Common errors**
> - `No collections yet. Build them first with:  python -m chunklab index` - run Step 8 first.
> - A hit rate measures **retrieval** only. It does not prove the model's answer is right (that is Step 15, and Episode 05's evals).

## Step 12 - Score offline, and run the tests

**Goal:** test the scoring code without the model, without the internet and without a key.
**New concept:** fake embeddings. `fake_embeddings()` from `testing/fakes.py` gives every text made-up numbers (the
same text always gets the same numbers). The scores become luck, but the **code** is fully exercised.

```bash
python -m chunklab score --fake
```

Expected output (the start and the end):

```text
FAKE embeddings: every text gets made-up numbers, so these scores are pure luck.
This run only proves the scoring code works: no embedding model, no internet, no API key.

Rank of the first chunk that holds the answer (MISS = not in the top 3):
#  question                                        character  recursive    tokens  headers + recursive
1  What is the refund window for a missed lu...       7 MISS     7 MISS   14 MISS              26 MISS
...
   hit@3                                                0/10       1/10      2/10                 0/10
...
No winner: fake vectors carry no meaning. Use the real model (Step 11) to compare.

Embedding model loaded: no.
```

It still reads MiniLM's tokenizer files and the tiktoken file from your disk (to cut the chunks), so run Steps 4 and
6 once online first. Now the tests:

```bash
python -m pytest
```

Expected output:

```text
..............s...........................                               [100%]
41 passed, 1 skipped in 5.97s
```

An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes. If
MiniLM or the tiktoken file was never downloaded, a few more tests skip and tell you which command to run once online.

How a scoring test works (from `tests/test_scoring.py`):

```python
def test_scoring_with_fake_embeddings_finds_an_exact_copy(breadcrumb_docs):
    # Fake embeddings give the SAME vector to the SAME text, so a question that copies a chunk ranks it 1st.
    question = {"question": breadcrumb_docs[1].page_content, "sources": ["plans.md"],
                "section": "Pausing your plan", "answer_contains": "10 days"}
    store = in_memory_store("copy", breadcrumb_docs, fake_embeddings())

    rows = score_store(store, [question], count=len)      # count characters: no tokenizer needed
```

The last test in that file runs the real MiniLM model and checks the Step 11 numbers, so if you change the
handbook or a setting, the tests tell you whether retrieval got better or worse.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'chunklab'` - run pytest from the episode folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called a real model or tried a download. Use the fakes from `testing/fakes.py`.
> - `SKIPPED ... tiktoken file not downloaded yet` or `MiniLM not downloaded yet` - run the command it names once with internet, then run the tests again.
> - `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized` and `Fatal Python error: Aborted` (seen on macOS) - FAISS and PyTorch each bring their own copy of the OpenMP library. `tests/lab_fixtures.py` and `chunklab/faiss_compare.py` set `KMP_DUPLICATE_LIB_OK=True`, the same setting scikit-learn makes when sentence-transformers imports it. Keep that line if you write your own FAISS code.

## Step 13 - Filtered search

**Goal:** search only inside some files.
**New concept:** a metadata filter narrows the rows **first**; only the chunks that pass it are compared with the question.

```python
def file_filter(files):
    """Chroma's filter for one file, or for several files with $in. None means no filter."""
    if not files:
        return None
    if len(files) == 1:
        return {"source": files[0]}
    return {"source": {"$in": files}}


def search(store, question, k=3, files=None):
    """The k closest chunks as (Document, distance), closest first, only from `files` if given."""
    return store.similarity_search_with_score(question, k=k, filter=file_filter(files))
```

First without a filter:

```bash
python -m chunklab search "What should I do if my food smells strange?"
```

```text
Question: What should I do if my food smells strange?
Collection: headers (43 chunks)   filter: None   chunks that pass the filter: 43

1. distance 0.324  [food_safety.md#8]  Dabbaloop Kitchen Food Safety and Allergen Guide > 7. If something seems wrong
   ## 7. If something seems wrong If the food smells sour, looks unusual or the seal on the lid is b...
2. distance 0.439  [support.md#4]  Dabbaloop Help, Complaints and Contact > Food safety concerns
   ## Food safety concerns If anything in your meal box looks, smells or tastes wrong, stop eating i...
3. distance 0.625  [food_safety.md#4]  Dabbaloop Kitchen Food Safety and Allergen Guide > 3. Allergens in our food
   ## 3. Allergens in our food Our menu labels show five allergens. Milk: paneer, curd, ghee and but...
```

Then only inside the food safety guide:

```bash
python -m chunklab search "What should I do if my food smells strange?" --file food_safety.md
```

```text
Question: What should I do if my food smells strange?
Collection: headers (43 chunks)   filter: {'source': 'food_safety.md'}   chunks that pass the filter: 9

1. distance 0.324  [food_safety.md#8]  Dabbaloop Kitchen Food Safety and Allergen Guide > 7. If something seems wrong
   ## 7. If something seems wrong If the food smells sour, looks unusual or the seal on the lid is b...
2. distance 0.625  [food_safety.md#4]  Dabbaloop Kitchen Food Safety and Allergen Guide > 3. Allergens in our food
   ## 3. Allergens in our food Our menu labels show five allergens. Milk: paneer, curd, ghee and but...
3. distance 0.669  [food_safety.md#7]  Dabbaloop Kitchen Food Safety and Allergen Guide > 6. Storing and reheating at home
   ## 6. Storing and reheating at home - Eat your meal within 3 hours of delivery. If you will eat l...
```

Only 9 of the 43 rows were compared, and `support.md` is gone from the results. The distance is a cosine **distance**:
lower is closer. Repeat `--file` for several files (`--file food_safety.md --file support.md` becomes
`{'source': {'$in': [...]}}`), and use `--strategy recursive -k 8` to see Episode 07's ranking again: `refunds.md#4` is
8th at distance 0.564, exactly as in Episode 07.

The rule that makes all of this work: the question and the chunks must be embedded by the **same** model.
(Many stores also offer **hybrid search**, which mixes a keyword score with the vector score; we do not use it here.)

> [!WARNING]
> **Common errors**
> - `No chunk passed the filter. File names are exact, for example: delivery.md, food_safety.md, ...` - the filter compares exact text: `foodsafety.md` matches nothing.
> - The `tokens`, `character` and `recursive` collections print `no breadcrumb (this splitter keeps no headings)`. Only the header strategy has `h1` and `h2` in its metadata.

## Step 14 - Compare with FAISS

**Goal:** put a library next to an embedded database, with the same chunks and the same vectors.
**New concept:** FAISS (from Meta) is a **library**: an index object inside your Python program that you can save to
files. Chroma is a **database**: it also stores the text and metadata, filters, and persists by itself.
LangChain's FAISS uses `IndexFlatL2`, an **exact** search that compares the question with every row; Chroma uses HNSW.
L2 is the straight-line distance between the tips of two vectors (FAISS reports it squared). Like Chroma's cosine
distance, lower means closer.

The FAISS lines of `chunklab/faiss_compare.py`, put together:

```python
from langchain_community.vectorstores import FAISS   # langchain-community is archived; FAISS still works

faiss_store = FAISS.from_embeddings(list(zip(texts, vectors)), embeddings, metadatas=metadatas, ids=ids)
faiss_store.save_local(str(FAISS_DIR))
loaded = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
```

Both stores get vectors embedded **once**, and both libraries are imported before the timers start, so the times
below measure the stores, not MiniLM and not Python loading a library.

```bash
python -m chunklab faiss
```

Expected output (your times will differ):

```text
Same 43 'headers' chunks for both stores. Embedded once with MiniLM in 0.43 s (that cost is the same for every store).

                                       build  query (median)     on disk
Chroma  (HNSW, cosine distance)       82.7ms         0.306ms      643 KB
FAISS   (IndexFlatL2, exact)           1.2ms         0.010ms       86 KB
(Chroma's folder holds a SQLite database plus its index files; FAISS saves index.faiss + index.pkl.)

Same top 3, in the same order, for 10 of 10 questions.

Example: What is the refund window for a missed lunch?
  1. Chroma delivery.md#7     cosine distance 0.442   FAISS delivery.md#7     squared L2 0.884
  2. Chroma refunds.md#8      cosine distance 0.475   FAISS refunds.md#8      squared L2 0.950
  3. Chroma refunds.md#3      cosine distance 0.493   FAISS refunds.md#3      squared L2 0.986
For unit-length vectors like MiniLM's, squared L2 = 2 x cosine distance, so the order is the same.

Saved FAISS to faiss_index/ and loaded it back: 43 vectors. (allow_dangerous_deserialization=True: index.pkl is a pickle, so only load files you made yourself.)
```

At 43 rows both are instant for a human. Chroma's extra time and disk space buy you a real database: text, metadata,
filters and persistence in one place. FAISS is smaller and faster, and you manage the files yourself.

> **Safety note.** `index.pkl` is a Python pickle, and loading a pickle can run code hidden inside it. That is why
> `load_local` refuses unless you pass `allow_dangerous_deserialization=True`. Only load index files that **you**
> created; never one you downloaded.

**Extra: exact versus approximate at scale.** With 43 rows you cannot see why HNSW exists. With 100,000 made-up
vectors you can (this runs FAISS only, about 10 seconds):

```bash
python -m chunklab faiss --scale
```

Expected output (your times will differ):

```text
100,000 made-up vectors of 384 numbers (seed 8), 100 questions, 1 CPU core

                              build      query   comparisons per query
exact (IndexFlatIP)           0.01s    2.331ms                 100,000
HNSW (IndexHNSWFlat)          8.79s    0.105ms                     504

HNSW found the same top 3 as exact search for 92 of 100 questions: faster, but approximate. (Real embeddings behave differently from made-up ones.)
```

Exact search compares the question with all 100,000 rows. HNSW hops through its graph and computes only about 504
distances, more than 20 times faster per question, but it found exactly the same top 3 for 92 questions out of 100.
It paid for that speed with a slow build (8.79 seconds instead of 0.01).

> [!WARNING]
> **Common errors**
> - `ValueError: The de-serialization relies loading a pickle file. ...` - `load_local` needs `allow_dangerous_deserialization=True`. Read the safety note first.
> - `DeprecationWarning: langchain-community is being sunset` - expected. FAISS still lives in `langchain-community` (archived in 2026, still working).
> - `OMP: Error #15 ...` on macOS - see Step 12. `faiss_compare.py` already sets `KMP_DUPLICATE_LIB_OK=True`.

## Choosing a vector store (as of 2026-10-23)

| Rung | What it is | Examples | Pick it when |
|---|---|---|---|
| Library | An index inside your program; you save the files | FAISS | Python only, you want speed and full control |
| Embedded database | A database that runs inside your app and saves to a folder | Chroma (this episode) | Learning, prototypes, small apps. Start here |
| Database extension | Vectors inside the database you already run | pgvector for PostgreSQL | Your data already lives in Postgres |
| Vector database server | A separate server you host | Qdrant, Weaviate, Milvus | Many users, big collections, a team to run servers |
| Managed cloud | Someone else runs the servers | Pinecone and others | You want no servers to run and accept the bill |

Chroma itself can also run as a separate server: the `chroma run` command starts one, and `chromadb.HttpClient`
connects to it. Products change fast: before you choose,
read the current docs and pricing, and move up a rung only when your own numbers say you need it.

## Step 15 - Answer with citations

**Goal:** answer a customer with Gemini and cite the section every fact came from.
**New concept:** the breadcrumb metadata becomes the citation. Each extract in the prompt starts with a tag such as
`[refunds.md > Missed or late meal]`, built from the chunk's `source` and `h2`, so the tag always points at a real section.

`chunklab/ask.py` (the important lines):

```python
def format_context(docs):
    """Each extract starts with its tag and its full breadcrumb, then the chunk text."""
    return "\n\n".join(f"[{citation(doc)}] ({breadcrumb(doc)})\n{doc.page_content}" for doc in docs)

    ...
    answer = (
        {"context": lambda inputs: format_context(inputs["docs"]), "question": lambda inputs: inputs["question"]}
        | PROMPT
        | llm
        | StrOutputParser()          # plain text, also for Gemini's list-of-blocks replies
    )
```

First look at the exact prompt, with no model and no key:

```bash
python -m chunklab ask --show-prompt "What is the refund window for a missed lunch?"
```

Expected output (the extracts part):

```text
Handbook extracts:
[delivery.md > Late or missed meal] (Dabbaloop Delivery Handbook > Late or missed meal)
## Late or missed meal  
The app shows your rider's live location from the moment your meal leaves the kitchen. If the meal still has not arrived 45 minutes after your slot ends, it is treated as missed. Report it in the app within 6 hours of the slot ending to get a full wallet credit. The full rules are in the Refunds and Credits Policy.

[refunds.md > What we cannot refund] (Dabbaloop Refunds and Credits Policy > What we cannot refund)
...
[refunds.md > Missed or late meal] (Dabbaloop Refunds and Credits Policy > Missed or late meal)
## Missed or late meal  
...
----- human message -----
What is the refund window for a missed lunch?
```

Now ask Gemini:

```bash
python -m chunklab ask "What is the refund window for a missed lunch?"
```

Expected output **shape** (Gemini writes its own sentences, so your wording will differ; the sources and distances
come from the real search, so you should see these three):

```text
Question: What is the refund window for a missed lunch?
Answer: <one to three sentences from Gemini, each ending with a tag such as [refunds.md > Missed or late meal]>
Sources:
  cited  [delivery.md > Late or missed meal]  distance 0.442  (Dabbaloop Delivery Handbook > Late or missed meal)
         [refunds.md > What we cannot refund]  distance 0.475  (Dabbaloop Refunds and Credits Policy > What we cannot refund)
  cited  [refunds.md > Missed or late meal]  distance 0.493  (Dabbaloop Refunds and Credits Policy > Missed or late meal)
```

`cited` marks the sources whose tag appears in the answer; which ones Gemini cites can differ from run to run.
Without a key, you get the free-key help from Step 0.3 instead, and nothing else runs.

> [!WARNING]
> **Common errors**
> - `No API key found for LLM_PROVIDER=gemini` - see Step 0.3.
> - `The model call failed: ... 429 RESOURCE_EXHAUSTED` - free-tier limit reached. Wait a minute and ask again.
> - The answer has no tags - the model ignored rule 1. Try again, or set `LLM_MODEL=gemini-3.8-flash` in `.env` (smarter, slower, uses more quota).

## Fix-It Corner: when retrieval goes wrong

| Symptom | Check first |
|---|---|
| The right rule is in the handbook but never retrieved | `python -m chunklab search "..." -k 8`: where does it rank? Is it cut away from its heading (Step 2)? |
| A chunk is longer than you asked | A list or table with no separator inside (Step 2). Use the recursive splitter. |
| Long chunks match badly | The embedder only reads 254 tokens of text (Step 4). Measure with `measure`. |
| Answers cite no section | Use the header strategy; only it has `h1` and `h2` metadata (Step 5). |
| Scores changed after an edit | Re-run `index`, then `score`. Compare with the numbers you wrote down. |
| Results come from the wrong file | Add a metadata filter (Step 13). |

## Final challenge

Add **true semantic chunking** (`SemanticChunker` from `langchain-experimental`, which cuts where the meaning of the
next sentence jumps) as a fifth strategy, keep every chunk under the embedder's limit, and try to beat
headers + recursive on the scoreboard. Details in [CHALLENGE.md](CHALLENGE.md).

## What you learned

- Chunk for three reasons: sharper search, smaller prompts, and an embedder that silently stops reading after 256 tokens.
- A vector store row is id + vector + text + metadata. A search compares vectors by distance, exactly or with HNSW hops, after any metadata filter.
- `CharacterTextSplitter` cuts at one separator only; `RecursiveCharacterTextSplitter` walks down a separator ladder; `TokenTextSplitter` cuts at a token count; `MarkdownHeaderTextSplitter` cuts at headings and keeps them as metadata.
- Measure chunk size in the embedder's own tokens (`from_huggingface_tokenizer`), and split long sections a second time.
- Decide with a scoreboard: on this handbook, headers + recursive found every answer first and lifted Episode 07's missing refund rule from 8th to 3rd.

## Next episode

**Episode 09: Prompt Engineering You Can Measure** - zero-shot, few-shot, chain of thought and JSON output, each one scored instead of guessed. -> `ep09-prompt-lab` (coming soon)

---

Dabbaloop, its people, prices and rules are invented for teaching. Any match with a real business is a coincidence.
