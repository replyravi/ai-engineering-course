# Episode 04: Foundation Models Explained: One Big Model, Many Jobs (Text, Images, Audio)

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: the link goes here when the episode is published  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep04-model-zoo

## What you will build

A small **model zoo**: a handful of short programs that borrow four foundation models and put each one to work
on **your own data**, ten reviews you write about a tea stall and four photos you take at home.
Kettle Point, the tea stall in the sample data, is invented for this course.

| Program | Model | Family | What it gives back |
|---|---|---|---|
| `encoder.py` | all-MiniLM-L6-v2 (on your laptop) | encoder | 384 numbers per review, and the review closest to a question |
| `decoder_local.py` | Qwen2.5-0.5B-Instruct (on your laptop) | decoder | a short reply, written one token at a time |
| `decoder_cloud.py` | Gemini 3.5 Flash-Lite (Google's servers) | multimodal | the same reply, side by side with the small model |
| `zero_shot.py` | Gemini 3.5 Flash-Lite | multimodal | one label per review, with no training |
| `multimodal_clip.py` | CLIP ViT-B/32 (on your laptop) | contrastive image-text | a match score between each photo and each label |
| `caption.py` | Gemini 3.5 Flash-Lite | multimodal | one sentence describing a photo |
| `zoo_report.py` | reads `model_cards.json` | - | which model for which job |

The last step prints this table (real output):

```text
Which model for which job
╭───────────────┬───────────────────────┬───────────────┬─────────┬───────────────┬────────────────╮
│ job           │ model                 │ family        │ writes? │ runs on       │ business use   │
├───────────────┼───────────────────────┼───────────────┼─────────┼───────────────┼────────────────┤
│ Find similar  │ all-MiniLM-L6-v2      │ encoder       │ no      │ your laptop   │ yes            │
│ reviews       │                       │               │         │               │                │
│ Write text    │ Qwen2.5-0.5B-Instruct │ decoder       │ yes     │ your laptop   │ yes            │
│ offline       │                       │               │         │               │                │
│ Write fluent  │ Gemini 3.5 Flash-Lite │ multimodal    │ yes     │ Google's      │ check the      │
│ text, label   │                       │               │         │ servers       │ terms          │
│ reviews       │                       │               │         │               │                │
│ Label photos  │ CLIP ViT-B/32         │ contrastive   │ no      │ your laptop   │ research only  │
│ with your     │                       │ image-text    │         │               │                │
│ words         │                       │               │         │               │                │
│ Caption a     │ Gemini 3.5 Flash-Lite │ multimodal    │ yes     │ Google's      │ check the      │
│ photo         │                       │               │         │ servers       │ terms          │
╰───────────────┴───────────────────────┴───────────────┴─────────┴───────────────┴────────────────╯
5 jobs, 4 model cards. Facts checked on 2026-10-23.
Licences and model versions change: read the card again before you build a product on it.
```

**You will learn:**

- What a **foundation model** is: pre-trained once on huge, mostly unlabelled data with **self-supervision**, then adapted to many jobs
- The difference between **parameters** (the learned numbers inside a model) and **tokens** (the pieces of text it reads and writes), with real counts
- Encoder, decoder and contrastive image-text models, and why a foundation model **does not have to generate** anything
- **Open weights** on your laptop versus a **closed model** behind an API, on the same prompt
- Adapting a model with a **prompt only** (zero-shot with structured output) before you think of fine-tuning
- Reading a **model card**: size, modality, context window, licence, open or closed, source and the date you checked it

> **Seen an older tutorial?** Three things changed.
> - **Hugging Face pipelines:** transformers v5 removed `pipeline("summarization")`, `"translation_en_to_fr"`,
>   `"text2text-generation"`, `"question-answering"` and `"visual-question-answering"`. Old code stops with a `KeyError`.
>   In 2026 we use `"text-generation"` with a small chat model (step 5). `"zero-shot-image-classification"` for CLIP
>   is still there in transformers 5.20 (step 8).
> - **Reading replies:** Gemini 3 returns `response.content` as a list of blocks, like `[{'type': 'text', 'text': '...'}]`.
>   Print `response.text`.
> - **Sending images:** older LangChain code used an OpenAI-style block, `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`.
>   LangChain v1 has one standard block for every provider: `{"type": "image", "base64": "...", "mime_type": "image/jpeg"}` (step 9).
>   The old block still works with Gemini, but new code uses the standard one.

## Prerequisites

- Finished Episode 01, folder `ep01-four-kinds-of-ai`: the shared `.venv`, the root `.env` and your **free** Gemini key ([SETUP.md](../SETUP.md)).
- Episode 03 explains the three Transformer families in depth. The short version you need today:
  **encoders** read the whole text at once and turn it into numbers (BERT, MiniLM),
  **decoders** write one token at a time (GPT, Llama, Qwen), and **encoder-decoders** read first and then write (T5, BART).
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- About 45 minutes, and about **2 GB of free disk** for three local models. Each one downloads once, the first time a step uses it:

| Model | Download | Licence | OK for a business? |
|---|---|---|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | about 91 MB (already there if you did Episode 01) | Apache-2.0 | yes (keep the licence notice if you share the files) |
| [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | about 1 GB | Apache-2.0 | yes (same) |
| [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | about 605 MB | MIT for the [code release](https://github.com/openai/CLIP); the model card names no licence | **no**: the card says any deployed use, commercial or not, is out of scope. Fine for learning. |

Gemini runs on Google's servers, so nothing downloads for it. Its use is covered by the
[Gemini API terms](https://ai.google.dev/gemini-api/terms). On the free tier Google may use what you send to improve
its products, so never send private text or photos. All licences were checked on 2026-10-23; step 10 shows how to keep that date with the facts.

## What is in this folder

```text
ep04-model-zoo/
├── data/reviews.txt          Step 2: ten reviews of Kettle Point (replace them with your own)
├── photos/                   Step 8: cup.jpg, plant.jpg, shoe.jpg, book.jpg (replace them with your own photos)
├── check_setup.py            Step 1: environment, key and which models are downloaded
├── common.py                 small helpers every script shares (reviews, photos, parameter counter, tables)
├── encoder.py                Steps 3-4: embed the reviews, count parameters, rank by cosine similarity
├── decoder_local.py          Step 5: a small open-weights decoder on your laptop
├── decoder_cloud.py          Steps 6 and 12: the same prompt through Gemini (or Ollama)
├── zero_shot.py              Step 7: one label per review, prompt only
├── multimodal_clip.py        Step 8: CLIP scores photos against your labels
├── caption.py                Step 9: Gemini describes a photo
├── model_cards.json          Step 10: the facts about today's four models, with sources and dates
├── model_cards_schema.py     Step 10: checks the cards
├── zoo_report.py             Step 11: which model for which job
├── tools/make_sample_photos.py   how the sample pictures were drawn (you do not need to run it)
├── llm.py, testing/          shared helpers (synced from _template, do not edit here)
├── tests/                    Step 13: offline tests (with their own copy of the sample data)
├── requirements.txt          the shared ../requirements-base.txt plus rich (tables)
├── CHALLENGE.md
└── outputs/                  made when you run steps 5 and 7 (git ignores it)
```

The four sample pictures are simple drawings made by `tools/make_sample_photos.py`, so no one else's photo is copied.
They are released as CC0 (public domain). Your own photos are better: step 8 shows how to use them.

Every file is already here. Type the code yourself as we go, or read along and run it.

## Step 0 - Get ready (about 2 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the repo root
(`ai-engineering-course/`). You created them once in Episode 1 ([SETUP.md](../SETUP.md)). New here? Do SETUP.md first, then come back.

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd ep04-model-zoo
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep04-model-zoo
```

Expected: your prompt starts with `(.venv)` and ends in `ep04-model-zoo`.

> [!WARNING]
> **Common errors**
> - `.venv\Scripts\Activate.ps1 cannot be loaded` / `No such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

### 0.2 Install this episode's one extra library

This episode uses the shared `requirements-base.txt` plus **rich**, the library that draws the tables.
Most setups already have rich (Chroma needs it too), but we name it because our code imports it. From this folder:

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
> - `Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'` - you are still in the repo root. Run `cd ep04-model-zoo` first, then install again.

---

## Step 1 - Check the setup

**Goal:** make sure the shared `.venv` and root `.env` from Episode 1 work here.
**New idea:** nothing new: same environment, same key. This check also tells you which of today's local models are already on your computer.

`check_setup.py`:

```python
"""
check_setup.py - check your workbench before Episode 4 (step 1).

Same shared .venv and root .env as Episode 1. This also tells you which of today's
three local models are already on your computer (the rest download on first use).
Nothing is sent over the internet, and your key is never printed.

Run:  python check_setup.py
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

import llm

PACKAGES = ["langchain", "langchain-google-genai", "transformers", "torch",
            "sentence-transformers", "pillow", "rich"]

# (model id, download size, the step that uses it)
LOCAL_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "about 91 MB", "step 3"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "about 1 GB", "step 5"),
    ("openai/clip-vit-base-patch32", "about 605 MB", "step 8"),
]


def line(name: str, value: str) -> None:
    print(f"{name + ' ':.<40} {value}")


def is_downloaded(model_id: str) -> bool:
    """True if the model is already in your Hugging Face cache (checked without the internet)."""
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def main() -> int:
    problems = 0
    print("Episode 4 - setup check\n")

    v = sys.version_info
    ok_python = (3, 11) <= (v.major, v.minor) <= (3, 13)
    line("Python", f"{v.major}.{v.minor}.{v.micro} " + ("OK" if ok_python else "NOT SUPPORTED - use Python 3.13"))
    problems += not ok_python

    in_venv = sys.prefix != sys.base_prefix
    line("Virtual env", f"active ({Path(sys.prefix).name})" if in_venv else "NOT active - activate .venv first")
    problems += not in_venv

    for package in PACKAGES:
        try:
            line(package, version(package))
        except PackageNotFoundError:
            line(package, "MISSING - run: pip install -r requirements.txt")
            problems += 1

    line(".env file", f"found ({Path(llm.ENV_FILE).parent.name}/.env)" if llm.ENV_FILE
         else "NOT FOUND - copy .env.example to .env in the repo root")
    try:
        provider = llm.get_provider()
    except ValueError as error:          # a typo in LLM_PROVIDER
        line("Provider", f"WRONG - {error}")
        return 1
    line("Provider", provider)
    line("Model", llm.get_model_name(provider))

    names = llm.API_KEY_NAMES[provider]
    found = [n for n in names if os.getenv(n, "").strip() and not os.getenv(n, "").startswith("paste-your")]
    if not names:
        line("API key", "not needed (local model)")
    elif found:
        line("API key", f"{found[0]} found (hidden)")
    else:
        line("API key", f"MISSING - put {names[0]}=... in the root .env file")
        problems += 1

    print()
    for model_id, size, step in LOCAL_MODELS:
        line(model_id, "downloaded" if is_downloaded(model_id) else f"not yet ({size}, downloads in {step})")

    print()
    if problems:
        print(f"{problems} thing(s) to fix, see above.")
        if provider in llm.WHERE_TO_GET_A_KEY and not found:
            print("\n" + llm.WHERE_TO_GET_A_KEY[provider])
        return 1
    print(f"Provider {provider}, key found. OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python check_setup.py
```

Expected output with your key in place (your Python patch number will differ; models you have not used yet say `not yet`):

```text
Episode 4 - setup check

Python ................................. 3.13.x OK
Virtual env ............................ active (.venv)
langchain .............................. 1.4.7
langchain-google-genai ................. 4.4.2
transformers ........................... 5.20.2
torch .................................. 2.14.1
sentence-transformers .................. 6.2.0
pillow ................................. 13.0.0
rich ................................... 15.0.0
.env file .............................. found (ai-engineering-course/.env)
Provider ............................... gemini
Model .................................. gemini-3.5-flash-lite
API key ................................ GEMINI_API_KEY found (hidden)

sentence-transformers/all-MiniLM-L6-v2 . downloaded
Qwen/Qwen2.5-0.5B-Instruct ............. downloaded
openai/clip-vit-base-patch32 ........... downloaded

Provider gemini, key found. OK
```

Before you add a key, it prints this instead (a real run with no `.env` file; the environment was called
`venv-langchain` on the recording machine, yours says `.venv`):

```text
Episode 4 - setup check

Python ................................. 3.11.17 OK
Virtual env ............................ active (venv-langchain)
langchain .............................. 1.4.7
langchain-google-genai ................. 4.4.2
transformers ........................... 5.20.2
torch .................................. 2.14.1
sentence-transformers .................. 6.2.0
pillow ................................. 13.0.0
rich ................................... 15.0.0
.env file .............................. NOT FOUND - copy .env.example to .env in the repo root
Provider ............................... gemini
Model .................................. gemini-3.5-flash-lite
API key ................................ MISSING - put GEMINI_API_KEY=... in the root .env file

sentence-transformers/all-MiniLM-L6-v2 . downloaded
Qwen/Qwen2.5-0.5B-Instruct ............. downloaded
openai/clip-vit-base-patch32 ........... downloaded

1 thing(s) to fix, see above.

Get a FREE Gemini key (no credit card needed):
  1. Open https://aistudio.google.com/apikey
  2. Sign in with your Google account and accept the terms
  3. Click 'Create API key' and copy it
  4. Put it in your .env file like this:  GEMINI_API_KEY=paste-your-key-here
```

If your root `.env` exists but still says `paste-your-key-here`, the `.env file` line says `found (ai-engineering-course/.env)`
and the `API key` line still says `MISSING`.

On a computer that has never downloaded the models, the three model lines say (real output, empty model cache):

```text
sentence-transformers/all-MiniLM-L6-v2 . not yet (about 91 MB, downloads in step 3)
Qwen/Qwen2.5-0.5B-Instruct ............. not yet (about 1 GB, downloads in step 5)
openai/clip-vit-base-patch32 ........... not yet (about 605 MB, downloads in step 8)
```

That is fine: each model downloads the first time its step runs.

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama` (step 12). Nothing else changes.

> [!WARNING]
> **Common errors**
> - `API key ... MISSING` - the root `.env` is missing, is named `.env.txt` (Windows hides extensions), sits in the wrong folder (it belongs in `ai-engineering-course/`), or still says `paste-your-key-here`.
> - `rich ... MISSING` - run `pip install -r requirements.txt` from this folder (step 0.2).
> - `Virtual env ... NOT active` - activate the `.venv` first (step 0.1).

## Step 2 - Write your own reviews file

**Goal:** a ten-line `data/reviews.txt`, one review per line.
**New idea:** your data, not a downloaded data set. None of today's models has ever seen these sentences,
which is exactly the point: a foundation model was trained once, by someone else, and still works on text it never met.

The sample file, written for this course (Kettle Point is invented):

```text
The masala chai here is strong and fresh, just how my grandmother used to make it.
I waited twenty minutes for one cup while the staff served their friends first.
The samosa tasted like it was fried yesterday and warmed up again.
Lovely ginger tea, but they charge ten rupees more than last month for the same small cup.
They take UPI, so I never have to hunt for change during the morning rush.
The bench under the tin roof is a lifesaver when the monsoon rain starts.
My filter coffee arrived lukewarm and the cup was sticky on the outside.
The owner is friendly and remembers my order every single day.
Bun maska was soft and buttery, though the plate could have been cleaner.
After six in the evening it is so crowded that I cannot even find a place to stand.
```

Replace them with ten reviews of a place you know: a canteen, a phone, a movie. Mix happy, unhappy and mixed
ones, and write them in your own words. Keep one review per line.

Every script reads the file with `load_reviews()` from `common.py`, the helper file all of today's scripts share:

```python
"""
common.py - small helpers shared by the Episode 04 scripts.

  load_reviews()       read data/reviews.txt, one review per line (step 2)
  count_parameters()   how many learned numbers a model has (steps 3, 5, 8)
  in_millions()        22713216 -> "22.7 million"
  find_photos()        the pictures in photos/ (steps 8, 9)
  make_table()         a neat table for the terminal (uses the rich library)
  OWNER_SYSTEM, OWNER_PROMPT   the ONE prompt we give every text model (steps 5, 6, 12)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

HERE = Path(__file__).resolve().parent
REVIEWS_FILE = HERE / "data" / "reviews.txt"
PHOTOS_DIR = HERE / "photos"
OUTPUTS = HERE / "outputs"            # results we save for later steps (git ignores this folder)
LOCAL_RESULT_FILE = OUTPUTS / "local_decoder.json"   # written by decoder_local.py, read by decoder_cloud.py
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp"}

console = Console()

# The same request goes to the small local model (step 5) and to Gemini (step 6).
OWNER_SYSTEM = "You are the owner of Kettle Point, a small tea stall. Reply to customers politely and briefly."
OWNER_PROMPT = (
    'A customer wrote this review: "My filter coffee arrived lukewarm and the cup was sticky on the outside." '
    "Write a two-sentence reply that says sorry and promises one fix."
)


def load_reviews(path=REVIEWS_FILE) -> list[str]:
    """Return the reviews in the file, one per line. Empty lines are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def count_parameters(model) -> int:
    """Count every learned number (weight) inside a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def in_millions(count: int) -> str:
    """22713216 -> '22.7 million'. Easier to say out loud than eight digits."""
    return f"{count / 1_000_000:.1f} million"


def find_photos(folder=PHOTOS_DIR) -> list[Path]:
    """All pictures in the folder (jpg, jpeg, png, webp), sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_TYPES)


def make_table(columns: list[str], rows: list[list[str]], title: str | None = None,
               keep_whole: tuple[str, ...] = ()) -> Table:
    """Build a rich Table: print it with console.print(table).

    keep_whole: names of columns that must never wrap (for example model names).

    Every cell is plain Text, so square brackets in your data are shown as they are
    (rich would otherwise read "[best]" as a style tag and hide it).
    """
    table = Table(title=Text(title) if title else None, box=box.ROUNDED, title_justify="left")
    for name in columns:
        # wrap long text onto the next line, never cut it off with "..."
        table.add_column(name, overflow="fold", no_wrap=name in keep_whole)
    for row in rows:
        table.add_row(*[Text(str(cell)) for cell in row])
    return table
```

```bash
python -c "import common; print(len(common.load_reviews()), 'reviews')"
```

Expected output:

```text
10 reviews
```

> [!WARNING]
> **Common errors**
> - `FileNotFoundError: ... data/reviews.txt` - the `data` folder was renamed or moved. `common.py` looks for `data/reviews.txt` next to itself.
> - A count other than 10 - a review is split over two lines, or two reviews share one line. Empty lines are skipped, so they do no harm.
> - `UnicodeDecodeError` - Windows Notepad saved the file in another encoding. In VS Code, click the encoding in the bottom bar and choose **Save with Encoding > UTF-8**.

## Step 3 - Embed every review and count the parameters

**Goal:** turn each review into a vector, and count the learned numbers inside the model.
**New idea:** an **encoder** foundation model. all-MiniLM-L6-v2 was pre-trained on a huge amount of text, then tuned
on more than a billion sentence pairs to put sentences with the same meaning close together. It reads; it writes nothing.

Two words we use all episode:

- **Parameters** are the learned numbers inside the model, its "weights". They are fixed once training ends.
- **Tokens** are the small pieces of text a model reads or writes. A model is trained on many tokens, and it reads
  your tokens when you use it.

`encoder.py` (steps 3 and 4 both use it):

```python
"""
encoder.py - an ENCODER foundation model reads your reviews (steps 3 and 4).

all-MiniLM-L6-v2 was pre-trained by other people on a huge amount of text. It has never
seen YOUR reviews, yet it turns each one into 384 numbers (a vector) that capture its
meaning. Reviews that mean similar things get vectors that point the same way.

It writes nothing. It only reads, and turns text into numbers.
So a foundation model does not have to be generative.

Run:  python encoder.py                               (step 3, then a search for the default query)
      python encoder.py "can I pay with my phone?"      (step 4 with your own query)
"""

import re
import sys
import time

import numpy as np

from common import REVIEWS_FILE, console, count_parameters, in_millions, load_reviews, make_table
from llm import get_embeddings

DEFAULT_QUERY = "the snacks were stale"
TOP = 3   # how many reviews to show

# Little words that say nothing about meaning (used only to compare words, step 4).
FILLER = {"a", "an", "the", "and", "or", "is", "are", "was", "were", "it", "i", "my", "me", "they",
          "to", "of", "for", "with", "in", "on", "at", "so", "can", "do", "did", "be", "when", "this",
          "that", "there", "here", "too", "very"}


# ---------------------------------------------------------------------------
# Step 3 - embed every review and count the parameters
# ---------------------------------------------------------------------------

def embed(texts: list[str], embeddings) -> np.ndarray:
    """One vector per text: a table with one row per text and 384 columns."""
    return np.array(embeddings.embed_documents(texts))


def parameters_of(embeddings) -> int | None:
    """Parameters of the local model inside get_embeddings(), or None for a cloud model."""
    model = getattr(embeddings, "_client", None)   # the sentence-transformers model inside the LangChain wrapper
    return count_parameters(model) if model is not None else None


# ---------------------------------------------------------------------------
# Step 4 - rank the reviews by cosine similarity to a query
# ---------------------------------------------------------------------------

def cosine_similarity(a, b) -> float:
    """How closely two vectors point the same way: 1.0 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank(query_vector, vectors, texts: list[str], top: int = TOP) -> list[tuple[float, int, str]]:
    """The `top` texts closest to the query, best first, as (score, review number, text)."""
    scored = [(cosine_similarity(query_vector, v), number, text)
              for number, (v, text) in enumerate(zip(vectors, texts), start=1)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top]


def shared_words(a: str, b: str) -> set[str]:
    """Meaningful words that appear in both texts (filler words removed)."""
    def words(text):
        return set(re.findall(r"[a-z]+", text.lower())) - FILLER
    return words(a) & words(b)


def main(argv=None, embeddings=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    query = " ".join(argv).strip() or DEFAULT_QUERY

    reviews = load_reviews()
    print(f"Read {len(reviews)} reviews from data/{REVIEWS_FILE.name}")

    # Step 3: the encoder turns each review into a vector
    start = time.perf_counter()
    if embeddings is None:
        embeddings = get_embeddings()          # local all-MiniLM-L6-v2 (no key, no cost)
    vectors = embed(reviews, embeddings)
    seconds = time.perf_counter() - start

    name = getattr(embeddings, "model_name", None) or type(embeddings).__name__
    print(f"Encoder: {name}")
    print(f"{vectors.shape[0]} vectors, {vectors.shape[1]} numbers each ({seconds:.1f} s, including loading the model)")
    print("Review 1 starts with:", ", ".join(f"{x:.3f}" for x in vectors[0][:5]), "...")
    params = parameters_of(embeddings)
    if params:
        print(f"Parameters: {params:,} (about {in_millions(params)})")
    else:
        print("Parameters: not published (a cloud model)")

    # Step 4: embed the query the same way, then find the closest reviews
    query_vector = np.array(embeddings.embed_query(query))
    results = rank(query_vector, vectors, reviews)
    rows = [[f"{score:.2f}", f"#{number}", text] for score, number, text in results]
    console.print()
    console.print(make_table(["score", "review", "text"], rows, title=f'Closest reviews to "{query}"'))

    common = shared_words(query, results[0][2])
    print("Words shared by the query and the top review:",
          ", ".join(sorted(common)) if common else "none, it matched on meaning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python encoder.py
```

Expected output, first part (your seconds will differ; the first run also shows download bars):

```text
Read 10 reviews from data/reviews.txt
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 6713.70it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Encoder: sentence-transformers/all-MiniLM-L6-v2
10 vectors, 384 numbers each (5.3 s, including loading the model)
Review 1 starts with: -0.031, -0.005, 0.001, 0.074, -0.043 ...
Parameters: 22,713,216 (about 22.7 million)
```

Ten reviews in, ten vectors of 384 numbers out. The model holds about 22.7 million parameters, and none of them
changed: we only used the model, we did not train it.

> [!WARNING]
> **Common errors**
> - `OSError: We couldn't connect to 'https://huggingface.co'` on the first run - the model downloads once (about 91 MB), so you need internet the first time. After that it works offline.
> - Offline after that first download, you may see `Could not reach the Hub ... Retrying in 1s` lines and wait about half a minute before the answer appears. That is the library trying to check the files on the Hub before it falls back to your downloaded copy; the result is the same.
> - `Warning: You are sending unauthenticated requests to the HF Hub` - harmless. It only means you have no Hugging Face account token. Downloads still work.
> - On a Mac, the very first line can be `W... NOTE: Redirects are currently not supported in MacOs.` - a harmless message from PyTorch. It appears in steps 3, 5 and 8 and is left out of the expected output on this page.
> - `ModuleNotFoundError: No module named 'rich'` - step 0.2.

## Step 4 - Rank reviews against a query

**Goal:** find the review that means the same thing as a question, even with different words.
**New idea:** **cosine similarity**. Two vectors that point the same way get a score near 1; unrelated ones get a score near 0.
We embed the query with the same model and sort the reviews by score.

The two functions that do it (already in `encoder.py`):

```python
def cosine_similarity(a, b) -> float:
    """How closely two vectors point the same way: 1.0 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank(query_vector, vectors, texts: list[str], top: int = TOP) -> list[tuple[float, int, str]]:
    """The `top` texts closest to the query, best first, as (score, review number, text)."""
    scored = [(cosine_similarity(query_vector, v), number, text)
              for number, (v, text) in enumerate(zip(vectors, texts), start=1)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top]
```

The second part of the output of `python encoder.py` (real output):

```text
Closest reviews to "the snacks were stale"
╭───────┬────────┬─────────────────────────────────────────────────────────────────────────────────╮
│ score │ review │ text                                                                            │
├───────┼────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 0.40  │ #3     │ The samosa tasted like it was fried yesterday and warmed up again.              │
│ 0.28  │ #2     │ I waited twenty minutes for one cup while the staff served their friends first. │
│ 0.28  │ #9     │ Bun maska was soft and buttery, though the plate could have been cleaner.       │
╰───────┴────────┴─────────────────────────────────────────────────────────────────────────────────╯
Words shared by the query and the top review: none, it matched on meaning
```

The top review shares **no** words with the query. It matched on meaning: stale snacks and a samosa fried yesterday.
Try your own question:

```bash
python encoder.py "is it a good place to sit when it rains"
```

```text
Closest reviews to "is it a good place to sit when it rains"
╭───────┬────────┬─────────────────────────────────────────────────────────────────────────────────╮
│ score │ review │ text                                                                            │
├───────┼────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 0.50  │ #6     │ The bench under the tin roof is a lifesaver when the monsoon rain starts.       │
│ 0.40  │ #10    │ After six in the evening it is so crowded that I cannot even find a place to    │
│       │        │ stand.                                                                          │
│ 0.14  │ #5     │ They take UPI, so I never have to hunt for change during the morning rush.      │
╰───────┴────────┴─────────────────────────────────────────────────────────────────────────────────╯
Words shared by the query and the top review: none, it matched on meaning
```

**An honest miss.** Small models have gaps. Ask about paying by phone:

```bash
python encoder.py "can I pay with my phone?"
```

```text
Closest reviews to "can I pay with my phone?"
╭───────┬────────┬─────────────────────────────────────────────────────────────────────────────────╮
│ score │ review │ text                                                                            │
├───────┼────────┼─────────────────────────────────────────────────────────────────────────────────┤
│ 0.11  │ #8     │ The owner is friendly and remembers my order every single day.                  │
│ 0.04  │ #4     │ Lovely ginger tea, but they charge ten rupees more than last month for the same │
│       │        │ small cup.                                                                      │
│ 0.03  │ #6     │ The bench under the tin roof is a lifesaver when the monsoon rain starts.       │
╰───────┴────────┴─────────────────────────────────────────────────────────────────────────────────╯
Words shared by the query and the top review: none, it matched on meaning
```

Review 5 ("They take UPI...") is the right answer, but it is not even in the top three, and every score is low.
This small English model does not connect "UPI" with paying by phone. Notice the low top score too (0.11):
a low score is a sign that the search found nothing good. Bigger or multilingual embedding models know more words;
Episode 08 measures search quality properly.

> [!WARNING]
> **Common errors**
> - Your scores differ in the last digit - different computers can round slightly differently. The order should match.
> - Quotes on Windows - in PowerShell use straight double quotes around the query, exactly as shown.
> - `Parameters: not published (a cloud model)` and different scores - you set `EMBEDDINGS_PROVIDER=gemini` in `.env`. That works with your key, but the vectors then come from Google's embedding model, so the numbers on this page will not match. Remove the line to use the local model again.

## Step 5 - Run a small local decoder

**Goal:** a small model writes a reply on your own laptop, with no internet and no key.
**New idea:** **open weights**. Qwen2.5-0.5B-Instruct is a decoder: it writes one token at a time. Its weights are a
file you download (about 1 GB). It was pre-trained, then **instruction-tuned** so it follows requests. While it writes,
tokens flow through the model, but its weights do not change. We prove that with a fingerprint of every weight,
taken before and after. `pipeline("text-generation", ...)` from the transformers library downloads the model and its
tokenizer (the part that cuts text into tokens) and wraps them in one object you can call with a chat.

`decoder_local.py`:

```python
"""
decoder_local.py - a small DECODER with open weights, running on your own laptop (step 5).

Qwen2.5-0.5B-Instruct writes text one token at a time, like the big chat models do.
Its weights (the learned numbers) are a file on your disk, about 1 GB, downloaded once.
Licence: Apache-2.0, so you may use it in your own projects, also commercially.

While it answers, tokens flow through the model but the weights stay exactly the same.
We prove that with a fingerprint of all the weights, taken before and after.

Run:  python decoder_local.py
"""

import hashlib
import json
import time

import torch
from transformers import pipeline

from common import LOCAL_RESULT_FILE, OUTPUTS, OWNER_PROMPT, OWNER_SYSTEM, count_parameters, in_millions

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
MAX_NEW_TOKENS = 80                            # the longest reply we allow, in tokens


def load_decoder():
    """Load the model (the first run downloads it). device='cpu' works on every laptop."""
    return pipeline("text-generation", model=MODEL_ID, device="cpu")


def chat_messages(system: str, prompt: str) -> list[dict]:
    """A chat in the format every chat model understands: a list of role + content."""
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def generate(chat, system: str, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Send one chat to the model and return its reply as plain text."""
    messages = chat_messages(system, prompt)
    # do_sample=False: always pick the most likely next token, so you get the same reply every run
    out = chat(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"]   # the last message in the chat is the model's reply


def count_tokens(chat, text: str) -> int:
    """How many tokens (pieces of text) the model's tokenizer cuts this text into."""
    return len(chat.tokenizer(text)["input_ids"])


def prompt_tokens(chat, system: str, prompt: str) -> int:
    """Tokens the model really reads: our text plus the chat format markers around it."""
    text = chat.tokenizer.apply_chat_template(chat_messages(system, prompt), tokenize=False,
                                              add_generation_prompt=True)
    return count_tokens(chat, text)


def weights_fingerprint(model) -> str:
    """A short fingerprint of every weight. If one number changed, the fingerprint would change."""
    digest = hashlib.sha256()
    for p in model.parameters():
        digest.update(p.detach().contiguous().view(torch.uint8).numpy())   # the raw bytes of the numbers
    return digest.hexdigest()[:12]


def main() -> int:
    torch.manual_seed(SEED)   # only matters if you switch to do_sample=True

    start = time.perf_counter()
    chat = load_decoder()
    load_seconds = time.perf_counter() - start

    params = count_parameters(chat.model)
    print(f"Decoder: {MODEL_ID} (open weights, running on this computer)")
    print(f"Parameters: {params:,} (about {in_millions(params)})")
    print(f"Loaded in {load_seconds:.1f} s")

    before = weights_fingerprint(chat.model)
    start = time.perf_counter()
    reply = generate(chat, OWNER_SYSTEM, OWNER_PROMPT)
    seconds = time.perf_counter() - start
    after = weights_fingerprint(chat.model)

    print(f"\nPrompt: {OWNER_PROMPT}")
    print(f"\nReply ({seconds:.1f} s):\n{reply}")
    print(f"\nTokens: {prompt_tokens(chat, OWNER_SYSTEM, OWNER_PROMPT)} read, {count_tokens(chat, reply)} written")
    print(f"Weights fingerprint before {before}, after {after}: "
          f"{'unchanged' if before == after else 'CHANGED'}")

    OUTPUTS.mkdir(exist_ok=True)
    result = {"model": MODEL_ID, "parameters": params, "seconds": round(seconds, 1), "reply": reply}
    LOCAL_RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved outputs/{LOCAL_RESULT_FILE.name} (decoder_cloud.py compares with it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python decoder_local.py
```

Expected output (the first run downloads about 1 GB; your seconds will differ):

```text
Loading weights: 100%|██████████| 290/290 [00:00<00:00, 10581.17it/s]
Decoder: Qwen/Qwen2.5-0.5B-Instruct (open weights, running on this computer)
Parameters: 494,032,768 (about 494.0 million)
Loaded in 1.5 s

Prompt: A customer wrote this review: "My filter coffee arrived lukewarm and the cup was sticky on the outside." Write a two-sentence reply that says sorry and promises one fix.

Reply (1.0 s):
I'm really sorry for any inconvenience caused by the delay in your order. Please let me know if you need anything else or have any other questions.

Tokens: 71 read, 30 written
Weights fingerprint before 43ec35af9634, after 43ec35af9634: unchanged
Saved outputs/local_decoder.json (decoder_cloud.py compares with it)
```

Look closely at the reply. It apologises for a **delay**, but the customer complained about lukewarm coffee and a
sticky cup, and it promises no fix. That is typical of a model this small: fluent words, weak understanding.
Run it again and you get exactly the same reply, because `do_sample=False` always picks the most likely next token.

The 71 tokens it read include the chat-format markers around our text; it wrote 30 tokens.

> **Seen an older tutorial?** It may call `pipeline("text2text-generation")` or `pipeline("summarization")`.
> transformers v5 removed both. A small chat model with `pipeline("text-generation")` does the same jobs.

> [!WARNING]
> **Common errors**
> - It seems stuck on the first run - it is downloading about 1 GB. Leave it; the progress bars move.
> - `MemoryError` or the computer slows down - the script used about 1.5 GB of memory on the recording machine. Close other apps.
> - A different reply - you changed `do_sample`, the prompt in `common.py`, or `MAX_NEW_TOKENS`.

## Step 6 - Same prompt through Gemini

**Goal:** send the very same prompt to Gemini and compare the two replies side by side.
**New idea:** a **closed model behind an API**. Gemini's weights never leave Google's servers. We send text over the
internet and get text back. Google does not publish its parameter count, so the table says `not published`.

`decoder_cloud.py`:

```python
"""
decoder_cloud.py - the same prompt through Gemini, a CLOSED model behind an API (step 6).

Gemini's weights never leave Google's servers. We send text over the internet and get
text back, and Google does not publish how many parameters the model has.
At the end we put its reply next to the small local model's reply from step 5.

Step 12: put LLM_PROVIDER=ollama in the root .env file and run this file again.
The same code then answers from a model on your own computer.

Run:  python decoder_cloud.py      (needs your free Gemini key in the root .env file)
"""

import json
import re
import time

import httpx   # the web library the Ollama client uses (installed with LangChain)
from langchain_core.messages import HumanMessage, SystemMessage

from common import LOCAL_RESULT_FILE, OWNER_PROMPT, OWNER_SYSTEM, console, make_table
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

# Where each provider's model runs, and what it needs.
PROVIDER_FACTS = {
    "gemini": ("Google's servers", "closed (stays with Google)", "not published", "internet + free key"),
    "openai": ("OpenAI's servers", "closed (stays with OpenAI)", "not published", "internet + paid key"),
    "anthropic": ("Anthropic's servers", "closed (stays with Anthropic)", "not published", "internet + paid key"),
    "ollama": ("this computer (Ollama)", "open (a file on your disk)", "see ollama.com/library", "the Ollama app, no key"),
}

# A rough check, not a real quality score: does the reply talk about the customer's actual problem?
PROBLEM_WORDS = {"coffee", "lukewarm", "hot", "warm", "temperature", "sticky", "cup", "clean", "wipe"}


def ask(llm, system: str = OWNER_SYSTEM, prompt: str = OWNER_PROMPT) -> str:
    """Send the chat to the model and return the reply as plain text."""
    reply = llm.invoke([SystemMessage(system), HumanMessage(prompt)])
    return reply.text   # .text, not .content: Gemini 3 returns content as a list of blocks


def mentions_problem(reply: str) -> str:
    """'yes (coffee, sticky)' if the reply names the real problem, else 'no'."""
    found = sorted(PROBLEM_WORDS & set(re.findall(r"[a-z]+", reply.lower())))
    return f"yes ({', '.join(found)})" if found else "no"


def comparison_rows(local: dict | None, cloud: dict) -> list[list[str]]:
    """Rows for the side-by-side table: [what we compare, local model, cloud model]."""
    if local is None:
        local = {"model": "run decoder_local.py first", "parameters": None, "seconds": None, "reply": ""}
    runs_on, weights, parameters, needs = PROVIDER_FACTS[cloud["provider"]]
    return [
        ["Model", local["model"], cloud["model"]],
        ["Runs on", "this computer", runs_on],
        ["Weights", "open (a file on your disk)", weights],
        ["Parameters", f"{local['parameters']:,}" if local["parameters"] else "-", parameters],
        ["Needs", "nothing after the download", needs],
        ["Time for this reply", f"{local['seconds']} s" if local["seconds"] is not None else "-", f"{cloud['seconds']} s"],
        ["Names the real problem?", mentions_problem(local["reply"]) if local["reply"] else "-",
         mentions_problem(cloud["reply"])],
        ["Reply", local["reply"] or "-", cloud["reply"]],
    ]


def main(llm=None) -> int:
    try:
        provider = get_provider()              # reads LLM_PROVIDER from the root .env (gemini by default)
        model = get_model_name(provider)
        print(f"Provider: {provider}   Model: {model}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")        # friendly help (no key, or a typo in LLM_PROVIDER)
        return 1

    print(f"\nPrompt: {OWNER_PROMPT}")
    start = time.perf_counter()
    try:
        reply = ask(llm)
    except (ConnectionError, httpx.ConnectError) as error:   # no internet, or Ollama is not running
        print(f"\nCould not reach the model: {error}")
        if provider == "ollama":
            print("Start the Ollama app (https://ollama.com), run 'ollama pull qwen3.5:4b', then try again.")
        else:
            print("Check your internet connection, then try again.")
        return 1
    seconds = round(time.perf_counter() - start, 1)
    print(f"\nReply ({seconds} s):\n{reply}\n")

    local = json.loads(LOCAL_RESULT_FILE.read_text(encoding="utf-8")) if LOCAL_RESULT_FILE.exists() else None
    cloud = {"provider": provider, "model": model, "seconds": seconds, "reply": reply}
    table = make_table(["", "small local model (step 5)", f"{provider} model"],
                       comparison_rows(local, cloud), title="Same prompt, two models")
    console.print(table)
    print("Which reply would you put on the shop's page? That judgement is yours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python decoder_cloud.py
```

Expected output with your key. Gemini writes its own words, so the `<...>` parts are filled in by Gemini and your
time will differ. (This layout comes from a test run with a placeholder reply, not from Gemini.)

```text
Provider: gemini   Model: gemini-3.5-flash-lite

Prompt: A customer wrote this review: "My filter coffee arrived lukewarm and the cup was sticky on the outside." Write a two-sentence reply that says sorry and promises one fix.

Reply (<N> s):
<a two-sentence reply written by Gemini>

Same prompt, two models
╭─────────────────────────┬────────────────────────────────────┬───────────────────────────────────╮
│                         │ small local model (step 5)         │ gemini model                      │
├─────────────────────────┼────────────────────────────────────┼───────────────────────────────────┤
│ Model                   │ Qwen/Qwen2.5-0.5B-Instruct         │ gemini-3.5-flash-lite             │
│ Runs on                 │ this computer                      │ Google's servers                  │
│ Weights                 │ open (a file on your disk)         │ closed (stays with Google)        │
│ Parameters              │ 494,032,768                        │ not published                     │
│ Needs                   │ nothing after the download         │ internet + free key               │
│ Time for this reply     │ 0.8 s                              │ <N> s                             │
│ Names the real problem? │ no                                 │ <yes or no>                       │
│ Reply                   │ I'm really sorry for any           │ <a two-sentence reply written by  │
│                         │ inconvenience caused by the delay  │ Gemini>                           │
│                         │ in your order. Please let me know  │                                   │
│                         │ if you need anything else or have  │                                   │
│                         │ any other questions.               │                                   │
╰─────────────────────────┴────────────────────────────────────┴───────────────────────────────────╯
Which reply would you put on the shop's page? That judgement is yours.
```

The **Names the real problem?** row is a rough check, not a quality score: it only looks for words such as coffee,
lukewarm or sticky in the reply. Read both replies and decide which one you would put on the shop's page.

Without a key you get friendly help instead of a crash (real output):

```text
Provider: gemini   Model: gemini-3.5-flash-lite

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
(.env file in use right now: none found)
```

If your root `.env` exists but still holds the placeholder, the last line shows its path instead of `none found`.

> [!WARNING]
> **Common errors**
> - `429 RESOURCE_EXHAUSTED` - a free-tier limit. Wait a minute (per-minute limit) or until the daily reset. Your limits: https://aistudio.google.com/rate-limit
> - `API key not valid. Please pass a valid API key.` - the key was copied with a missing character or an extra space. Copy it again from AI Studio into the root `.env`.
> - The small model's column says `run decoder_local.py first` - run step 5 once; it saves its reply in `outputs/local_decoder.json`.
> - `Could not reach the model: ...` and `Check your internet connection` - Gemini runs on Google's servers, so this step needs internet.
> - You see `[{'type': 'text', ...}]` - you printed `.content`. Print `.text`.

## Step 7 - Zero-shot sentiment with structured output

**Goal:** one label per review, positive, negative or mixed, without training anything.
**New idea:** **adapting by prompt only**. From cheapest to most expensive, the ways to adapt a foundation model are:
write a better prompt, add your own documents to the prompt (RAG, Episode 07), fine-tune a small part of the model
(LoRA), fine-tune all of it, and, for big labs only, pre-train a new one. Always start at the bottom of that ladder.
Here the prompt is all we change. `with_structured_output` makes Gemini fill in a small form (the `ReviewLabels` class),
so we get exactly one label per review instead of a paragraph. All ten reviews go in **one** request, which saves quota.

`zero_shot.py`:

```python
"""
zero_shot.py - adapt a foundation model with a prompt only, no training (step 7).

We never show Gemini a single labelled example. We describe the job in the prompt,
and with_structured_output makes it fill in a form (the ReviewLabels class), so we get
back exactly one label per review instead of a paragraph.
All ten reviews go in ONE request, which saves your free-tier quota.

Run:  python zero_shot.py      (needs your free Gemini key in the root .env file)
"""

import json
from collections import Counter
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common import OUTPUTS, console, load_reviews, make_table
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

LABELS_FILE = OUTPUTS / "zero_shot_labels.json"


class ReviewLabel(BaseModel):
    """The sentiment of one numbered review."""

    number: int = Field(description="the review number, as given in the list")
    label: Literal["positive", "negative", "mixed"] = Field(
        description="positive, negative, or mixed (praises one thing and complains about another)")


class ReviewLabels(BaseModel):
    """One label for every review in the list."""

    labels: list[ReviewLabel]


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You label customer reviews of a small tea stall.\n"
     "positive: the customer is happy.\n"
     "negative: the customer is unhappy.\n"
     "mixed: the review praises one thing and complains about another.\n"
     "Give exactly one label for every numbered review."),
    ("human", "Reviews:\n{reviews}"),
])


def numbered(reviews: list[str]) -> str:
    """'1. first review' on the first line, '2. second review' on the next, and so on."""
    return "\n".join(f"{number}. {text}" for number, text in enumerate(reviews, start=1))


def build_labeller(llm=None):
    """prompt -> model that must answer with a ReviewLabels form."""
    if llm is None:
        llm = get_llm()
    return prompt | llm.with_structured_output(ReviewLabels)


def match_labels(reviews: list[str], answer: ReviewLabels) -> list[str]:
    """Put the model's labels in review order, and check that every review got exactly one."""
    by_number = {}
    for item in answer.labels:
        if item.number in by_number:
            raise ValueError(f"review {item.number} got two labels")
        by_number[item.number] = item.label
    expected = set(range(1, len(reviews) + 1))
    missing = sorted(expected - set(by_number))
    unknown = sorted(set(by_number) - expected)
    if missing:
        raise ValueError(f"no label for review(s) {missing}")
    if unknown:
        raise ValueError(f"labels for review numbers that do not exist: {unknown}")
    return [by_number[number] for number in sorted(expected)]


def label_reviews(reviews: list[str], llm=None) -> list[str]:
    """One request for all reviews. Returns one label per review, in order."""
    answer = build_labeller(llm).invoke({"reviews": numbered(reviews)})
    return match_labels(reviews, answer)


def main(llm=None) -> int:
    try:
        provider = get_provider()
        model = get_model_name(provider)
        print(f"Provider: {provider}   Model: {model}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")        # friendly help (no key, or a typo in LLM_PROVIDER)
        return 1

    reviews = load_reviews()
    print(f"Labelling {len(reviews)} reviews in one request. Labelled examples given to the model: 0\n")
    try:
        labels = label_reviews(reviews, llm)
    except ValueError as error:                 # the form came back incomplete or with a wrong label
        print(f"The model's answer did not fit the form ({error}). Run it again.")
        return 1

    rows = [[str(number), label, text] for number, (label, text) in enumerate(zip(labels, reviews), start=1)]
    console.print(make_table(["#", "label", "review"], rows, title="Zero-shot labels"))
    counts = Counter(labels)
    print("Counts:", ", ".join(f"{name} {counts[name]}" for name in ["positive", "negative", "mixed"]))

    OUTPUTS.mkdir(exist_ok=True)
    LABELS_FILE.write_text(json.dumps({"model": model, "labels": labels}, indent=2), encoding="utf-8")
    print(f"Saved outputs/{LABELS_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python zero_shot.py
```

Expected output with your key. Gemini chooses the labels, so `<label>` and `<N>` are filled in by Gemini.
(Layout from a test run with placeholder labels, not from Gemini.)

```text
Provider: gemini   Model: gemini-3.5-flash-lite
Labelling 10 reviews in one request. Labelled examples given to the model: 0

Zero-shot labels
╭────┬──────────┬──────────────────────────────────────────────────────────────────────────────────╮
│ #  │ label    │ review                                                                           │
├────┼──────────┼──────────────────────────────────────────────────────────────────────────────────┤
│ 1  │ <label>  │ The masala chai here is strong and fresh, just how my grandmother used to make   │
│    │          │ it.                                                                              │
│ 2  │ <label>  │ I waited twenty minutes for one cup while the staff served their friends first.  │
│ 3  │ <label>  │ The samosa tasted like it was fried yesterday and warmed up again.               │
│ 4  │ <label>  │ Lovely ginger tea, but they charge ten rupees more than last month for the same  │
│    │          │ small cup.                                                                       │
│ 5  │ <label>  │ They take UPI, so I never have to hunt for change during the morning rush.       │
│ 6  │ <label>  │ The bench under the tin roof is a lifesaver when the monsoon rain starts.        │
│ 7  │ <label>  │ My filter coffee arrived lukewarm and the cup was sticky on the outside.         │
│ 8  │ <label>  │ The owner is friendly and remembers my order every single day.                   │
│ 9  │ <label>  │ Bun maska was soft and buttery, though the plate could have been cleaner.        │
│ 10 │ <label>  │ After six in the evening it is so crowded that I cannot even find a place to     │
│    │          │ stand.                                                                           │
╰────┴──────────┴──────────────────────────────────────────────────────────────────────────────────╯
Counts: positive <N>, negative <N>, mixed <N>
Saved outputs/zero_shot_labels.json
```

Before you run it, guess the labels yourself. Reviews 4 and 9 praise one thing and complain about another, so
`mixed` is the honest answer for them. Does Gemini agree with you?

Without a key: the same friendly help as in step 6.

> **Seen an older tutorial?** Some say to "train" a big model on your reviews to get sentiment. For a job like
> this you do not need to: a clear prompt and a structured form do it with zero examples. Older code also parsed the
> reply with `PydanticOutputParser` and format instructions. That still works, but `with_structured_output` is simpler.

> [!WARNING]
> **Common errors**
> - `The model's answer did not fit the form (no label for review(s) [7])` - the model skipped a review. Run it again. `match_labels` checks every answer, so a skipped review never slips through.
> - `429 RESOURCE_EXHAUSTED` - wait a minute. This step makes only one request per run.

## Step 8 - Score your photos with CLIP

**Goal:** for each photo, the right label scores highest.
**New idea:** **contrastive image-text matching**. CLIP was pre-trained on about 400 million picture and caption pairs
from the internet ([CLIP paper](https://arxiv.org/abs/2103.00020)). It learned to pull each picture close to its own
caption and push the wrong captions away. So it can score **any** labels you write, with no training on your photos.
CLIP is a foundation model that generates nothing: it cannot write a caption, it only scores.

Take four photos at home, one each of a cup, a plant, a shoe and a book, and put them in `photos/` as
`cup.jpg`, `plant.jpg`, `shoe.jpg` and `book.jpg` (replace the sample drawings). Only your own things, and no faces of
other people without their consent. The file name is only used to check the answer; CLIP never sees it.

`multimodal_clip.py`:

```python
"""
multimodal_clip.py - CLIP matches your photos with sentences (step 8).

CLIP was pre-trained on a huge number of picture + caption pairs from the internet.
It learned to pull a picture and its own caption close together, and push the wrong
captions away (contrastive learning). So it can score ANY labels you write, with no
training on your photos.

It cannot write a caption. It only scores how well each sentence fits each photo:
another foundation model that generates nothing.

Run:  python multimodal_clip.py                          (the five labels below)
      python multimodal_clip.py "a dog" "a car" "a tree"   (your own labels)
      python multimodal_clip.py --grid                     (look inside: photo x caption similarity)
"""

import os
import sys
import time

# This checkpoint stores its weights as pytorch_model.bin. Without this line, transformers
# also downloads a second copy (about 600 MB, .safetensors format) in the background.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import torch  # noqa: E402  (imported after the line above on purpose)
from PIL import Image, ImageOps  # noqa: E402
from transformers import pipeline  # noqa: E402

from common import console, count_parameters, find_photos, in_millions, make_table  # noqa: E402

MODEL_ID = "openai/clip-vit-base-patch32"
LABELS = ["a cup", "a plant", "a shoe", "a book", "a bicycle"]   # the bicycle matches none of the photos
TEMPLATE = "a photo of {}."                                       # CLIP compares whole sentences


def load_clip():
    """Load CLIP (the first run downloads about 600 MB)."""
    return pipeline("zero-shot-image-classification", model=MODEL_ID, device="cpu")


def open_photo(path) -> Image.Image:
    """Open a photo the right way up (phones store the rotation separately), in RGB colours."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def score_photo(clip, image, labels: list[str]) -> list[tuple[str, float]]:
    """(label, score) pairs, best first. The scores of all labels add up to 1."""
    results = clip(image, candidate_labels=labels, hypothesis_template=TEMPLATE)
    return [(r["label"], r["score"]) for r in results]


def expected_label(path, labels: list[str]) -> str | None:
    """'a cup' for a file called cup.jpg or my_cup_2.png. None if the name matches no label."""
    name = path.stem.lower()
    for label in labels:
        if label.split()[-1] in name:   # the last word of the label: "cup"
            return label
    return None


def similarity_grid(clip, images, captions: list[str]) -> torch.Tensor:
    """Cosine similarity of every photo (rows) with every caption (columns)."""
    with torch.no_grad():
        text = clip.tokenizer(captions, padding=True, return_tensors="pt")
        pixels = clip.image_processor(images, return_tensors="pt")
        out = clip.model(**text, **pixels)
    return out.image_embeds @ out.text_embeds.T   # both are unit-length vectors, so this is the cosine


def show_scores(clip, photos, labels: list[str]) -> None:
    rows, right, checked = [], 0, 0
    for path in photos:
        scores = score_photo(clip, open_photo(path), labels)
        (best, best_score), (second, second_score) = scores[0], scores[1]
        expected = expected_label(path, labels)
        if expected is None:
            verdict = "-"
        else:
            checked += 1
            right += best == expected
            verdict = "yes" if best == expected else f"no, expected {expected}"
        rows.append([path.name, best, f"{best_score:.1%}", f"{second} ({second_score:.1%})", verdict])
    console.print(make_table(["photo", "best label", "score", "runner-up", "matches file name?"], rows,
                             title=f"CLIP scores against: {', '.join(labels)}"))
    if checked:
        print(f"Top label matches the file name for {right} of {checked} photos.")


def show_grid(clip, photos, labels: list[str]) -> None:
    matched = [(p, expected_label(p, labels)) for p in photos]
    matched = [(p, label) for p, label in matched if label]
    if not matched:
        print("--grid needs photos named after a label, like cup.jpg or plant.png.")
        return
    captions = [TEMPLATE.format(label) for _, label in matched]
    grid = similarity_grid(clip, [open_photo(p) for p, _ in matched], captions)
    rows = []
    for (path, _), values in zip(matched, grid.tolist()):
        best = max(values)
        rows.append([path.name] + [f"[{v:.2f}]" if v == best else f"{v:.2f}" for v in values])
    console.print(make_table(["photo"] + captions, rows, title="Cosine similarity, photo x caption ([best] in each row)"))
    print("The right caption wins each row by only a few hundredths. CLIP multiplies these numbers by",
          f"{clip.model.logit_scale.exp().item():.0f} before turning them into percentages, so small gaps become big scores.")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    grid_mode = "--grid" in argv
    labels = [a for a in argv if a != "--grid"] or LABELS

    photos = find_photos()
    if not photos:
        print("No pictures found in photos/. Add a few .jpg or .png files and run again.")
        return 1

    start = time.perf_counter()
    clip = load_clip()
    params = count_parameters(clip.model)
    print(f"CLIP: {MODEL_ID}   Parameters: {params:,} (about {in_millions(params)})")
    print(f"Loaded in {time.perf_counter() - start:.1f} s. Photos: {len(photos)}\n")

    if grid_mode:
        show_grid(clip, photos, labels)
    else:
        show_scores(clip, photos, labels)
        print("CLIP wrote nothing. It only scored how well each label fits each photo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python multimodal_clip.py
```

Expected output with the sample pictures (the first run downloads about 605 MB; your own photos give different scores):

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████| 398/398 [00:00<00:00, 55378.62it/s]
CLIP: openai/clip-vit-base-patch32   Parameters: 151,277,313 (about 151.3 million)
Loaded in 1.3 s. Photos: 4

CLIP scores against: a cup, a plant, a shoe, a book, a bicycle
╭───────────┬────────────┬───────┬───────────────┬────────────────────╮
│ photo     │ best label │ score │ runner-up     │ matches file name? │
├───────────┼────────────┼───────┼───────────────┼────────────────────┤
│ book.jpg  │ a book     │ 97.3% │ a cup (1.4%)  │ yes                │
│ cup.jpg   │ a cup      │ 99.9% │ a book (0.0%) │ yes                │
│ plant.jpg │ a plant    │ 97.6% │ a cup (2.0%)  │ yes                │
│ shoe.jpg  │ a shoe     │ 99.8% │ a cup (0.1%)  │ yes                │
╰───────────┴────────────┴───────┴───────────────┴────────────────────╯
Top label matches the file name for 4 of 4 photos.
CLIP wrote nothing. It only scored how well each label fits each photo.
```

The scores of the five labels always add up to 100%, so **CLIP must choose one of your labels**, even a wrong one.
See what happens when none of the labels fits:

```bash
python multimodal_clip.py "a dog" "a car" "a tree"
```

```text
CLIP scores against: a dog, a car, a tree
╭───────────┬────────────┬───────┬───────────────┬────────────────────╮
│ photo     │ best label │ score │ runner-up     │ matches file name? │
├───────────┼────────────┼───────┼───────────────┼────────────────────┤
│ book.jpg  │ a car      │ 43.3% │ a dog (42.8%) │ -                  │
│ cup.jpg   │ a dog      │ 44.1% │ a car (34.6%) │ -                  │
│ plant.jpg │ a tree     │ 65.7% │ a dog (28.8%) │ -                  │
│ shoe.jpg  │ a dog      │ 55.5% │ a car (34.2%) │ -                  │
╰───────────┴────────────┴───────┴───────────────┴────────────────────╯
CLIP wrote nothing. It only scored how well each label fits each photo.
```

The plant becomes "a tree" and the book becomes "a car" at 43%. Low, close scores mean CLIP is unsure.

**Look inside.** `--grid` compares every photo with every caption, the raw **cosine similarity** before CLIP turns it
into percentages:

```bash
python multimodal_clip.py --grid
```

```text
Cosine similarity, photo x caption ([best] in each row)
╭───────────┬────────────────────┬───────────────────┬─────────────────────┬────────────────────╮
│ photo     │ a photo of a book. │ a photo of a cup. │ a photo of a plant. │ a photo of a shoe. │
├───────────┼────────────────────┼───────────────────┼─────────────────────┼────────────────────┤
│ book.jpg  │ [0.27]             │ 0.22              │ 0.21                │ 0.22               │
│ cup.jpg   │ 0.21               │ [0.29]            │ 0.21                │ 0.20               │
│ plant.jpg │ 0.21               │ 0.25              │ [0.29]              │ 0.24               │
│ shoe.jpg  │ 0.21               │ 0.21              │ 0.20                │ [0.28]             │
╰───────────┴────────────────────┴───────────────────┴─────────────────────┴────────────────────╯
The right caption wins each row by only a few hundredths. CLIP multiplies these numbers by 100 before turning them into percentages, so small gaps become big scores.
```

Each photo's own caption wins its row: that diagonal is what contrastive training builds.

> [!WARNING]
> **Common errors**
> - `No pictures found in photos/` - the photos are in another folder, or they are `.heic` files from an iPhone. Save or export them as JPG.
> - `cannot identify image file` - the file is not really an image (or it is HEIC renamed to .jpg). Open it in an image viewer and export as JPG.
> - `matches file name?` says `-` - the file name contains none of the label words. Name it after what is in it, like `my_cup.jpg`.
> - A wrong top label on your own photo - try a plainer background and more light, or reword the label (`a coffee mug`). That is prompt adaptation again.

## Step 9 - Caption one photo with Gemini

**Goal:** one descriptive sentence about one photo.
**New idea:** a **multimodal input**. One message carries two content blocks, our question (text) and the photo (image).
Gemini reads both and writes a sentence: unlike CLIP, it generates. We shrink the photo first, so a 4 MB phone photo
becomes a few hundred KB.

`caption.py`:

```python
"""
caption.py - Gemini looks at one photo and describes it (step 9).

A multimodal model takes more than one kind of input. Here ONE message carries text
AND a picture, as two content blocks. Gemini then writes a sentence: unlike CLIP, it generates.

We shrink the photo first (longest side 1024 pixels). A 4 MB phone photo becomes a few
hundred KB, which uploads faster. The free tier may use what you send to improve Google's
products, so send only photos you are happy to share (no faces of other people, no documents).

Run:  python caption.py                     (the first photo in photos/)
      python caption.py photos/plant.jpg      (a photo you choose)
"""

import base64
import io
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from PIL import Image, ImageOps

from common import HERE, find_photos
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

QUESTION = "Describe this photo in one sentence for someone who cannot see it."
MAX_SIDE = 1024   # pixels


def image_block(path, max_side: int = MAX_SIDE) -> dict:
    """The photo as a LangChain image content block: the JPEG bytes as base64 text, plus its type."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # right way up, RGB colours
    image.thumbnail((max_side, max_side))                              # keeps the shape, only ever shrinks
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return {"type": "image", "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/jpeg"}


def build_message(path, question: str = QUESTION) -> HumanMessage:
    """One message, two content blocks: our question (text) and the photo (image)."""
    return HumanMessage(content=[{"type": "text", "text": question}, image_block(path)])


def caption(path, llm=None) -> str:
    """Ask the model to describe the photo. Returns its sentence as plain text."""
    if llm is None:
        llm = get_llm()
    return llm.invoke([build_message(path)]).text   # .text, not .content (Gemini 3 returns a list of blocks)


def main(argv=None, llm=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    photos = [Path(argv[0])] if argv else find_photos()[:1]
    if not photos or not photos[0].is_file():
        print("No photo found. Put a .jpg or .png in photos/, or run: python caption.py path/to/photo.jpg")
        return 1
    photo = photos[0]

    try:
        provider = get_provider()
        print(f"Provider: {provider}   Model: {get_model_name(provider)}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")        # friendly help (no key, or a typo in LLM_PROVIDER)
        return 1

    block = image_block(photo)
    sent_kb = len(base64.b64decode(block["base64"])) // 1024
    full = photo.resolve()
    shown = full.relative_to(HERE).as_posix() if full.is_relative_to(HERE) else str(photo)
    print(f"Photo: {shown} (sent as a {sent_kb} KB JPEG)")
    print(f"Question: {QUESTION}")

    start = time.perf_counter()
    sentence = caption(photo, llm)
    print(f"\nCaption ({time.perf_counter() - start:.1f} s):\n{sentence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python caption.py photos/cup.jpg
```

Expected output with your key (the sentence is Gemini's own; this layout comes from a test run with a placeholder reply):

```text
Provider: gemini   Model: gemini-3.5-flash-lite
Photo: photos/cup.jpg (sent as a 20 KB JPEG)
Question: Describe this photo in one sentence for someone who cannot see it.

Caption (<N> s):
<one descriptive sentence written by Gemini>
```

Without a key: the same friendly help as in step 6.

Privacy: on the free tier, Google may use what you send to improve its products. Send photos of things, not of people
or documents.

> [!WARNING]
> **Common errors**
> - `No photo found` - check the path. From inside `ep04-model-zoo` it is `photos/cup.jpg`.
> - `400 ... image` errors - the file is not a normal photo. `image_block` re-saves every photo as JPEG, so open the file in an image viewer first and check it is a picture.
> - The caption describes the photo sideways - your photo viewer rotated it, but the file did not. `ImageOps.exif_transpose` fixes phone rotation; old editing apps sometimes remove that information.

## Step 10 - Fill model_cards.json

**Goal:** the facts about today's four models in one file that passes a schema check.
**New idea:** **reading a model card**. A model card is like the label on a food packet: size, ingredients,
best-before date and who should avoid it. For a model that means parameters, inputs and outputs (modality), context
window, open or closed weights, licence and intended use. Every fact needs a source and the date you checked it,
because models and licences change. Three terms worth keeping apart: **open source** (code, and sometimes training
data, published under an open licence), **open weights** (you can download the learned numbers, under the licence's
terms) and **closed** (you can only call it through an API). And Hugging Face is a hub that hosts models, not a model.

`model_cards.json` (one of the four cards; open the file to see all of them):

```json
{
  "name": "all-MiniLM-L6-v2",
  "id": "sentence-transformers/all-MiniLM-L6-v2",
  "maker": "sentence-transformers project (Hugging Face community)",
  "family": "encoder",
  "generates": false,
  "inputs": ["text"],
  "outputs": ["vector of 384 numbers"],
  "parameters": 22713216,
  "context_window": 256,
  "weights": "open",
  "licence": "Apache-2.0",
  "business_use": "yes",
  "runs_on": "your laptop",
  "download_mb": 91,
  "released": "2021",
  "source": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
  "checked": "2026-10-23",
  "notes": "Parameters counted by encoder.py. The card says text longer than 256 word pieces is cut off. Trained during a 2021 Hugging Face community week on over 1 billion sentence pairs."
}
```

`model_cards_schema.py`:

```python
"""
model_cards_schema.py - check that model_cards.json is complete and believable (step 10).

A model card is like the label on a food packet: size, ingredients, best-before date,
and who should not use it. Every fact needs a source and the date you checked it,
because model facts go stale fast.

Run:  python model_cards_schema.py                  (checks model_cards.json)
      python model_cards_schema.py other_cards.json   (checks another file)
"""

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from common import HERE, in_millions

CARDS_FILE = HERE / "model_cards.json"
STALE_AFTER_DAYS = 180   # re-check a card that is older than about six months

Family = Literal["encoder", "decoder", "encoder-decoder", "contrastive image-text",
                 "multimodal", "diffusion", "speech-to-text"]


class ModelCard(BaseModel):
    """The facts we keep about one model. Field(...) means: you must fill it in (null is allowed where shown)."""

    name: str = Field(min_length=2)
    id: str = Field(min_length=2, description="what you type in code, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    maker: str = Field(min_length=2)
    family: Family
    generates: bool = Field(description="does it write new text, images or sound?")
    inputs: list[str] = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)
    parameters: int | None = Field(..., gt=0, description="null when the maker does not publish it")
    context_window: int | None = Field(..., gt=0, description="how many tokens it reads at once")
    weights: Literal["open", "closed"]
    licence: str = Field(min_length=2)
    business_use: Literal["yes", "check the terms", "research only"]
    runs_on: str = Field(min_length=2)
    download_mb: int | None = Field(..., gt=0)
    released: str = Field(pattern=r"^\d{4}(-\d{2})?$", description="2024 or 2024-09")
    source: str
    checked: date
    notes: str = ""

    @model_validator(mode="after")
    def facts_fit_together(self):
        if not self.source.startswith("https://"):
            raise ValueError("source must be a web link that starts with https://")
        if self.checked > date.today():
            raise ValueError(f"checked date {self.checked} is in the future")
        if self.weights == "closed" and self.download_mb is not None:
            raise ValueError("closed weights cannot be downloaded, so download_mb must be null")
        if self.weights == "open" and self.download_mb is None:
            raise ValueError("open weights are a file you download: give download_mb")
        return self


def explain(error: ValidationError) -> list[str]:
    """Turn pydantic's error list into short lines like 'weights: Input should be 'open' or 'closed''."""
    lines = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "card"
        message = re.sub(r"^Value error, ", "", item["msg"])
        lines.append(f"{where}: {message}")
    return lines


def check_cards(path=CARDS_FILE) -> tuple[list[ModelCard], list[str]]:
    """Read and check every card. Returns (good cards, problems). No problems = the file is fine."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [], [f"{Path(path).name} is not valid JSON: {error}"]
    if not isinstance(raw, list):
        return [], ["the file must hold a list of cards: [ {...}, {...} ]"]

    cards, problems, seen = [], [], set()
    for number, item in enumerate(raw, start=1):
        label = item.get("name", f"card {number}") if isinstance(item, dict) else f"card {number}"
        try:
            card = ModelCard.model_validate(item)
        except ValidationError as error:
            problems += [f"{label}: {line}" for line in explain(error)]
            continue
        if card.id in seen:
            problems.append(f"{label}: the id {card.id} appears twice")
        seen.add(card.id)
        cards.append(card)
    return cards, problems


def load_cards(path=CARDS_FILE) -> list[ModelCard]:
    """The checked cards. Stops with a clear message when the file has a problem."""
    cards, problems = check_cards(path)
    if problems:
        raise ValueError("model_cards.json has problems:\n  " + "\n  ".join(problems))
    return cards


def is_stale(card: ModelCard, today: date | None = None) -> bool:
    """True when the card was checked more than STALE_AFTER_DAYS ago."""
    return ((today or date.today()) - card.checked).days > STALE_AFTER_DAYS


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else CARDS_FILE
    cards, problems = check_cards(path)

    for card in cards:
        size = in_millions(card.parameters) + " parameters" if card.parameters else "size not published"
        stale = "  <- older than six months, check it again" if is_stale(card) else ""
        print(f"OK   {card.name:<24} {card.family:<23} {card.weights:<7} {size:<25} checked {card.checked}{stale}")
    for problem in problems:
        print(f"FIX  {problem}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) in {path.name}. Fix them and run again.")
        return 1
    print(f"{len(cards)} cards in {path.name}, all passed the schema check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python model_cards_schema.py
```

Expected output:

```text
OK   all-MiniLM-L6-v2         encoder                 open    22.7 million parameters   checked 2026-10-23
OK   Qwen2.5-0.5B-Instruct    decoder                 open    494.0 million parameters  checked 2026-10-23
OK   Gemini 3.5 Flash-Lite    multimodal              closed  size not published        checked 2026-10-23
OK   CLIP ViT-B/32            contrastive image-text  open    151.3 million parameters  checked 2026-10-23

4 cards in model_cards.json, all passed the schema check.
```

Now break it on purpose. Make a copy called `broken_cards.json`:

**Windows (PowerShell):**

```powershell
Copy-Item model_cards.json broken_cards.json
```

**macOS / Linux:**

```bash
cp model_cards.json broken_cards.json
```

Open the copy in VS Code. Change Qwen's `"weights": "open"` to `"free"`, change Gemini's `"download_mb": null` to
`500`, and delete CLIP's `"source"` line. Save it, then check the copy:

```bash
python model_cards_schema.py broken_cards.json
```

```text
OK   all-MiniLM-L6-v2         encoder                 open    22.7 million parameters   checked 2026-10-23
FIX  Qwen2.5-0.5B-Instruct: weights: Input should be 'open' or 'closed'
FIX  Gemini 3.5 Flash-Lite: card: closed weights cannot be downloaded, so download_mb must be null
FIX  CLIP ViT-B/32: source: Field required

3 problem(s) in broken_cards.json. Fix them and run again.
```

Each broken fact is caught, with the card and the field that is wrong. Delete `broken_cards.json` when you are done
(in VS Code: right-click it, **Delete**), so you never mistake it for the real file.

> [!WARNING]
> **Common errors**
> - `... is not valid JSON` - usually a comma after the last item in a list or card. JSON allows no trailing commas.
> - `checked: Input should be a valid date` - write dates as `2026-10-23` (year-month-day).
> - `released: String should match pattern` - write `2024` or `2024-09`.

## Step 11 - Print the zoo report

**Goal:** a five-row table: which model for which job.
**New idea:** **choosing by job, not by hype**. Pick the smallest model that does the job well, where it may run
(your laptop or a server), and a licence that allows what you plan to do.

`zoo_report.py`:

```python
"""
zoo_report.py - which model for which job? (step 11)

Reads model_cards.json (checked by model_cards_schema.py) and prints one row per job
we did today. Choose by the job, not by the hype: the smallest model that does the
job well, with a licence that allows what you plan to do.

Run:  python zoo_report.py
"""

from common import console, make_table
from model_cards_schema import ModelCard, load_cards

# (the job, the id of the model we used for it, the script that did it today)
JOBS = [
    ("Find similar reviews", "sentence-transformers/all-MiniLM-L6-v2", "encoder.py"),
    ("Write text offline", "Qwen/Qwen2.5-0.5B-Instruct", "decoder_local.py"),
    ("Write fluent text, label reviews", "gemini-3.5-flash-lite", "decoder_cloud.py, zero_shot.py"),
    ("Label photos with your words", "openai/clip-vit-base-patch32", "multimodal_clip.py"),
    ("Caption a photo", "gemini-3.5-flash-lite", "caption.py"),
]

COLUMNS = ["job", "model", "family", "writes?", "runs on", "business use"]


def report_rows(cards: list[ModelCard], jobs=JOBS) -> list[list[str]]:
    """One row per job, filled in from the matching model card."""
    by_id = {card.id: card for card in cards}
    rows = []
    for job, model_id, script in jobs:
        if model_id not in by_id:
            raise KeyError(f"no card for {model_id} in model_cards.json (job: {job})")
        card = by_id[model_id]
        rows.append([job, card.name, card.family, "yes" if card.generates else "no",
                     card.runs_on, card.business_use])
    return rows


def main() -> int:
    try:
        cards = load_cards()
        rows = report_rows(cards)
    except (ValueError, KeyError) as error:
        print(str(error).strip("'\""))
        print("Run: python model_cards_schema.py")
        return 1

    console.print(make_table(COLUMNS, rows, title="Which model for which job", keep_whole=("model", "writes?")))
    oldest = min(card.checked for card in cards)
    print(f"{len(rows)} jobs, {len(cards)} model cards. Facts checked on {oldest}.")
    print("Licences and model versions change: read the card again before you build a product on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
python zoo_report.py
```

Expected output:

```text
Which model for which job
╭───────────────┬───────────────────────┬───────────────┬─────────┬───────────────┬────────────────╮
│ job           │ model                 │ family        │ writes? │ runs on       │ business use   │
├───────────────┼───────────────────────┼───────────────┼─────────┼───────────────┼────────────────┤
│ Find similar  │ all-MiniLM-L6-v2      │ encoder       │ no      │ your laptop   │ yes            │
│ reviews       │                       │               │         │               │                │
│ Write text    │ Qwen2.5-0.5B-Instruct │ decoder       │ yes     │ your laptop   │ yes            │
│ offline       │                       │               │         │               │                │
│ Write fluent  │ Gemini 3.5 Flash-Lite │ multimodal    │ yes     │ Google's      │ check the      │
│ text, label   │                       │               │         │ servers       │ terms          │
│ reviews       │                       │               │         │               │                │
│ Label photos  │ CLIP ViT-B/32         │ contrastive   │ no      │ your laptop   │ research only  │
│ with your     │                       │ image-text    │         │               │                │
│ words         │                       │               │         │               │                │
│ Caption a     │ Gemini 3.5 Flash-Lite │ multimodal    │ yes     │ Google's      │ check the      │
│ photo         │                       │               │         │ servers       │ terms          │
╰───────────────┴───────────────────────┴───────────────┴─────────┴───────────────┴────────────────╯
5 jobs, 4 model cards. Facts checked on 2026-10-23.
Licences and model versions change: read the card again before you build a product on it.
```

Read the `writes?` column: two of our four foundation models write nothing at all. And read `business use`:
CLIP is great for learning, but its card rules out deployed use.

> [!WARNING]
> **Common errors**
> - `no card for ... in model_cards.json` - a job in `JOBS` names a model id that has no card. Copy the id exactly from the card.
> - `model_cards.json has problems` - run `python model_cards_schema.py` to see which card and field.

## Step 12 - Optional: switch to a local runtime

**Goal:** the same `decoder_cloud.py` answers from a model on your own computer.
**New idea:** `LLM_PROVIDER=ollama`. [Ollama](https://ollama.com) is a free app that runs open-weights models
locally. `get_llm()` in `llm.py` switches with one line of `.env`; the script does not change.

1. Install Ollama from https://ollama.com and start it.
2. Download the course's local model (a download of a few GB): `ollama pull qwen3.5:4b`
3. In the root `.env`, set `LLM_PROVIDER=ollama`.
4. Run `python decoder_cloud.py` again.

Expected output with Ollama running (the reply and time are your model's; layout from a test run with a placeholder reply):

```text
Provider: ollama   Model: qwen3.5:4b

Prompt: A customer wrote this review: "My filter coffee arrived lukewarm and the cup was sticky on the outside." Write a two-sentence reply that says sorry and promises one fix.

Reply (<N> s):
<a two-sentence reply written by qwen3.5:4b on your computer>

Same prompt, two models
╭─────────────────────────┬───────────────────────────────────┬────────────────────────────────────╮
│                         │ small local model (step 5)        │ ollama model                       │
├─────────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
│ Model                   │ Qwen/Qwen2.5-0.5B-Instruct        │ qwen3.5:4b                         │
│ Runs on                 │ this computer                     │ this computer (Ollama)             │
│ Weights                 │ open (a file on your disk)        │ open (a file on your disk)         │
│ Parameters              │ 494,032,768                       │ see ollama.com/library             │
│ Needs                   │ nothing after the download        │ the Ollama app, no key             │
│ Time for this reply     │ 0.8 s                             │ <N> s                              │
│ Names the real problem? │ no                                │ <yes or no>                        │
│ Reply                   │ I'm really sorry for any          │ <a two-sentence reply written by   │
│                         │ inconvenience caused by the delay │ qwen3.5:4b on your computer>       │
│                         │ in your order. Please let me know │                                    │
│                         │ if you need anything else or have │                                    │
│                         │ any other questions.              │                                    │
╰─────────────────────────┴───────────────────────────────────┴────────────────────────────────────╯
Which reply would you put on the shop's page? That judgement is yours.
```

If Ollama is not running, you get this instead of a crash (real output, recorded on a computer without Ollama):

```text
Provider: ollama   Model: qwen3.5:4b

Prompt: A customer wrote this review: "My filter coffee arrived lukewarm and the cup was sticky on the outside." Write a two-sentence reply that says sorry and promises one fix.

Could not reach the model: [Errno 61] Connection refused
Start the Ollama app (https://ollama.com), run 'ollama pull qwen3.5:4b', then try again.
```

Set `LLM_PROVIDER=gemini` again when you are done.

> [!WARNING]
> **Common errors**
> - `Could not reach the model: [Errno 61] Connection refused` (Windows: `[WinError 10061]`) - the Ollama app is not running. Start it and try again.
> - `model "qwen3.5:4b" not found` - run `ollama pull qwen3.5:4b` first.
> - Very slow replies - a 4-billion-parameter model on a laptop CPU takes a while. Run `ollama pull qwen3.5:2b`, then set `LLM_MODEL=qwen3.5:2b` in the root `.env`.

## Step 13 - Run the offline tests

**Goal:** prove every part works with no key and no internet.
**New idea:** testing without keys. The Gemini steps run with a **fake model** from `testing/fakes.py` that gives scripted
answers; the local models run for real if you have downloaded them, and are skipped if you have not.
The tests check the code, not your data: they use their own copy of the ten sample reviews (`tests/sample_reviews.txt`)
and draw the four sample pictures in a temporary folder. So your own reviews and photos never make a test fail.

```bash
python -m pytest
```

Expected output (after you have run steps 3, 5 and 8 once):

```text
....................................s.......................                                 [100%]
59 passed, 1 skipped in 7.93s
```

An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes.
On a computer that has not downloaded the three local models yet, three more are skipped (real output with an empty model cache):

```text
....................................s....sss................                                 [100%]
56 passed, 4 skipped in 1.83s
```

How a test checks the zero-shot step without Gemini (from `tests/test_zero_shot.py`). `labels(...)` fills in the
`ReviewLabels` form, and `structured_reply(...)` makes the fake model hand that form back, exactly as
`with_structured_output` would. The test scrambles the order on purpose:

```python
def test_labels_come_back_in_review_order():
    llm = fake_llm(structured_reply(labels((3, "mixed"), (1, "positive"), (2, "negative"))))
    assert label_reviews(REVIEWS, llm) == ["positive", "negative", "mixed"]
```

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'encoder'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called a real model. Pass `llm=fake_llm(...)` to your function.
> - `OSError: We couldn't connect to 'https://huggingface.co' ...` - a test tried to load a model that is not downloaded. Run its script once with internet, then test again.

## Final challenge

**Add an ear to the zoo.** Record a ten-second voice note, transcribe it on your laptop with the tiny Whisper-style model
`openai/whisper-tiny`, add its card to `model_cards.json` and a sixth job to the report. Full instructions:
[CHALLENGE.md](CHALLENGE.md). Post your transcript line in the comments.

<details>
<summary>Hint</summary>

`pipeline("automatic-speech-recognition", model="openai/whisper-tiny")` takes a dictionary
`{"raw": samples, "sampling_rate": 16000}`. Save your recording as a 16000 Hz mono WAV file and read it with Python's `wave` module.

</details>

## What you learned

- A foundation model is pre-trained once on huge, mostly unlabelled data with self-supervision, then adapted to many jobs.
- Parameters are the learned numbers (22.7 million in MiniLM, 494 million in Qwen2.5-0.5B, 151 million in CLIP); tokens are the text pieces that flow through them. Using a model does not change its weights.
- Encoders read and return vectors, decoders write one token at a time, CLIP scores pictures against sentences, and a multimodal model like Gemini takes text and images together.
- A foundation model does not have to generate: MiniLM and CLIP write nothing.
- Open weights run on your laptop; closed models run behind an API. The same prompt shows the size gap.
- Start adapting with a prompt, then RAG, and fine-tune only when you must.
- A model card tells you size, modality, context window, licence and intended use, and every fact needs a source and a checked date.

## Next episode

**Episode 05: How to Choose an LLM in 2026: Benchmarks, Your Own Eval and AI Safety Basics** - today you picked models
by job. Next time you measure them: benchmarks, a small test set of your own, and the safety basics. -> `ep05-model-scorecard` (coming soon)
