# Episode 16: Hugging Face for Beginners (2026): Run AI Models on Your Own Laptop

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep16-local-ai-toolbox

## What you will build

**Tiffin Toolbox**, a command-line toolbox that runs on your own laptop with three small Hugging Face models
(Tiffin Toolbox is a name we invented for this course: like a tiffin, everything is packed at home).
It summarises your notes, tries to translate them into Hindi (and shows you how to catch a bad translation),
answers questions about your photo and finds notes by meaning. It prints a licence report for every model before
you build on it, compares the summary, translation and photo answers with Gemini, and keeps working with the Wi-Fi
switched off.

```text
$ python app.py search "train ticket"
Query: "train ticket"

1. +0.310  jaipur_trip      Jaipur trip for Meera didi's wedding. Book two sleeper be...
2. +0.132  amma_birthday    Amma's birthday is on Sunday. Order a small eggless choco...
3. +0.117  groceries        Groceries for this week: 5 kg atta, 1 kg toor dal, onions...

$ python app.py ask-photo
Photo: desk.jpg (960x640)   Model: HuggingFaceTB/SmolVLM-256M-Instruct on cpu (ready in 2.6 s)

Q: What objects are on the desk?
A: There is a laptop, a pencil, a cup, and a pencil sharpener on the desk.   (3.2 s)

Q: What colour is the mug?
A: The mug is red.   (2.8 s)
```

(Real output from the local models, recorded with all network access blocked. Your seconds will differ. Yes, there is
no pencil sharpener on that desk: step 10 is about exactly that.)

**You will learn:**

- How the Hugging Face Hub is organised (models, datasets, Spaces), how to read a model card, and why
  "free to download" does not always mean "free for your business".
- How to run models locally with **transformers v5**: download once, cache, `dtype`, `device`, and offline mode.
- Why old tutorials crash on v5, and the 2026 replacements: a small chat model through `text-generation` and a
  vision-language model through `image-text-to-text`.
- Extractive versus abstractive summaries, why length limits count **tokens**, and how to judge a translation you cannot read.
- Sentence embeddings, a full cosine-similarity matrix (from -1 to 1) and a tiny semantic search over your own notes.
- When local wins and when the cloud wins: speed, quality, privacy, cost and offline, measured on your own machine.

> **Seen an older tutorial?** It used `pipeline("summarization", model="facebook/bart-large-cnn")`,
> `pipeline("translation", model="facebook/nllb-200-distilled-600M", torch_dtype=...)` and
> `BlipForQuestionAnswering` for questions about a picture. **transformers v5 removed the summarization,
> translation and visual-question-answering pipelines**, so those `pipeline(...)` lines now stop with a `KeyError`
> (step 3). The BLIP classes still load, but BLIP is from 2022 and has no pipeline task any more (step 10).
> In 2026 one small chat model does summaries and translations through `pipeline("text-generation", ...)`,
> a small vision-language model answers questions about photos through `pipeline("image-text-to-text", ...)`,
> and `torch_dtype=` is now `dtype=`. Also: the NLLB translation model is licensed **CC-BY-NC-4.0**, which means
> non-commercial use only (step 2).

## Prerequisites

- Finished Episode 02 (`ep02-attention-xray`, tokens versus embeddings), Episode 04 (`ep04-model-zoo`, model cards
  and licences) and Episode 07 (`ep07-ask-the-handbook`, embeddings for search). Or know: a chat model reads token IDs,
  an embedding is one list of numbers for a whole sentence, and every model has a licence.
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- The shared `.venv` and root `.env` with your **free** Gemini key from Episode 1 ([SETUP.md](../SETUP.md)).
  Every local step works without the key; only the comparison steps (5, 6, 11) need it.
- About 45 minutes, and about **1.6 GB of free disk** for the three models (step 2 prints the exact sizes).
- On the free tier, Google may use what you send to improve its products. The comparison steps send your note or
  photo to Gemini, so use only notes and photos you are happy to share. The local steps never send anything.

### The three models (all downloaded once, then used offline)

| Model | Job | Licence | Weights file |
|---|---|---|---|
| [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | summarise, translate | Apache-2.0 | 988 MB |
| [`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | questions about a photo | Apache-2.0 | 513 MB |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | search notes | Apache-2.0 | 91 MB |

Apache-2.0 allows personal and commercial use; keep the licence notice if you share the model files.
Sizes and licences were read from the Hugging Face Hub on 2026-10-26. If you did Episode 04, MiniLM and Qwen are
already on your computer (MiniLM also after Episode 01 or 07). SmolVLM is the only new download for everyone.

## What is in this folder

```text
ep16-local-ai-toolbox/
├── my_notes/                 8 short sample notes (replace them with your own)
├── photos/desk.jpg           a drawn sample desk (made by tools/make_desk_photo.py, CC0); add your own photo here
├── common.py                 shared helpers: paths, model names, stopwatch, memory reading
├── models.py                 Step 2: licence and size report (saves model_cards.json)
├── model_cards.json          the saved cards, so step 2 also works offline
├── old_tasks.py              Step 3: the task names transformers v5 removed
├── skills/
│   ├── chat_model.py         the local chat model shared by summarise and translate
│   ├── summarise.py          Step 4: highlighter versus writer
│   ├── translate.py          Step 6: into Hindi and back
│   ├── search.py             Steps 7-9: embeddings, similarity matrix, semantic search
│   ├── ask_photo.py          Steps 10-11: questions about a photo
│   └── lazy.py               Step 13: load a model on first use, release it later
├── compare.py                Steps 5, 6, 11, 14: local versus Gemini, side by side
├── app.py                    Step 12: one command, four skills, and a menu
├── memory_check.py           Step 13: memory before and after releasing a model
├── tools/make_desk_photo.py  how the sample photo was drawn (you do not need to run it)
├── llm.py, testing/          shared helpers (synced from _template)
├── tests/                    offline tests (fake Gemini, real local models when downloaded)
├── requirements.txt          ../requirements-base.txt + psutil + a huggingface_hub pin
├── CHALLENGE.md
└── outputs/                  made by compare.py and ask_photo.py (git-ignored)
```

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
cd ep16-local-ai-toolbox
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep16-local-ai-toolbox
```

Expected: your prompt starts with `(.venv)` and ends in `ep16-local-ai-toolbox`.

> [!WARNING]
> **Common errors**
> - Windows: `The term '.\.venv\Scripts\Activate.ps1' is not recognized` / macOS, Linux: `no such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

### 0.2 Install the one extra library

This episode uses the shared `requirements-base.txt` plus **psutil**, a small library that reads how much memory a
program uses (step 13). It also pins `huggingface_hub==1.33.0`, the version every step here was tested with (the base
file leaves it open, and newer releases are untested). From this folder:

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
> - `ModuleNotFoundError: No module named 'psutil'` when you run step 13 - you skipped this install. Run it now.

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

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama`. Nothing else changes.

> [!WARNING]
> **Common errors**
> - `No API key found for LLM_PROVIDER=gemini` - the `.env` file is missing, sits in the wrong folder (it belongs in `ai-engineering-course/`), is named `.env.txt`, or still says `paste-your-key-here`. You can still do every local step; the comparison steps print the same help.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute (per-minute limit) or until the daily reset. Your limits: https://aistudio.google.com/rate-limit

---

## Step 1 - Check transformers

**transformers** is the Hugging Face library that downloads and runs models. Version 5 changed a lot, so first make
sure the shared `.venv` has the pinned version 5.

```bash
python -c "import transformers; print(transformers.__version__)"
```

Expected output:

```text
5.20.2
```

Which hardware can PyTorch (the maths engine under transformers) use?

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), torch.cuda.is_available())"
```

On an Apple Silicon Mac (the recorded run):

```text
2.14.1 True False
```

On Windows or Linux with the CPU-only PyTorch from SETUP.md, the first value reads `2.14.1+cpu` and both checks say
`False`. `True False` on the Mac means an Apple GPU (MPS) is there and no NVIDIA GPU (CUDA). Tiffin Toolbox still runs on the **CPU** by
default: it works on every laptop and gives the same answer every run. To try the GPU, add `LOCAL_DEVICE=mps` (or
`LOCAL_DEVICE=cuda`) to the root `.env`.

> [!WARNING]
> **Common errors**
> - It prints `4.x` - an old environment. From this folder: `pip install -r requirements.txt`. Do not "just upgrade" one library by hand: the course pins versions that were tested together.
> - On a Mac, a line like `W... NOTE: Redirects are currently not supported in MacOs.` appears whenever PyTorch starts. It is harmless; Windows and Linux never show it. (We leave it out of the outputs on this page.)

## Step 2 - Read the model card before you build

The Hub is a giant public library. Every book (model) has a label on its spine, the **model card**: what it does, how
big it is, which languages, and who may use it for what. Models live in **model repos**, datasets in **dataset repos**,
and **Spaces** host small demo apps. Before you build on a model, read three things on its card: the task, the size and
the **licence**.

`models.py` asks the Hub for the cards of our three models and of five models you meet in older tutorials, prints a
report, and saves everything in `model_cards.json`. In Episode 04 we typed cards by hand; here `huggingface_hub` reads them:

```python
from huggingface_hub import HfApi

info = HfApi().model_info(model_id, files_metadata=True)   # public models: no token needed
card = info.card_data or {}
licence = card.get("license")                              # e.g. "apache-2.0"
parameters = info.safetensors.total if info.safetensors else None
gated = bool(info.gated)                                   # must you accept terms (and log in) first?
```

```bash
python models.py
```

Expected output (the "on disk" column shows what is already in YOUR cache folder, so yours may show `-`):

```text
From the Hugging Face Hub (saved to model_cards.json)

Models this project downloads:
  model                     licence        business use             weights   params  gated  on disk
  Qwen2.5-0.5B-Instruct     apache-2.0     yes (keep the notice)     988 MB  494.0 M  no      1.0 GB
      task text-generation | languages en | updated 2024-09-25 | used for: summarise + translate
  SmolVLM-256M-Instruct     apache-2.0     yes (keep the notice)     513 MB  256.5 M  no      518 MB
      task image-text-to-text | languages en | updated 2025-04-08 | used for: ask about a photo
  all-MiniLM-L6-v2          apache-2.0     yes (keep the notice)      91 MB   22.7 M  no       92 MB
      task sentence-similarity | languages en | updated 2026-06-01 | used for: search notes

Models from older tutorials, or next steps (we only read their cards):
  model                     licence        business use             weights   params  gated  on disk
  bart-large-cnn            mit            yes (keep the notice)     1.6 GB  406.3 M  no           -
      task summarization | languages en | updated 2024-02-13 | used for: old summarization pipeline
  nllb-200-distilled-600M   cc-by-nc-4.0   NO: non-commercial        2.5 GB        -  no           -
      task translation | languages 196 listed | updated 2024-02-14 | used for: old translation pipeline
  blip-vqa-base             bsd-3-clause   yes (keep the notice)     1.5 GB  384.7 M  no           -
      task visual-question-answering | languages - | updated 2025-02-03 | used for: old visual Q&A
  Qwen2.5-3B-Instruct       other          read it (qwen-research)   6.2 GB    3.1 B  no           -
      task text-generation | languages en | updated 2024-09-25 | used for: a bigger sibling of our chat model
  FLUX.1-dev                other          NO: non-commercial       24.1 GB   11.9 B  YES          -
      task text-to-image | languages en | updated 2025-06-27 | used for: text to image

Cards checked on 2026-10-26. Licences change: read the card again before a product.
Gated = accept the terms on the model page, then log in with a read-only HF_TOKEN in .env.
```

Three lessons hide in that table:

1. **Open is not the same as free for business.** NLLB (the translation model older tutorials use) is non-commercial.
   So is FLUX.1-dev, which is also **gated**.
2. **The same family can change licence with size.** Our 0.5B Qwen is Apache-2.0; its 3B sibling has its own
   "qwen-research" licence. Read the card of the exact model you download.
3. **Check the languages.** The card's metadata says `en` for all three of our models. The Qwen card text adds more
   languages (step 6 tests that claim with Hindi).

No internet? `python models.py --saved` prints the same report from `model_cards.json`.

### Do you need a Hugging Face token?

Not for this episode: all three models are public and not gated. You need a token only for **gated** models (accept
the terms on the model page first), **private** repos, or higher download limits. If you make one:

1. Sign in at https://huggingface.co, open **Settings > Access Tokens** (https://huggingface.co/settings/tokens) and click **New token**.
2. Choose the **Read** role (the other roles are **Write** and **Fine-grained**). Reading is all we ever do.
3. Put it in the root `.env` as `HF_TOKEN=...` (the Hugging Face libraries read that name by themselves).
   Never paste a token into code, a screenshot or a video. If one leaks, delete it on the same page.

> [!WARNING]
> **Common errors**
> - `Could not reach the Hub (...). Showing the saved model_cards.json` - no internet, or `HF_HUB_OFFLINE=1` is set. That is fine: you see the saved report.
> - `Warning: You are sending unauthenticated requests to the HF Hub` - harmless. A free read token in `.env` silences it.
> - A licence called `other` - not an error. It means "read the licence file in the repo": the report shows its name.

## Step 3 - Try an old task name

Here is why code from older tutorials crashes. `old_tasks.py` calls each removed task name inside `try/except`:

```python
def try_old_task(task):
    from transformers import pipeline
    try:
        pipeline(task)                 # no model= on purpose: the task name is checked first, nothing downloads
    except KeyError as error:
        return False, str(error.args[0])
    return True, "still works in this version"
```

```bash
python old_tasks.py
```

Expected output:

```text
transformers 5.20.2

old task name               in this version       use instead
summarization               removed (KeyError)    text-generation + a small chat model
translation                 removed (KeyError)    text-generation + a small chat model
visual-question-answering   removed (KeyError)    image-text-to-text + a vision-language model

The library's own message for "summarization":
  The task summarization has been removed from transformers. Use the `text-generation` pipeline with a chat model instead. See the migration guide for more details: https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md#pipelines

Also renamed in v5: torch_dtype=... is now dtype=... (the old name still works, with a warning).

This version knows 27 task names, including text-generation and image-text-to-text.
```

`python old_tasks.py --list` prints all 27 names. The fix for an old tutorial is never "just upgrade until the error
goes away": pin the versions in a virtual environment and move the code to the new task names.

> [!WARNING]
> **Common errors**
> - It says `works` instead of `removed` - your transformers is older than v5 (step 1).
> - Do not try `pipeline("image-to-text")` without a model to "see what happens": v5 prints a one-line warning, renames it to `image-text-to-text` and picks a default model with about 2 billion parameters (`Qwen/Qwen3-VL-2B-Instruct`) to download.

## Step 4 - Summarise a note locally

Two kinds of summary:

- **Extractive** (a highlighter): copy the one sentence of the note that best matches the whole note. It never invents
  a fact, but it can miss most of the note.
- **Abstractive** (a friend retelling the story): a chat model writes a new sentence in its own words. It reads better,
  but a small model can get a detail wrong.

The writer is a small instruction-tuned chat model through the `text-generation` pipeline. A system message says
which job to do (`skills/chat_model.py` and `skills/summarise.py`):

```python
from transformers import pipeline

chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
                dtype="auto",          # the dtype stored in the model files (bfloat16). Old name: torch_dtype
                device="cpu")          # or "mps" / "cuda"

SYSTEM = ("You summarise personal notes. Reply with ONE short sentence (at most 25 words). "
          "Use only facts from the note.")
messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Summarise this note:\n\n{text}"}]
output = chat(messages, max_new_tokens=60, do_sample=False)   # 60 TOKENS, not words; greedy = same answer every run
summary = output[0]["generated_text"][-1]["content"]          # the last message is the model's reply
```

What is `dtype`? It is the number format of the model's weights. Qwen's files store each of its 494 million numbers
as **bfloat16**, which takes 2 bytes, instead of the 4 bytes of an ordinary `float32` number. That is why the weights
file is 988 MB (494 million x 2 bytes). `dtype="auto"` keeps the stored format, so the model needs half the memory it
would need as `float32`.

> **Classic vs 2026:** older tutorials summarised with `pipeline("summarization", model="facebook/bart-large-cnn")`
> and `max_length=` / `min_length=`. That task is gone in v5. Now a chat model writes the summary through
> `text-generation`, a system message describes the job, and `max_new_tokens=` caps the reply (in tokens).

The highlighter reuses the embeddings model of step 7: it picks the sentence whose vector is closest to the vector of
the whole note.

```bash
python -m skills.summarise
```

(`python -m skills.summarise` runs `skills/summarise.py` as part of this folder, so it can import `common.py`.)

Expected output:

```text
Note: my_notes/jaipur_trip.txt (67 words, 89 tokens for the chat model)

Highlighter (extractive: copies the best sentence)
  "Jaipur trip for Meera didi's wedding."

Writer (abstractive: Qwen/Qwen2.5-0.5B-Instruct on cpu)
  "Meera's wedding was planned for Thursday, 12th, and booked two sleeper berths on the overnight train from Mumbai Central."
  model ready in 2.5 s | read 134 tokens, wrote 28 (limit 60) in 1.4 s
```

Read both against the note. The highlighter picked the title line: true, but it says nothing about the train. The
writer kept the berths and the station, but the note never says the wedding is on Thursday; the **train** leaves on
Thursday. The same answer comes back on every run (`do_sample=False`); only the seconds change.

Now a note where the small writer goes wrong:

```bash
python -m skills.summarise my_notes/laptop_repair.txt
```

```text
Note: my_notes/laptop_repair.txt (54 words, 59 tokens for the chat model)

Highlighter (extractive: copies the best sentence)
  "The laptop battery now lasts barely an hour and the charger gets very hot."

Writer (abstractive: Qwen/Qwen2.5-0.5B-Instruct on cpu)
  "The laptop battery is running low, while the charger is overheating. A new battery costs $100, but it will take two weeks to arrive. Back up the project folder first, then change the Wi-Fi passwords on the college network."
  model ready in 2.3 s | read 104 tokens, wrote 50 (limit 60) in 1.2 s
```

The note names no amount (only that a price was quoted) and says **two working days**. The writer invented "$100" and "two weeks", wrote three
sentences instead of one, and turned "remove the saved password" into "change the passwords". Small local models are
fast and private, but always check their summaries.

**The first run is slower:** the model is downloaded once into your Hugging Face cache folder (`~/.cache/huggingface/hub`
on macOS and Linux, `C:\Users\<you>\.cache\huggingface\hub` on Windows). In our test with an empty cache, the first run
printed `First run: downloading Qwen/Qwen2.5-0.5B-Instruct (about 1 GB, only once)...` and was ready after 142.7 s;
the next run was ready in 3.9 s. Your download time depends on your internet.

> [!WARNING]
> **Common errors**
> - `Qwen/Qwen2.5-0.5B-Instruct is not on this computer yet, and the Hugging Face Hub cannot be reached.` - the first run needs internet (about 1 GB). Connect, remove `HF_HUB_OFFLINE=1` if you set it, and run again.
> - Lines like `Loading weights: 100%|...| 290/290` - normal progress bars while the model loads (we leave them out of this page).
> - `Could not reach the Hub (... offline mode is enabled ...). Using cached commit hash for 'sentence-transformers/all-MiniLM-L6-v2'.` - printed when `HF_HUB_OFFLINE=1` is set. Harmless: it uses the copy on your disk.
> - Your summary is cut off - it hit `max_new_tokens`. The limit counts tokens, and Hindi or long words use more of them.

## Step 5 - The same note through Gemini

`compare.py` gives the cloud model the **same** system message and note, times both, and prints them side by side.
It also logs every answer to `outputs/runs.csv`:

```python
def cloud_row(skill, llm, item, question=""):
    with Stopwatch() as watch:
        reply = llm.invoke(cloud_messages(skill, item, question))   # [SystemMessage(SYSTEM), HumanMessage(note)]
    return Row("cloud", model, "Google's servers", watch.seconds, reply.text.strip())   # .text, not .content
```

```bash
python compare.py summarise
```

Expected shape with your key (the local row is the real run; Gemini writes its own sentence, and its time depends on
your connection):

```text
summarise: jaipur_trip.txt

local  Qwen/Qwen2.5-0.5B-Instruct  (your laptop, 1.0 s, 19 words)
  Meera's wedding was planned for Thursday, 12th, and booked two sleeper berths on the overnight train from Mumbai Central.

cloud  gemini-3.5-flash-lite  (Google's servers, <N> s, <N> words)
  <Gemini's one-sentence summary of the note, in its own words>

engine  where it ran             seconds  words   data left your laptop?
local   your laptop                  1.0     19   no
cloud   Google's servers           <N>     <N>   yes
```

Without a key, the local half still runs, and you get the friendly help instead of the cloud row (real output):

```text
engine  where it ran             seconds  words   data left your laptop?
local   your laptop                  1.0     19   no

No API key found for LLM_PROVIDER=gemini.
Looked for: GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.

Get a FREE Gemini key (no credit card needed):
  1. Open https://aistudio.google.com/apikey
  ...
```

Look at the last column. That is the privacy difference in one word: the local model never sends your note anywhere.

> [!WARNING]
> **Common errors**
> - `The cloud call failed: ... 429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute; the local row is still printed and logged.
> - Want only one side? `python compare.py summarise --local-only` or `--cloud-only`.

## Step 6 - Translate a note into Hindi and back

The same chat model, a different system message (`skills/translate.py`):

```python
def system_message(language):
    script = f", in {SCRIPTS[language]} script" if language in SCRIPTS else ""      # Hindi -> Devanagari
    return (f"You are a translator. Translate the user's text into {language}. "
            f"Reply with the translation only{script}.")

there = translate(text, "Hindi")
back = translate(there.text, "English")      # the trip back is your judge
```

> **Classic vs 2026:** older tutorials used `pipeline("translation", model="facebook/nllb-200-distilled-600M", torch_dtype=...)`
> with language codes. In v5 the task is gone and `torch_dtype` is called `dtype`; one chat model translates with a
> system message instead. NLLB itself still exists, but its licence is non-commercial (step 2).

Can't read Hindi? Translate it back. If the English that comes back has lost the meaning, the Hindi was wrong too.

```bash
python -m skills.translate
```

Expected output:

```text
Local model: Qwen/Qwen2.5-0.5B-Instruct on cpu (ready in 2.3 s)

English:
  Water the tulsi every morning and keep it in the balcony sun. The money plant only needs water once a week. Please water the plants on Monday and Thursday while we are away.

Hindi (3.5 s, 199 tokens, stopped at the 200-token limit):
  पूरा तुलसी को दिन में प्राप्त करते हुए चलने और बंधन जोड़े। अमेरिकी टैक्स वह एक छवि लगातार 1 वर्ष की आयुक्रम की खोज करता है। इसका उपयोग यह सबसे अच्छा था। भाग डिमिंग फैलाते हुए तुलसी को दिन में प्राप्त करें और बंधन

Back to English (1.3 s):
  The dog should be trained to bark for one year and barks twice daily. This was the best choice.

Judge it: does the trip back keep the meaning? Compare with Gemini: python compare.py translate
```

Be honest with yourself here: **this is a failed translation.** The letters are Devanagari, but the sentences do not
make sense (one even mentions American tax), the model rambled until the 200-token limit, and the trip back is about
a dog. A one-line test gives the same verdict:

```bash
python -m skills.translate --text "Please water the plants on Monday and Thursday while we are away."
```

```text
Local model: Qwen/Qwen2.5-0.5B-Instruct on cpu (ready in 2.6 s)

English:
  Please water the plants on Monday and Thursday while we are away.

Hindi (0.8 s, 34 tokens):
  प्रकाश को दिन में और तीसरा वहाँ पड़े।

Back to English (1.0 s):
  I'm sorry, but I can't translate that sentence for you as it appears to be in a language I don't recognize or understand. Could you please provide me with the original text so I can assist you better?

Judge it: does the trip back keep the meaning? Compare with Gemini: python compare.py translate
```

Why? Read the card again. It promises "over 29 languages" for the Qwen2.5 family and names thirteen of them; Hindi
is not one of the names. And 0.5 billion parameters is the smallest size of that family, with little room for a
language it rarely saw. Try a language the card does name, `python -m skills.translate --to French`: at a glance it looks
like ordinary French, but the meaning drifts. The trip back says the tulsi "needs to be covered all day long", the
money plant becomes "the branches of the life tree" that "do not deserve to be covered fifteen times per year", and
the watering turns into "We must have a meal on Tuesday and Thursday" (the French itself even says Tuesday and
Wednesday). Smooth sentences are not the same as a correct translation.

Now the cloud side:

```bash
python compare.py translate
```

Expected shape with your key (the local row is real; rate the two Devanagari outputs yourself, or ask a Hindi-speaking friend):

```text
translate: plant_care.txt

local  Qwen/Qwen2.5-0.5B-Instruct  (your laptop, 3.5 s, 44 words)
  पूरा तुलसी को दिन में प्राप्त करते हुए चलने और बंधन जोड़े। अमेरिकी टैक्स ...

cloud  gemini-3.5-flash-lite  (Google's servers, <N> s, <N> words)
  <Gemini's Hindi translation, in Devanagari>

engine  where it ran             seconds  words   data left your laptop?
local   your laptop                  3.5     44   no
cloud   Google's servers           <N>     <N>   yes
```

The lesson: for Indian languages, a tiny local chat model is not good enough. Use the cloud model, a bigger local model
(Challenge B tries one through Ollama), or a model made only for translation, after you read its licence. NLLB covers
200 languages, but its CC-BY-NC-4.0 licence means non-commercial projects only.

> [!WARNING]
> **Common errors**
> - The Hindi looks right but you cannot read it - use the trip back, or ask someone who reads Hindi. Never ship a translation nobody checked.
> - A broken character at the end - the reply was cut in the middle of a letter at the token limit. Our code removes it; the text was unfinished anyway.
> - Empty boxes instead of Hindi letters (Windows) - the old console window has no font for Devanagari. Run the command in the VS Code terminal or in Windows Terminal; the text itself is fine.

## Step 7 - Encode all your notes

An embeddings model turns a whole note into ONE list of 384 numbers. This is not how the chat model reads: the chat
model reads token IDs (Episode 02). Embeddings come from a separate, small model and are used for search, clustering
and RAG (Episode 07).

`skills/search.py`:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
vectors = encoder.encode(texts, normalize_embeddings=True)   # every row gets length 1
```

```bash
python -m skills.search --shape
```

Expected output:

```text
Encoded 8 notes with sentence-transformers/all-MiniLM-L6-v2 in 2.5 s
Shape: (8, 384)   (8 notes x 384 numbers each)
First 5 numbers of amma_birthday: [-0.029, 0.123, 0.062, 0.006, -0.084]
Length of every vector: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  (normalised, so a dot product is the cosine)
```

`(8, 384)`: 8 notes, 384 numbers each. MiniLM reads at most 256 word pieces per text and cuts the rest, so keep notes short (ours are 40 to 85).

> [!WARNING]
> **Common errors**
> - `Put at least two .txt notes in my_notes first.` - the folder is empty or has no `.txt` files.
> - Your numbers differ - you changed the notes. That is the idea: they are YOUR notes now.

## Step 8 - The full similarity matrix

Picture every note as an arrow. Cosine similarity measures the angle between two arrows: **1.00** = same direction
(same meaning), **0** = unrelated, **below 0** = pointing away. Because every vector has length 1, the cosine is a dot
product, and one matrix multiplication gives every pair at once:

```python
def similarity_matrix(vectors):
    return vectors @ vectors.T          # N x N, 1.00 on the diagonal
```

(`sentence_transformers` also has `model.similarity(a, b)`, and older tutorials use `util.cos_sim(a, b)`. Same numbers;
here we do the maths ourselves to see it.)

```bash
python -m skills.search --matrix
```

Expected output:

```text
Cosine similarity of every note with every note (-1 to 1):

                        1      2      3      4      5      6      7      8
 1. amma_birthday    1.00  -0.08   0.21   0.43   0.00   0.30   0.14   0.03
 2. dbms_class      -0.08   1.00  -0.06  -0.08   0.02  -0.02  -0.05   0.14
 3. groceries        0.21  -0.06   1.00   0.17  -0.00   0.24   0.28   0.12
 4. jaipur_trip      0.43  -0.08   0.17   1.00   0.14   0.25   0.14   0.02
 5. laptop_repair    0.00   0.02  -0.00   0.14   1.00   0.10   0.10   0.13
 6. monthly_budget   0.30  -0.02   0.24   0.25   0.10   1.00   0.32  -0.04
 7. plant_care       0.14  -0.05   0.28   0.14   0.10   0.32   1.00   0.00
 8. python_class     0.03   0.14   0.12   0.02   0.13  -0.04   0.00   1.00

Most similar pair:  amma_birthday + jaipur_trip  (0.43)
Least similar pair: amma_birthday + dbms_class  (-0.08)
```

How to read it:

- **The diagonal is always 1.00**: every note is identical to itself. It tells you nothing. Read the other cells.
- The grid is a mirror: row 1, column 4 equals row 4, column 1.
- The brightest cell off the diagonal is **0.43**: Amma's birthday and the Jaipur wedding trip. Both are family
  events with a gift to carry.
- Some cells are **negative** (-0.08): a database lecture and a birthday point slightly away from each other.
  Read the sign and the decimals: -0.08 is not "8", and `-0.00` is a tiny negative number that rounds to zero.
- The two class notes score only **0.14**: "class notes" is not a topic to this model; databases and Python lists are.

> [!WARNING]
> **Common errors**
> - "0.43 looks low for similar notes" - scores from this small model are usually between about -0.1 and 0.6 for different texts. Compare cells with each other, not with 1.0.

## Step 9 - Semantic search

Search = encode the question with the same model, then rank the notes by cosine score:

```python
def search(query, notes, k=3):
    vectors = encode([note.text for note in notes])
    query_vector = encode([query])[0]
    return rank(query_vector, vectors, [note.name for note in notes])[:k]   # (name, score), best first
```

```bash
python -m skills.search "train ticket"
```

Expected output:

```text
Query: "train ticket"

1. +0.310  jaipur_trip      Jaipur trip for Meera didi's wedding. Book two sleeper be...
2. +0.132  amma_birthday    Amma's birthday is on Sunday. Order a small eggless choco...
3. +0.117  groceries        Groceries for this week: 5 kg atta, 1 kg toor dal, onions...
```

The word "ticket" is in none of the notes. The journey note still wins clearly, because "sleeper berths",
"overnight train" and "booking confirmation" mean the same thing.

Now an honest miss:

```bash
python -m skills.search "what should I buy for mom"
```

```text
Query: "what should I buy for mom"

1. +0.255  monthly_budget   Budget for this month: rent and electricity first, then t...
2. +0.239  amma_birthday    Amma's birthday is on Sunday. Order a small eggless choco...
3. +0.228  groceries        Groceries for this week: 5 kg atta, 1 kg toor dal, onions...
```

The right note is only second, and the three scores are close together. This English-only model does not know that
"Amma" means mom. Ask `"Amma birthday gift"` instead and the birthday note jumps to the top with +0.784: shared words
still matter a lot. Close scores mean "not sure", so show the user more than one result.

> [!WARNING]
> **Common errors**
> - `usage: python -m skills.search [-h] ...` - you forgot the question. Put it in quotes: `"train ticket"`.
> - Every score is low - your question is about something that is not in any note. That is a correct answer too.

## Step 10 - Ask two questions about your photo

Take a photo of your own desk with your phone and copy it into `photos/` (for example `photos/my_desk.jpg`). Only
photos you took, and no other people's faces. No photo yet? Use the drawn sample `photos/desk.jpg`.

![The sample desk: an open laptop, a red mug, a plant in a pot, three books and five yellow pencils](photos/desk.jpg)

How a vision-language model (VLM) answers: it resizes the photo to fit 512 x 512 pixels and cuts it into small
square **patches** of 16 x 16 pixels (32 x 32 = 1,024 patches). A vision encoder turns the patches into vectors, groups them 16 at a time, and passes
**64 image tokens** to a small language model, right next to the tokens of your question. The language model attends to
both and writes the answer word by word. **There is no database and no search anywhere in this:** the answer is
generated from the pixels and your question.

`skills/ask_photo.py`:

```python
from transformers import AutoProcessor, pipeline

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct",
                                          do_image_splitting=False)   # ONE 512 x 512 view instead of many tiles
vlm = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM-256M-Instruct",
               processor=processor, dtype="auto", device="cpu")

messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": question}]}]
output = vlm(text=messages, return_full_text=False,
             generate_kwargs={"max_new_tokens": 40, "do_sample": False})
answer = output[0]["generated_text"]
```

> **Classic vs 2026:** older tutorials loaded `BlipProcessor` and `BlipForQuestionAnswering` (BLIP, from 2022). Those
> classes still import in transformers 5.20.2, but the `visual-question-answering` pipeline is gone, and current small
> vision-language models answer in full sentences through `image-text-to-text`. Either way, it is not a database search.

```bash
python -m skills.ask_photo
```

Expected output with the sample photo:

```text
Photo: desk.jpg (960x640)   Model: HuggingFaceTB/SmolVLM-256M-Instruct on cpu (ready in 2.4 s)

Q: What objects are on the desk?
A: There is a laptop, a pencil, a cup, and a pencil sharpener on the desk.   (2.8 s)

Q: What colour is the mug?
A: The mug is red.   (2.5 s)
```

The colour is right. The list is half right: it found the laptop, a pencil and the cup, missed the plant and the
books, and invented a pencil sharpener. Your own photo: `python -m skills.ask_photo photos/my_desk.jpg "What is on the desk?"`

Why `do_image_splitting=False`? With splitting on, the processor cuts the photo into 12 tiles plus one overview:
832 image tokens instead of 64. In our test the same counting question took 29.4 s instead of 2.5 s on the CPU, and the
answer got worse: "2 pencils" instead of "4" (the right answer is 5, see step 11).

> [!WARNING]
> **Common errors**
> - `HuggingFaceTB/SmolVLM-256M-Instruct is not on this computer yet` - the first run downloads about 0.5 GB. Connect and run again.
> - `Cannot read my_desk.HEIC as a picture` - iPhones save HEIC by default. Export the photo as JPEG (or set the camera to "Most Compatible").
> - The photo is sideways - `open_photo()` already applies the phone's rotation flag; if it is still sideways, rotate the file once in your photo app.

## Step 11 - A question the small model gets wrong

Small models have limits. Counting is a classic one. We know the right answer, so we pass it with `--truth`, and a
wrong answer is written to `outputs/limits.csv`, a list of what this model cannot do:

```bash
python -m skills.ask_photo photos/desk.jpg "How many pencils are on the desk?" --truth 5
```

Expected output:

```text
Photo: desk.jpg (960x640)   Model: HuggingFaceTB/SmolVLM-256M-Instruct on cpu (ready in 2.5 s)

Q: How many pencils are on the desk?
A: There are 4 pencils on the desk.   (2.5 s)
   WRONG: the true answer is 5. Logged in outputs/limits.csv
```

`outputs/limits.csv` now holds (your time stamp will differ):

```text
when,model,photo,question,answer,truth
2026-10-26 16:34,HuggingFaceTB/SmolVLM-256M-Instruct,desk.jpg,How many pencils are on the desk?,There are 4 pencils on the desk.,5
```

It is not a one-off: `"How many books are there?" --truth 3` gets `There are 10 books.` Now the same counting question
for Gemini:

```bash
python compare.py ask-photo photos/desk.jpg "How many pencils are on the desk?" --truth 5
```

Expected shape with your key (the local row is real; Gemini's answer and verdict come from your run):

```text
ask-photo: desk.jpg  Q: "How many pencils are on the desk?"

local  HuggingFaceTB/SmolVLM-256M-Instruct  (your laptop, 2.5 s, 7 words)
  There are 4 pencils on the desk.

cloud  gemini-3.5-flash-lite  (Google's servers, <N> s, <N> words)
  <Gemini's answer to the counting question>

engine  where it ran             seconds  words   data left your laptop?
local   your laptop                  2.5      7   no
cloud   Google's servers           <N>     <N>   yes
local: WRONG (the true answer is 5)
cloud: <right or WRONG> (the true answer is 5)
```

`compare.py` shrinks the photo to at most 1024 pixels and sends it as a JPEG inside the message. That photo now
leaves your laptop: another reason to use only your own pictures.

> [!WARNING]
> **Common errors**
> - `right` when you expected a mistake - with your own photo, the model may count correctly. Try a harder question: small text, many similar objects, or "what is to the left of the mug?"
> - `--truth` checks whether the right value appears in the answer ("5" also matches "five"). It is a quick check, not a perfect judge.

## Step 12 - Wrap the skills in a CLI

`app.py` puts the four skills behind one command. Each command hands the rest of the line to its skill:

```python
COMMANDS = {
    "summarise": ("skills.summarise", "a one-sentence summary of a note"),
    "translate": ("skills.translate", "translate a note into Hindi and back"),
    "ask-photo": ("skills.ask_photo", "ask questions about your photo"),
    "search": ("skills.search", "find notes by meaning"),
}

def run(command, args):
    module = importlib.import_module(COMMANDS[command][0])   # import the skill (and PyTorch) only now
    return module.main(args)
```

```bash
python app.py --help
```

```text
Tiffin Toolbox: local AI on your laptop

Commands:
  1. summarise  a one-sentence summary of a note
  2. translate  translate a note into Hindi and back
  3. ask-photo  ask questions about your photo
  4. search     find notes by meaning

Examples:
  python app.py summarise my_notes/jaipur_trip.txt
  python app.py search "train ticket"
  python app.py ask-photo photos/desk.jpg "What colour is the mug?"
```

Every command works the same way as its skill, for example `python app.py search "train ticket"` prints the step 9
table. Run `python app.py` with nothing after it for a small menu. The menu keeps the models loaded between choices,
so the second summary starts at once. In the recorded run (typed: `1`, Enter, `1`, Enter, `4`, `train ticket`, `q`)
the first summary printed `model ready in 3.0 s` (2.4 s in an earlier run: it depends on your laptop) and the second
`model ready in 0.0 s`.

> [!WARNING]
> **Common errors**
> - `Unknown command: summarize` - the commands use the British spelling: `summarise`.
> - `python app.py search train ticket` searches only for "train" (the shell splits words). Use quotes.

## Step 13 - Load lazily and release

Each model takes hundreds of megabytes of memory while it is loaded. Tiffin Toolbox loads a model only the first time
a skill needs it, and can let it go again (`skills/lazy.py`):

```python
_loaded = {}

def lazy_load(name, load_function):
    if name not in _loaded:
        _loaded[name] = load_function()       # the first call loads; later calls reuse it
    return _loaded[name]

def release(name=None):
    names = [name] if name else list(_loaded)
    released = [n for n in names if _loaded.pop(n, None) is not None]
    gc.collect()                              # free what nobody points to any more
    # (and torch.cuda.empty_cache() / torch.mps.empty_cache() on a GPU)
    return released
```

```bash
python memory_check.py
```

Expected output (your megabytes will differ a little):

```text
moment                                       memory      change
start (no model, no PyTorch loaded yet)       50 MB               models loaded: none
after one summary                          1,604 MB   +1,553 MB   models loaded: chat
after release('chat')                        575 MB   -1,029 MB   models loaded: none
after one photo question                   1,067 MB     +492 MB   models loaded: vision
after release()                              611 MB     -456 MB   models loaded: none
```

The drop after `release('chat')` (1,029 MB) is about the size of the Qwen weights file (988 MB). What stays after a
release (about 600 MB) is mostly PyTorch and transformers themselves, which stay imported. On a laptop with 8 GB of memory, load one model at a time.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'psutil'` - step 0.2.
> - The memory does not drop - something still points to the model (for example a variable in your own code). `release()` can only free what nobody uses.

## Step 14 - Go offline

Switch the Wi-Fi off. Then tell the Hugging Face libraries to use only the files on your disk:

**macOS / Linux:**

```bash
export HF_HUB_OFFLINE=1
python app.py summarise
python app.py translate
python app.py ask-photo
python app.py search "train ticket"
python compare.py summarise
```

**Windows (PowerShell):**

```powershell
$env:HF_HUB_OFFLINE = "1"
python app.py summarise
python app.py translate
python app.py ask-photo
python app.py search "train ticket"
python compare.py summarise
```

Expected: every local command prints the same answers as in steps 4, 6, 9 and 10; only the seconds change (we
recorded them with all network access blocked). The comparison prints the local row and then a clear message instead of the cloud row:

```text
engine  where it ran             seconds  words   data left your laptop?
local   your laptop                  1.2     19   no

No internet: skipped the cloud column (gemini-3.5-flash-lite).
The local results above came from your laptop and never needed the internet.
```

Why `HF_HUB_OFFLINE=1`? Without it, the libraries first try to ask the Hub whether a newer version exists. With the
network blocked and no `HF_HUB_OFFLINE`, our search still worked but took 26.2 s instead of about 3 s, after five
retry lines, from `Retrying in 1s [Retry 1/5]` to `Retrying in 8s [Retry 5/5]`. You can also put `HF_HUB_OFFLINE=1` in the root `.env`; remove it again when you
want to download a new model. (Turn it off in this terminal with `unset HF_HUB_OFFLINE`, or `Remove-Item Env:HF_HUB_OFFLINE` on Windows.)

Now the whole picture, built from every comparison you ran:

```bash
python compare.py table
```

Expected shape after steps 5, 6 and 11 with your key (the local speeds are from our run; your numbers come from `outputs/runs.csv`):

```text
                  local (Hugging Face models)       cloud (gemini-3.5-flash-lite)
summarise speed   1.1 s (2 runs)                    <N> s (<N> runs)
translate speed   3.5 s (1 run)                     <N> s (1 run)
ask-photo speed   2.5 s (1 run)                     <N> s (1 run)
privacy           stays on your laptop              sent to Google's servers
cost              free (your electricity)           free tier with limits, or paid
works offline     yes (HF_HUB_OFFLINE=1)            no
quality           you judge: read the answers in outputs/runs.csv and outputs/limits.csv
Gemini free tier: Google may use what you send to improve its products. Send nothing private.
```

Privacy comes from **where the model runs**, not from the model being open. An open model on someone else's server
still receives your data; a closed model you cannot run locally always does.

> [!WARNING]
> **Common errors**
> - `... is not on this computer yet, and the Hugging Face Hub cannot be reached.` - that model was never downloaded. Run its step once with internet, then go offline.
> - Each command waits about 20-30 seconds with the Wi-Fi off - you forgot `HF_HUB_OFFLINE=1` in this terminal.

## Step 15 - Run the tests

The tests use a **fake Gemini** with scripted answers, so they need no key, cost nothing and give the same result
every time. They also block the internet on purpose. Tests for the real local models run only when those models are
already downloaded.

```bash
python -m pytest
```

Expected output:

```text
..............................s.....................................                         [100%]
67 passed, 1 skipped in 11.32s
```

Your time will differ. An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the
CrewAI episodes. On a computer where the three models were never downloaded you see `60 passed, 8 skipped`:
run steps 4, 7 and 10 once with internet, then test again.

What they check: ranking on fixed sentences (`test_search.py`), the matrix maths and its two-decimal grid
(`test_matrix.py`), the command line and the menu (`test_cli.py`), the Gemini path and the no-key and no-internet
messages (`test_compare_fake.py`), the licence report from saved cards (`test_models_report.py`), the removed task
names (`test_old_tasks.py`) and the real models (`test_local_models.py`).

How a test checks the cloud prompt (from `tests/test_compare_fake.py`):

```python
def test_cloud_summary_gets_the_same_instructions_as_the_local_model():
    llm = fake_llm("Two sleeper berths to Jaipur are booked; be at the station by 9 pm.")

    row = compare.cloud_row("summarise", llm, NOTE)

    assert row.engine == "cloud" and row.where == "Google's servers"
    sent = llm.received[0]
    assert sent[0].type == "system" and sent[0].content == SYSTEM
    assert sent[-1].type == "human" and NOTE in sent[-1].content      # Gemini needs the human turn last
```

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'app'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called a real model. Pass `llm=fake_llm(...)`.
> - More `s` than expected - run `python -m pytest -rs` to see why. "is not downloaded yet" means: run that step once with internet.

## Local or cloud? A decision you can now measure

| Question | Lean towards |
|---|---|
| Is the data private (health, money, other people's details)? | **local** |
| Do you need the best quality, or Indian languages? | **cloud** (or a bigger local model you have tested) |
| No internet, or unreliable internet? | **local** |
| Thousands of requests a day? | measure both: local costs a machine and electricity, cloud costs per request |
| Small laptop memory? | one small local model at a time (step 13), or cloud |
| Does the licence allow your use? | read the card first (step 2), for local AND cloud |

Cooking at home versus ordering in: home cooking gives you control and privacy, ordering in gives you range and
convenience. Most real projects do both.

## Final challenge

Give Tiffin Toolbox a fifth skill: **speech to text** with a small Whisper-style model on a voice note you record, or
**Ollama** as a third column in the comparison table. Details in [CHALLENGE.md](CHALLENGE.md).

## What you learned

- The Hub has model repos, dataset repos and Spaces. Read the card first: task, size, languages and licence.
  Apache-2.0 and MIT allow business use; NLLB and FLUX.1-dev do not. Tokens: read-only, in `.env`, only when needed.
- transformers v5 runs models with `pipeline(task, model=..., dtype="auto", device=...)`. The old summarization,
  translation and visual-question-answering tasks are gone; `text-generation` and `image-text-to-text` replace them.
- A model downloads once into the cache; `HF_HUB_OFFLINE=1` makes every later run local and fast.
- Small local models are fast, free and private, and they make mistakes you must check: an invented price, a broken
  Hindi translation, a wrong count.
- Embeddings are not tokens. Cosine scores run from -1 to 1; read the whole matrix, not just the diagonal.

## The sixteen projects you built

| Ep | Project | Folder |
|---|---|---|
| 01 | Byte Bites: canteen feedback sorted four ways, from rules to Gemini | [ep01-four-kinds-of-ai](../ep01-four-kinds-of-ai/) |
| 02 | Attention X-Ray: tokens, embeddings and a real attention head, opened up | [ep02-attention-xray](../ep02-attention-xray/) |
| 03 | Next-Token Lab: a decoder that writes one token at a time | [ep03-next-token-lab](../ep03-next-token-lab/) |
| 04 | Model Zoo: four foundation models, each doing one job | [ep04-model-zoo](../ep04-model-zoo/) |
| 05 | Model Scorecard: your own exam for two Gemini models, plus a safety gate | [ep05-model-scorecard](../ep05-model-scorecard/) |
| 06 | Help Desk Ladder: a Dabbaloop assistant built one LangChain rung at a time | [ep06-helpdesk-ladder](../ep06-helpdesk-ladder/) |
| 07 | Ask the Handbook: RAG answers with citations, and "I don't know" | [ep07-ask-the-handbook](../ep07-ask-the-handbook/) |
| 08 | Chunk Lab: splitting strategies measured, not guessed | [ep08-chunk-lab](../ep08-chunk-lab/) |
| 09 | Chai Kitty Prompt Lab: messy trip notes into typed, checked expenses | [ep09-prompt-lab](../ep09-prompt-lab/) |
| 10 | Kadak Kettle Counter Bot: loaders for CSV, HTML and PDF, plus memory | [ep10-counter-bot](../ep10-counter-bot/) |
| 11 | Party Planner: one sentence into a menu, a shopping list and a budget check | [ep11-party-planner-chains](../ep11-party-planner-chains/) |
| 12 | Dabbaloop Help Desk: RAG, a router and read-only SQL in one assistant | [ep12-dabbaloop-help-desk](../ep12-dabbaloop-help-desk/) |
| 13 | Shelf Sifter: a three-agent CrewAI team on a practice bookshop | [ep13-shelf-sifter-crew](../ep13-shelf-sifter-crew/) |
| 14 | Crew Office: email drafts that wait for your click, and an education-only research desk | [ep14-crew-office](../ep14-crew-office/) |
| 15 | Mandi Tool Belt: one agent choosing between a calculator, read-only SQL and search | [ep15-mandi-tool-belt](../ep15-mandi-tool-belt/) |
| 16 | Tiffin Toolbox: local AI on your laptop, compared with the cloud | this folder |

The habits that carried through all sixteen: **measure** before you believe, **ground** answers in your own data,
**guard** anything that acts (read-only SQL, human approval, licences), and **test offline** with fakes.

## What to build next

- Point Episode 07's handbook bot at your own college or office documents, with local embeddings.
- Put Tiffin Toolbox behind a small web page (Streamlit is already in the course requirements).
- Build an eval set for your own task (Episode 05) and use it to pick between a local and a cloud model.
- Share what you built in the comments. Ravi reads them.

---

The notes in `my_notes/` and the drawing in `photos/` were made for this course; any match with real people or
plans is a coincidence. The drawing is released under CC0. Model licences are listed in step 2.
