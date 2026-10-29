<!--
README_TEMPLATE.md - copy this into each episode folder as README.md and fill in every {{PLACEHOLDER}}.
Rules for every episode README:
  - Baby steps: one small action per step, and show what the viewer should SEE after it.
  - Every step has: what we do + code + expected output + a "Common errors" box.
  - Use Gemini (free key) in the on-screen run. Mention the one-line .env switch once.
  - One shared .venv and ONE .env at the repo root. Never tell viewers to make a per-episode venv or .env.
  - Name the classic LangChain name when we use its modern replacement.
  - Delete these comments when done.
-->

# Episode {{NN}}: {{TITLE}}

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: {{YOUTUBE_LINK}}  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/{{FOLDER}}

## What you will build

{{ONE_OR_TWO_SENTENCES: what the finished project does, in plain words.}}

```text
{{SHORT EXAMPLE OF THE FINISHED PROGRAM RUNNING, e.g.
You: What is an embedding?
AI:  An embedding is a list of numbers that captures the meaning of text...}}
```

**You will learn:**

- {{CONCEPT_1}}
- {{CONCEPT_2}}
- {{CONCEPT_3}}

<!-- Only if the episode replaces an old LangChain class: -->
> **Seen an older tutorial?** It used `{{CLASSIC_NAME}}`. In 2026 we use `{{MODERN_NAME}}` instead.
> Same idea, newer and supported API. (The old class still exists in `langchain_classic`, but it is deprecated.)

## Prerequisites

- Finished [Episode {{NN-1}}]({{LINK}}) (or know {{WHAT_YOU_NEED_TO_KNOW}})
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- The shared `.venv` and root `.env` with your **free** Gemini key from Episode 1 ([SETUP.md](../SETUP.md))
- About {{MINUTES}} minutes{{, and ~XXX MB of disk for the local model, if any}}

## Step 0 - Get ready (about 2 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the
repo root (`ai-engineering-course/`). You created them once in Episode 1
([SETUP.md](../SETUP.md)). New here? Do SETUP.md first, then come back.
<!-- CrewAI episodes (13-14): use .venv-crewai instead of .venv everywhere in this step,
     and requirements-crewai.txt instead of requirements-base.txt. -->

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd {{FOLDER}}
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd {{FOLDER}}
```

Expected: your prompt starts with `(.venv)` and ends in `{{FOLDER}}`.

> [!WARNING]
> **Common errors**
> - `.venv\Scripts\Activate.ps1 cannot be loaded` / `No such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. Run `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again:
>   `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`
> - VS Code already shows `(.venv)` in new terminals once you pick the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.

### 0.2 Make sure the libraries are up to date

Nothing new to install if you already ran Episode 1: this episode uses the shared `requirements-base.txt`.
If you pulled a newer version of the repo, update the environment from the repo root:

```bash
pip install -r ../requirements-base.txt
pip check
```

Expected (last line):

```text
No broken requirements found.
```

<!-- Episode needs an extra package? Add it to the ROOT requirements-base.txt (and _template's copy),
     never to a per-episode venv. tools/sync_template.py reports when the two copies drift. -->

> [!WARNING]
> **Common errors**
> - `No matching distribution found for torch==2.14.1` - the `.venv` was made with Python 3.14/3.15. Delete the `.venv` folder and recreate it with Python 3.13 ([SETUP.md](../SETUP.md)).
> - Linux without an NVIDIA GPU: install the CPU-only PyTorch first (SETUP.md), otherwise pip downloads several GB of CUDA files.

### 0.3 Check that your key is found from this folder

Your key lives in `ai-engineering-course/.env` (made in Episode 1). `llm.py` looks for it in this folder
and then in each parent folder, so it finds the root file by itself. Do not copy `.env` into episode folders.

```bash
python llm.py
```

Expected output (the greeting will differ; your path will differ):

```text
Settings file: /.../ai-engineering-course/.env
Provider: gemini   Model: gemini-3.5-flash-lite
API key found. Asking the model a tiny question...
Hello! It's nice to meet you.
```

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama` (free, local - install from https://ollama.com and run `ollama pull qwen3.5:4b`). Nothing else changes.

> [!WARNING]
> **Common errors**
> - `Settings file: no .env found` and `No API key found` - the root `.env` is missing or is named `.env.txt` (Windows hides extensions!). Create it from the repo root: `Copy-Item .env.example .env` (Windows) or `cp .env.example .env` (macOS/Linux), then paste your key.
> - `429 RESOURCE_EXHAUSTED` - you hit the free-tier limit. Wait a minute (per-minute limit) or until tomorrow (daily limit, resets at midnight Pacific time). See your limits at https://aistudio.google.com/rate-limit
> - `404 ... model not found` - the model name in `LLM_MODEL` is wrong or retired. Leave `LLM_MODEL=` empty to use the default.

---

## Step 1 - {{STEP_TITLE}}

{{ONE_OR_TWO_SENTENCES: what we are doing and WHY.}}

Create a file `{{FILE_NAME}}.py`:

```python
{{CODE - short, commented, no clever tricks}}
```

Run it:

```bash
python {{FILE_NAME}}.py
```

Expected output:

```text
{{EXACT OR CLOSE OUTPUT; say "(your text will differ)" when the AI writes it}}
```

> [!WARNING]
> **Common errors**
> - `{{ERROR_MESSAGE}}` - {{WHY IT HAPPENS}}. Fix: {{FIX}}.
> - `{{ERROR_MESSAGE}}` - {{WHY IT HAPPENS}}. Fix: {{FIX}}.

## Step 2 - {{STEP_TITLE}}

{{Same structure as Step 1: why, code, run, expected output, common errors.}}

## Step 3 - {{STEP_TITLE}}

{{...add as many small steps as needed...}}

## Step {{N}} - Test it without internet

Every project comes with tests that use a **fake AI model**, so they run without a key, for free, and give the same result every time.

```bash
python -m pytest
```

Expected output:

```text
...........s                                                             [100%]
{{N}} passed, 1 skipped in 1.2s
```

An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes. (In episodes 13-14 you will see `5 skipped` instead: the LangChain-only checks.)

How a test works (from `tests/test_app.py`):

```python
from testing.fakes import fake_llm

def test_explain_returns_the_model_answer():
    llm = fake_llm("Embeddings turn text into lists of numbers.")   # we script the answer
    assert explain("embeddings", llm=llm) == "Embeddings turn text into lists of numbers."
```

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'app'` - run pytest from the episode folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called the real model. Pass `llm=fake_llm(...)` to your function.
> - `OSError: We couldn't connect to 'https://huggingface.co' ...` - tests only use Hugging Face models that are already downloaded. Run the app once with internet (it downloads the model), then run the tests again.

## Final challenge

{{ONE small extension the viewer does alone, e.g. "Add a second prompt that answers in Hindi and route questions to it."}}

<details>
<summary>Hint</summary>

{{A nudge, not the full answer.}}

</details>

## What you learned

- {{RECAP_1}}
- {{RECAP_2}}
- {{RECAP_3}}

## Next episode

**Episode {{NN+1}}: {{NEXT_TITLE}}** - {{ONE_SENTENCE_TEASER}} -> [{{NEXT_FOLDER}}](../{{NEXT_FOLDER}})
