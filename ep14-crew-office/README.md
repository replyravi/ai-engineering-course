# Episode 14: CrewAI Projects: Email Drafts With Human Approval and an Education-Only Stock Research Desk

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep14-crew-office

## What you will build

Two small CrewAI crews that do useful work and never act on their own.

**Part 1, the outreach office.** Three agents write a personalised first email for each of three invented businesses, from local files only: a **Researcher** collects facts with their source lines, a **Writer** drafts from those facts only, a **Reviewer** checks the tone with a real local model and checks that no number appears that the brief does not support. Then **you** read every draft in `approve.py` and approve, edit or reject it. Approved drafts are saved as files with a `mailto:` link. Nothing is ever sent by code.

**Part 2, the research desk (education only).** Three agents explain the invented prices of an invented company, `DEMO1`: what moved, what the numbers mean, what a paper test of a simple rule showed, and the risks. Every number comes from Python (pandas), our code stamps a disclaimer on the report, and a claims checker proves that no number in the agents' text was guessed. Then we run the same desk with a **manager** agent (hierarchical) and measure what that costs.

```text
$ python outreach_crew.py --all
Outreach crew: 3 lead(s)   agents: researcher, writer, reviewer   today: <your date>
  ... CrewAI's step-by-step log, the typed brief, the draft and the verdict for each lead ...
Summary (3 leads, one kickoff_for_each)
  lead                     traced   outside  reviewer  requests  tokens  draft
  dhoop-chhaon-textiles    6 of 6   0        pass      <n>       <n>     drafts/pending/dhoop-chhaon-textiles.json
  ...
$ python approve.py
[a]pprove  [e]dit  [r]eject  [s]kip ? a
Approved. Saved drafts/approved/dhoop-chhaon-textiles.md
mailto:farah.sheikh@example.com?subject=...
Saved 1 draft. Nothing was sent.

$ python desk_crew.py --ticker DEMO1
Saved reports/DEMO1.md and reports/DEMO1.numbers.json
Result: 0 unsupported numbers, disclaimer present
```

(This is the **shape** of a live run. Lines with `<...>` and every sentence an agent writes come from Gemini and will differ. The file names, the checks and every number in Part 2 come from our code.)

**You will learn:**

- Trace one CrewAI run end to end: inputs, tasks, agents, tools, process, final output.
- Reuse one crew for many cases with `{placeholders}` and `kickoff_for_each`, and check that every input is really used.
- Build a real custom tool and test it alone, with known inputs, before an agent uses it.
- Pass one task's answer to the next with `context`, and get typed answers with `output_pydantic`.
- Put a human approval gate outside the crew, so no email is ever sent automatically.
- Compare the sequential and hierarchical processes with an explicit manager model, and cap the cost with `max_iter`, `max_rpm` and usage metrics.
- Explain why a finance crew's report is not advice, why every number must come from code, and why today's date must be passed in.

> **Seen an older tutorial?** Seven things changed or were never right (checked with CrewAI 1.15.28 on 2026-10-26):
> - **A stub tool that always answers "positive"** teaches the agent nothing: it asks again and again and gives up. Our tone tool runs a real local model and was tested with known sentences first (Step 3).
> - **An email agent that sends by itself**, with no approval step. Our crew has no tool that can send, and a person decides in `approve.py` (Steps 6 and 7).
> - **Inputs that no task uses** (a "budget" or a "news" flag) are silently ignored by CrewAI. We check every input against the task text (Step 2).
> - **`memory=True` with nothing else set.** In CrewAI 1.15.28 that adds model calls before and after every task, pastes earlier runs' memories into new prompts, and uses OpenAI's embeddings unless you choose an `embedder`. We keep memory off for independent cases and show a local embedder (Step 8).
> - **A "trading" crew that reads search snippets and ends with a sell rating**, with no disclaimer. Our desk works on invented prices, computes every number in pandas, gives no buy, sell or hold view, and code adds the disclaimer (Steps 9 to 14).
> - **`manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)`**: a LangChain model, older and weaker than the workers, with a high temperature. CrewAI has its own `LLM` class; we choose the manager on purpose (Gemini: `gemini-3.8-flash`) and send no temperature (Step 13).
> - **`IPython.display.Markdown(result)` in a terminal script** prints an object, not formatted text. Print `result.raw`, or use the typed `result.pydantic`.

## Prerequisites

- Finished **Episode 13** (folder `ep13-shelf-sifter-crew`): the CrewAI environment `.venv-crewai`, agents, tasks, tools and logs. Helpful: **Episode 05** (`ep05-model-scorecard`, excessive agency and human approval) and **Episode 11** (`ep11-party-planner-chains`, let Python do the maths).
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15** - CrewAI 1.15 does not install on 3.14 or newer.
- The root `.env` with your **free** Gemini key from Episode 1. For crews we suggest `LLM_MODEL=gemini-3.8-flash` (Step 1).
- Internet for Step 1, for the first run of Steps 3 and 8 (two model downloads), and for every live crew run. All tests and every `--replay` run work offline.
- About 70 minutes, and about 600 MB of disk for the two local models.
- **The models this episode downloads, and their licences:**
  - [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) by Cardiff NLP, licence **CC BY 4.0** (free to use, also commercially, if you credit the authors). We run its ONNX copy [`Xenova/twitter-roberta-base-sentiment-latest`](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest), published on Hugging Face for Transformers.js, about 500 MB (Step 3). Checked 2026-10-26.
  - [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), licence **Apache-2.0**, in Chroma's ONNX copy, about 80 MB (Step 8). Already on your computer if you did Episode 13.
- On the free tier, Google may use your prompts to improve its products. Everything we send here is invented, so that is fine.

## Safety rules for this episode

**Emails: drafts only.**

1. The leads are **invented** businesses and people. Their addresses use `example.com`, a domain reserved for examples ([RFC 2606](https://www.rfc-editor.org/rfc/rfc2606)), so no real inbox can receive them.
2. No tool and no line of code in this project can send an email: no mail server, no password, no network code of its own. `tests/test_approve.py` reads every Python file to prove it.
3. The recipient's address is copied **by code** from the lead file. The model never types an address.
4. Code adds an opt-out line to every draft: "If this is not useful, just reply 'no thanks' and we will not write again."
5. A person reads every draft (`approve.py`). Even an approved draft only becomes a `mailto:` link: your own mail app opens, and **you** press Send, or not.

**Writing to real people?** You need their consent, and you must follow your country's law. In India that is the Digital Personal Data Protection Act, 2023, built on consent and purpose limitation; its Rules were notified on 14 November 2025, and most duties start on 13 May 2027 (Press Information Bureau, [backgrounder of 17 November 2025](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf), checked 2026-10-23). This is general information, not legal advice.

**Research desk: education only.**

1. `DEMO1`, `DEMO2` and `DEMO3` are made-up labels for **invented companies**. Their prices and headlines are generated by a seeded program on your computer: invented sample data, not real companies or prices.
2. No live prices, no broker, no trading API, no orders, no money. The backtest is a **paper** calculation.
3. The desk explains numbers. It never says buy, sell or hold, never gives a price target and never sizes a position. The claims checker flags phrases that sound like advice.
4. Every report starts with a disclaimer that **code** inserts, "Education only. Synthetic data. Not investment advice.", and ends with a second one (invented tickers, invented data, no buy, sell or hold).

## What is in this folder

```text
ep14-crew-office/
├── check_setup.py          Step 1: the CrewAI environment, this episode's packages, your key
├── leads/                  Step 2: three invented businesses (Pune, Kochi, Jaipur)
├── product.md              Step 2: Pickup Parrot, an invented product
├── inputs_check.py         Steps 2, 8: placeholders versus inputs, no blank left
├── tools/
│   ├── tone.py             Step 3: check_tone, a local sentiment model + pushy-phrase list
│   ├── files.py            Steps 4, 12: read_lead_file and read_data_file, with a fence
│   ├── brief_check.py      Step 6: check_against_brief, "no number outside the brief"
│   ├── indicators.py       Step 10: price_indicators (pandas)
│   ├── backtest.py         Step 11: paper_backtest (pandas)
│   └── calculator.py       Step 12: the safe calculator from Episode 11
├── models.py               Steps 4-6, 12: LeadBrief, EmailDraft, Verdict, DeskReport
├── notebook.py             what the tools REALLY did: CaseFile and Ledger
├── outreach_crew.py        Steps 4-6, 8: Crew A, the scorecard, pending drafts
├── excessive_agency.py     Step 7: what a "send" tool would do, and our crew's audit
├── approve.py              Step 7: the human approval gate
├── memory_demo.py          Step 8: crew memory with a local embedder
├── make_prices.py          Step 9: seeded, invented prices and headlines
├── desk_crew.py            Step 12: Crew B, the research note
├── compare_process.py      Step 13: sequential versus hierarchical, measured
├── claims_check.py         Step 14: every number traced to a tool, the disclaimer
├── replay.py               the scripted model behind every --replay run
├── privacy.py              turns off CrewAI's telemetry, traces and update check, and ONNX Runtime's telemetry
├── data/README.md          about the invented data (prices.csv and news.csv are made by Step 9)
├── tests/                  offline tests (scripted model, no key, no internet)
├── llm.py, testing/        shared helpers (synced from _template)
├── requirements.txt        ../requirements-crewai.txt plus pandas, numexpr and the tested pins
├── CHALLENGE.md
├── drafts/                 made by Steps 6-8 (git-ignored): pending/, approved/, rejected/
└── reports/                made by Steps 12-13 (git-ignored)
```

Every file is already here. Type the code yourself as we go, or read along and run it.

**No key yet?** Every crew step has a `--replay` switch: a **scripted** model (from `testing/fakes.py`, driven by `replay.py`) answers with lines we wrote, marked `[scripted]`. CrewAI, the tools, the files and every check are real. The expected outputs below that say "replay" are exact; the live ones show the shape.

## Step 0 - Get ready: the CrewAI environment (about 3 minutes)

Same environment as Episode 13: `.venv-crewai` at the repo root (`ai-engineering-course/`), and the same root `.env`.

### 0.1 Activate `.venv-crewai`

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**). Skipped Episode 13? Create the environment first, as in its Step 0.1 ([SETUP.md](../SETUP.md), part 8).

**Windows (PowerShell):**

```powershell
.\.venv-crewai\Scripts\Activate.ps1
cd ep14-crew-office
```

**macOS / Linux:**

```bash
source .venv-crewai/bin/activate
cd ep14-crew-office
```

Expected: your prompt starts with `(.venv-crewai)` and ends in `ep14-crew-office`.

> [!WARNING]
> **Common errors**
> - `No such file or directory` / `cannot be loaded` - you are not in the repo root, or `.venv-crewai` does not exist yet. Make it as in Episode 13, Step 0.1.
> - Windows: `running scripts is disabled on this system` - `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`, then activate again.
> - Your prompt shows `(.venv)` - that is the LangChain environment. Run `deactivate`, then activate `.venv-crewai`.

### 0.2 Install this episode's two new packages

```bash
pip install -r requirements.txt
pip check
```

New in this episode: `pandas` 2.2.3 (the version CrewAI's own pandas extra uses) and `numexpr` 2.14.2 (for the Episode 11 calculator). On our Episode 13 environment pip added only these and `pytz`. In an environment made fresh from `requirements-crewai.txt` (we did that on 2026-10-26), pip also swaps two packages to the versions this course tested, which is fine:

```text
  Attempting uninstall: numpy
    Found existing installation: numpy 2.5.4
...
  Attempting uninstall: huggingface_hub
    Found existing installation: huggingface_hub 2.3.0
...
Successfully installed huggingface_hub-2.2.0 numexpr-2.14.2 numpy-2.4.6 pandas-2.2.3 pytz-2026.5
```

Expected (last line):

```text
No broken requirements found.
```

> [!WARNING]
> **Common errors**
> - `Requires-Python <3.14` or `No matching distribution found for crewai==1.15.28` - the environment was made with Python 3.14 or 3.15. Make `.venv-crewai` again with Python 3.13.
> - Why no PyTorch? Both local models run in their ONNX versions with `onnxruntime`, which CrewAI already installs. (PyTorch and transformers *would* install here too, a dry run on 2026-10-26 changed no existing package, but they add several hundred MB for the same scores.)

### 0.3 Check that your key is found from this folder

```bash
python llm.py
```

Expected with your key: `Settings file: /.../ai-engineering-course/.env`, the provider and model, and one short reply from Gemini (your words will differ). Without a key it prints the free-key help, as in Step 1.

> [!WARNING]
> **Common errors**
> - `Settings file: no .env found` - the root `.env` is missing or is called `.env.txt` (Windows hides extensions). From the repo root: `Copy-Item .env.example .env` (Windows) or `cp .env.example .env`, then paste your key.
> - `ModuleNotFoundError: No module named 'langchain_google_genai'` from another episode's file - expected here: this environment has CrewAI's own Gemini support. `llm.py` falls back to it by itself.

---

# Part 1 · The outreach office

## Step 1 - Check the CrewAI setup

**Same `.venv-crewai` as Episode 13.** This check proves you are in the CrewAI environment, that this episode's packages are there, that the privacy switches are on, and that Gemini answers through CrewAI's own `LLM` class, the one our agents use.

**Pick the model (suggested).** Crews make several model calls and must follow tool rules carefully, so use Google's smarter free model. In the root `.env`:

```text
LLM_MODEL=gemini-3.8-flash
```

It is slower and uses more of your free quota than the default `gemini-3.5-flash-lite`; everything here also runs with the default.

```bash
python check_setup.py
```

Expected with your key (your patch number, folder name and masked key differ):

```text
AI Engineering From Zero - Episode 14 setup check (the .venv-crewai of Episode 13)

Python .................. 3.13.x OK
Virtual env ............. active (.venv-crewai)
crewai .................. 1.15.28
crewai-tools ............ 1.15.28
pandas .................. 2.2.3
numexpr ................. 2.14.2
onnxruntime ............. 1.31.0
tokenizers .............. 0.23.3
huggingface_hub ......... 2.2.0
chromadb ................ 1.5.9
lancedb ................. 0.30.0
numpy ................... 2.4.6
python-dotenv ........... 1.2.4
pytest .................. 9.1.1
Tone model .............. not yet (Step 3 downloads it once)
CrewAI telemetry ........ off
CrewAI tracing .......... off
ONNX Runtime telemetry .. off
.env file ............... found (ai-engineering-course/.env)
Provider ................ gemini
Model ................... gemini-3.8-flash   (CrewAI name: gemini/gemini-3.8-flash)
API key ................. GEMINI_API_KEY found (AIza...xxxx)

Asking gemini/gemini-3.8-flash one tiny question through crewai.LLM ...
Reply: <one word from Gemini, usually OK>

provider=gemini, model=gemini-3.8-flash, reply OK
```

No key yet? The check stops before any call (real output, Python 3.11 on our machine):

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

No key yet? The crew steps also run offline with a scripted model: add --replay.
```

`python check_setup.py --offline` runs the same checks without the question. Every program here that loads CrewAI or a local model starts with `import privacy`, which switches off CrewAI's telemetry, traces and update check before CrewAI loads (Episode 13, Step 1). New in this episode: it also sets `ORT_DISABLE_TELEMETRY=1`. ONNX Runtime, which runs our two local models, has usage data turned on by default in its official builds and keeps a device id in your user folder; this switch stops both (ONNX Runtime's [Privacy.md](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md), checked 2026-10-26). As in Episode 13, CrewAI keeps the task outputs of your latest run in a small file on your own computer, `latest_kickoff_task_outputs.db`, in a folder named `ep14-crew-office` (Mac: `~/Library/Application Support/ep14-crew-office/`, Windows: `C:\Users\<you>\AppData\Local\CrewAI\ep14-crew-office\`). Delete it whenever you like.

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` (also `pip install "crewai[anthropic]==1.15.28"`) or `ollama`.

> [!WARNING]
> **Common errors**
> - `crewai .................. MISSING` - wrong environment: activate `.venv-crewai`.
> - `pandas .................. MISSING` - Step 0.2 not done: `pip install -r requirements.txt`.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute, or until the daily reset. Your limits: https://aistudio.google.com/rate-limit

## Step 2 - Write the leads and the product sheet

**Invented, local data only.** A mail-merge letter has blanks: "Dear {name}, congratulations on {news}". CrewAI fills `{placeholders}` in the agents' and tasks' text from `kickoff(inputs={...})`. Two quiet mistakes are common: a blank with no value (CrewAI stops with a `ValueError`, good) and a value no blank asks for (CrewAI ignores it in silence, bad). We check both before every run.

The three leads are small Markdown files, one fact per line. `leads/kesar-crumb-bakehouse.md`:

```markdown
# Kesar Crumb Bakehouse

Invented business for a course demo. The business, the person and the email address are not real.

- City: Pune, Maharashtra
- Contact: Nandini Apte, co-founder
- Email: nandini.apte@example.com
- What they sell: eggless cakes, nankhatai biscuits and festival gift boxes
- Outlets: 4, in Baner, Aundh, Kothrud and Viman Nagar
- Recent news: opened the Viman Nagar outlet in August 2026
- How pre-orders work today: customers phone the outlet, and staff copy each order into a paper register
- Busiest week: the week before Diwali, with about 300 gift-box pre-orders
- Problem she mentioned at a Pune bakers' meet-up: some customers came at the wrong pick-up time last Diwali
- Prefers: short emails in English, no attachments
```

The other two are a tuition centre in Kochi and a block-print exporter in Jaipur. `product.md` describes **Pickup Parrot**, an invented service that messages customers when their order, sample or class is ready (open it: 7 lines of facts, a free trial of 30 days, the sender's name).

Our code builds the inputs for each lead from these files (`outreach_crew.py`):

```python
def lead_inputs(lead_file: Path, today: str) -> dict:
    """The kickoff inputs for one lead: exactly the placeholders our tasks use."""
    company = lead_file.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    return {
        "lead_file": lead_file.as_posix(),
        "company": company,
        "sender": field(PRODUCT_FILE, "Sender of our outreach emails"),
        "today": today,
    }
```

And `inputs_check.py` compares them with the crew's text:

```python
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

def check_inputs(crew, inputs: dict) -> tuple[list[str], list[str]]:
    """(missing, unused): placeholders with no input, and inputs no placeholder asks for."""
    wanted = placeholders_in(crew)
    missing = sorted(name for name in wanted if name not in inputs)
    unused = sorted(name for name in inputs if name not in wanted)
    return missing, unused
```

```bash
python inputs_check.py
```

Expected output (your date will differ; it comes from your computer):

```text
Step 2: the leads and the product sheet (all invented, all local files)

  leads/dhoop-chhaon-textiles.md           13 lines
  leads/kayal-learning-circle.md           13 lines
  leads/kesar-crumb-bakehouse.md           14 lines
  product.md                               11 lines

Placeholders in the crew's text:
  {company}    Lead Researcher goal, Outreach Writer goal, Email Reviewer goal, task 1 description, task 2 description, task 3 description
  {lead_file}  task 1 description
  {today}      task 1 description, task 2 description
  {sender}     task 2 description

Inputs we build for each lead (today = 2026-10-26):
  Dhoop Chhaon Textiles      OK, every input used, no blank left
  Kayal Learning Circle      OK, every input used, no blank left
  Kesar Crumb Bakehouse      OK, every input used, no blank left

The classic mistakes, on purpose:
  missing: ['today']   -> CrewAI would stop: Missing required template variable
  unused:  ['budget']  -> CrewAI would ignore it without a word
```

`{today}` is an input on purpose: a model has no calendar. Tell it the date, or it may guess the year its training data ended (Part 2 checks for exactly that).

> [!WARNING]
> **Common errors**
> - `ValueError: ... has no line '- Contact: ...'` - a lead file lost one of its `- Label:` lines. Our code reads `Contact` and `Email`; keep those labels exactly.
> - `kickoff()` with **no** inputs at all does not complain: CrewAI then leaves `{company}` in the text as it is, braces and all (we checked this in 1.15.28). Always pass inputs.

## Step 3 - Build and test the tone tool

**Test tools alone first.** You test a new weighing scale with a known weight before you use it in the shop. A tool is the same: an agent cannot tell a broken tool from a correct one, so we test it with sentences whose tone we already know.

We first tried a famous two-label sentiment model (SST-2, trained on film reviews). Our known weights showed the problem at once: it called `The order is ready for pickup.` **NEGATIVE 0.9968**, and a polite outreach email NEGATIVE 0.9974. A scale that reads "heavy" for everything is useless. A model with a **neutral** label fits business emails better, so `tools/tone.py` uses Cardiff NLP's Twitter-RoBERTa sentiment model (three labels), plus a pushy-phrase list, an exclamation count and a word count:

```python
def tone_report(text: str, sentiment) -> dict:
    overall = sentiment.scores(text)
    cold = []
    for number, sentence in enumerate(sentences(text), start=1):
        negative = sentiment.scores(sentence)["negative"]
        if negative >= COLD_SENTENCE:                        # 0.60: this sentence "reads cold"
            cold.append({"sentence": number, "negative": negative, "text": sentence})
    pushy = [phrase for phrase in PUSHY_PHRASES if phrase in text.lower()]
    ...
    return {"label": label_of(overall), "scores": overall, "cold_sentences": cold,
            "pushy_phrases": pushy, ..., "verdict": "ok" if not reasons else "check", "reasons": reasons}
```

The model runs with `onnxruntime` (no PyTorch). We compared its scores with the Hugging Face `transformers` pipeline on the original model in our other environment: the same to 4 decimals on all 13 test texts (one differed by 0.0001). The limits (0.60 for a sentence, 0.50 for the whole email, 150 words, 1 exclamation mark) are **our** rules; change them if you like.

```bash
python -m tools.tone
```

The first run downloads the model once (about 500 MB into your Hugging Face cache: 85 seconds on our connection). Expected output:

```text
Loading Xenova/twitter-roberta-base-sentiment-latest (ONNX). The first run downloads about 500 MB once ...
Engine: twitter-roberta-sentiment (local ONNX)

label     negative  neutral positive  verdict
positive    0.0017   0.0138   0.9845  ok
  Congratulations on opening your fourth outlet. We would love to help your team handle the festival rush.
neutral     0.4163   0.5018   0.0819  check
  Act now! This offer expires tonight and you will regret missing it.
    - pushy phrases: act now, expires, you will regret
negative    0.9129   0.0806   0.0065  check
  Your staff keep losing pre-orders and that is costing you customers.
    - the whole email reads negative (0.9129)
    - sentence 1 reads cold (0.9129): 'Your staff keep losing pre-orders and that is costing you customers.'
neutral     0.0064   0.5303   0.4632  ok
  The order is ready for pickup.
```

Look at the pushy sentence: the model calls it only **neutral** 0.5018. The word list catches what the model misses. Two simple checks beat one clever one.

No download possible? `python -m tools.tone --words` uses the small, tested word-list fallback (same four sentences: `ok`, `check`, `check`, `ok`).

Now the tests, with known sentences and hand-checked numbers:

```bash
python -m pytest tests/test_tone.py -vv
```

```text
tests/test_tone.py::test_word_list_counts_warm_and_cold_words PASSED     [ 12%]
tests/test_tone.py::test_a_plain_email_passes PASSED                     [ 25%]
tests/test_tone.py::test_pushy_phrases_and_exclamation_marks_are_reasons PASSED [ 37%]
tests/test_tone.py::test_one_cold_sentence_is_named_so_the_reviewer_can_fix_it PASSED [ 50%]
tests/test_tone.py::test_long_emails_are_flagged PASSED                  [ 62%]
tests/test_tone.py::test_sentences_splits_lines_and_full_stops PASSED    [ 75%]
tests/test_tone.py::test_the_tool_returns_json_and_notes_the_call PASSED [ 87%]
tests/test_tone.py::test_the_real_model_on_the_known_sentences PASSED    [100%]

============================== 8 passed in 2.20s ===============================
```

The last test only runs once the model is downloaded (before that it shows as `SKIPPED`).

> **Seen an older tutorial?** Its custom tool's `_run` returned the same word every time, so the agent looped and gave up. A tool must return information that can change the agent's next decision: here, which sentence reads cold and which phrase is pushy.

> [!WARNING]
> **Common errors**
> - `Could not load the sentiment model (LocalEntryNotFoundError)` - not downloaded yet and no internet. Run `python -m tools.tone` once online.
> - `Warning: You are sending unauthenticated requests to the HF Hub` - harmless. A free Hugging Face token (`HF_TOKEN`) only raises download limits.
> - Different scores in the 4th decimal on another computer - normal for floating-point maths on different CPUs. The labels and verdicts stay the same.

## Step 4 - The Researcher, with a scoped file tool

**Typed `LeadBrief` output, every fact with its source line.** CrewAI's ready-made `FileReadTool` reads any path the model writes. Our Researcher needs four files at most, so `tools/files.py` builds a tool with a fence: only `leads/*.md` and `product.md`. Each line comes back with its file and line number, so every fact can carry an exact source.

```python
def resolve_inside(path: str, allowed: list[str], root: Path = PROJECT) -> Path:
    target = (root / path).resolve()            # follows "..", so "leads/../../.env" is caught too
    if not target.is_relative_to(root.resolve()):
        raise PermissionError(f"{path!r} is outside this project folder")
    ...
```

Try the tool by hand, with no agent and no model:

```bash
python -m tools.files leads/kesar-crumb-bakehouse.md
python -m tools.files ../.env
```

```text
Tool name:    read_lead_file
Description:  Read one lead file (leads/<name>.md) or the product sheet (product.md). Returns every line ...

tool.run(path='leads/kesar-crumb-bakehouse.md') returned:
leads/kesar-crumb-bakehouse.md:1: # Kesar Crumb Bakehouse
...
leads/kesar-crumb-bakehouse.md:10: - Recent news: opened the Viman Nagar outlet in August 2026
leads/kesar-crumb-bakehouse.md:11: - How pre-orders work today: customers phone the outlet, and staff copy each order into a paper register
...

Tool name:    read_lead_file
...
tool.run(path='../.env') returned:
  Refused: '../.env' is outside this project folder
```

The name and the description are what the agent reads to decide when to call the tool, so the tool prints them first.

Your API key lives in `../.env`. The Researcher cannot read it, whatever a model or a planted line in a lead file asks for.

The typed answer (`models.py`): a `Fact` is an id, a text and a source; a `LeadBrief` holds exactly 3 facts about the lead plus 1 to 3 product points. There is **no email address** in it: our code copies the recipient from the lead file later.

```python
class Fact(BaseModel):
    id: str = Field(description="F1, F2, F3 ... for lead facts; P1, P2 ... for product points")
    text: str = Field(description="the fact in plain words, taken from ONE line of a file")
    source: str = Field(description="file and line number exactly as read_lead_file shows them, ...")


class LeadBrief(BaseModel):
    company: str = Field(description="the business name from the lead file")
    contact_name: str = Field(description="the contact person's name from the lead file")
    facts: list[Fact] = Field(min_length=3, max_length=3, description="exactly 3 facts about the lead")
    product_points: list[Fact] = Field(min_length=1, max_length=3, ...)
```

The task (`outreach_crew.py`) asks for the fact types and the source format, and a task **callback** gives our code the brief as soon as the task ends:

```python
research_task = Task(
    description="Read the lead file {lead_file} with read_lead_file, then read product.md.\n"
                "Write a brief about {company} for a first email about Pickup Parrot: ..."
                "... its source as '<file>:<line number>' exactly as the tool shows it. "
                "Never add anything the files do not say. Today is {today}.",
    expected_output="A LeadBrief: company, contact_name, 3 facts and 1 to 3 product points, ...",
    agent=agents["researcher"],
    output_pydantic=LeadBrief,      # CrewAI turns the answer into a LeadBrief object
    callback=case.save_brief,       # our code keeps the brief for check_against_brief (Step 6)
)
```

```bash
python outreach_crew.py --stop-after researcher
```

Then **our code checks the agent** (the Episode 13 scorecard idea): does every source point at a line the tool really read, and is every number in the fact on that line? Offline first, with the scripted model (`--replay`, exact output, CrewAI's log boxes cut out here):

```bash
python outreach_crew.py --stop-after researcher --replay
```

```text
REPLAY: the model is SCRIPTED (replay.py). The texts marked [scripted] were written by us;
CrewAI, the tools, the files and every check are real.

Outreach crew: 1 lead(s)   agents: researcher   today: 2026-10-26
Model: scripted replay (no AI model)   Process: sequential   memory: off   max_rpm: 10

  ... Crew Execution Started, Task Started, Agent Started ...
  ... Tool Execution Started (#1)  Tool: read_lead_file  Args: {"path": "leads/kesar-crumb-bakehouse.md"}
  ... Tool Execution Started (#2)  Tool: read_lead_file  Args: {"path": "product.md"}
  ... Agent Final Answer ...
[Finalize] todos_count=0, todos_with_results=0

======================================================================
The Researcher's typed answer: LeadBrief with 3 facts and 3 product point(s)
{
  "company": "Kesar Crumb Bakehouse",
  "contact_name": "Nandini Apte",
  "facts": [
    {
      "id": "F1",
      "text": "Recent news: opened the Viman Nagar outlet in August 2026",
      "source": "leads/kesar-crumb-bakehouse.md:10"
    },
    ... F2 (line 11) and F3 (line 13) ...
  ],
  "product_points": [
    ... P1 (product.md:5), P2 "Set-up: about 15 minutes, no new hardware" (product.md:7),
        P3 "Free trial: 30 days, no card needed" (product.md:8) ...
  ]
}

Scorecard (kesar-crumb-bakehouse)
  read_lead_file ............... 2 call(s)
  items traced to a line ....... 6 of 6
Usage for kesar-crumb-bakehouse: requests 0   prompt tokens 0   completion tokens 0   reasoning tokens 0   total 0
(A scripted model reports no tokens, so the usage is 0. A live run shows real numbers here.)

All checks passed. No draft was saved: this run stopped after the researcher.
```

The `[Finalize] ...` line comes from CrewAI's own log (one per agent); you can ignore it. With your key (no `--replay`), Gemini picks the facts and words them itself: expect the same shape, `read_lead_file ... 2 call(s)`, your own fact texts, and `items traced to a line ... 6 of 6` (or a `PROBLEM:` line that names the fact whose source is wrong). With a real Gemini model CrewAI uses Gemini's own tool calling, not the text format of the scripted model; this live path is still to be recorded for the video.

> [!WARNING]
> **Common errors**
> - `The crew stopped with an error: ToolExecutionFailedError ... Refused:` - the model asked for a file outside the fence (`tool_failure_policy="raise"` stops the crew, as in Episode 13). Read the path in the message.
> - `PROBLEM: F2: source 'leads/kesar-crumb-bakehouse.md:99' is not a line the file tool read` or `PROBLEM: F2: ['400'] not on line leads/kesar-crumb-bakehouse.md:12` - the model made up a line number or a number. That is what the scorecard is for. Run again, or use `LLM_MODEL=gemini-3.8-flash`.
> - `PROBLEM: the Researcher's answer did not fit LeadBrief` - for example 4 facts instead of 3. Read its Final Answer box.

## Step 5 - The Writer, with context

**`context=[research_task]` and a typed `EmailDraft`.** A journalist may only write what the researcher's notes support. `context` pastes the brief into the Writer's prompt: that is the hand-off, visible and testable. The Writer has **no tools**: it only writes.

```python
write_task = Task(
    description="Write a first email from Pickup Parrot to the contact person of {company}, "
                "using ONLY the brief.\n"
                "1. 120 words or fewer, plain text, no exclamation marks.\n"
                "2. Use one or two lead facts and at least one product point, and list the ids "
                "you used in facts_used.\n"
                "3. Every number you write must appear in the brief.\n"
                "4. No pressure: no deadlines, no 'act now'. End with one simple question.\n"
                "5. Sign off as: {sender}\n"
                "6. Do not write an email address, a link or an unsubscribe line: our code adds "
                "the recipient and the opt-out line.\n"
                "Today is {today}; do not invent dates.",
    expected_output="An EmailDraft: subject, body and facts_used.",
    agent=agents["writer"],
    context=[research_task],        # the brief is pasted into this task's prompt: the hand-off
    output_pydantic=EmailDraft,
)
```

```bash
python outreach_crew.py --stop-after writer --replay --quiet
```

`--quiet` hides CrewAI's log. Exact output after the brief:

```text
Scorecard (kesar-crumb-bakehouse)
  read_lead_file ............... 2 call(s)
  items traced to a line ....... 6 of 6
  numbers outside the brief .... 0

The Writer's typed answer: EmailDraft (facts used: F1, F2, P1, P2, P3)
Subject: [scripted] Pick-up messages for Kesar Crumb's festival boxes

Dear Nandini,

Congratulations on opening the Viman Nagar outlet in August 2026. When festival orders come in by phone and go into a paper register, keeping every pick-up time clear is a big job.

Pickup Parrot sends each customer a short message when their order is ready, and keeps one shared list of pre-orders. Set-up takes about 15 minutes, and the free trial runs for 30 days with no card needed.

Would it help to try it at one outlet before Diwali?

Warm regards,
Neha Rao
Partnerships, Pickup Parrot

Usage for kesar-crumb-bakehouse: requests 0   prompt tokens 0   completion tokens 0   reasoning tokens 0   total 0
(A scripted model reports no tokens, so the usage is 0. A live run shows real numbers here.)

All checks passed. No draft was saved: this run stopped after the writer.
```

Every number in it (2026, 15, 30) is in the brief, so `numbers outside the brief .... 0`. Live, Gemini writes its own email in this shape; the scorecard line tells you at once whether it added a number.

> [!WARNING]
> **Common errors**
> - `numbers outside the brief .... 1 ['40%']` - the model "improved" the email with a statistic nobody gave it. Step 6 shows the Reviewer catching it.
> - `PROBLEM: links or addresses in the draft: [...]` - the Writer typed an address or a link against rule 6. Our code adds the recipient; ask for a fix or reject the draft in Step 7.

## Step 6 - The Reviewer

**The tone tool plus a "no fact outside the brief" check.** The Reviewer gets two tools and both earlier answers (`context=[research_task, write_task]`), and returns a typed `Verdict`: `pass`, or `fix` with a list of fixes. The brief check is plain Python (`tools/brief_check.py`): every number in the draft must appear in the brief, and no link or address may appear. It reads the brief from our notebook (saved by the task callback), so the model cannot hand it a different one.

```python
def check_against_brief(text: str, brief: LeadBrief) -> dict:
    outside = unsupported_numbers(text, brief_numbers(brief))
    links = LINK_OR_ADDRESS.findall(text)
    return {"numbers_in_draft": find_numbers(text), "numbers_outside_brief": outside,
            "links_or_addresses": links, "ok": not outside and not links}
```

First with one planted number: `--plant` makes the scripted Writer add "Bakeries that use Pickup Parrot see 40% fewer missed pick-ups."

```bash
python outreach_crew.py --replay --plant --quiet
```

Exact output (the end):

```text
  numbers outside the brief .... 1 ['40%']
  PROBLEM: numbers not in the brief: ['40%']
  ...
The Reviewer's typed answer: {
  "status": "fix",
  "tone": "[scripted] positive 0.7084, ok",
  "numbers_outside_brief": [
    "40%"
  ],
  "fixes": [
    "[scripted] remove 40%: it is not in the brief"
  ]
}
  check_tone ................... 1 call(s)
  check_against_brief .......... 1 call(s)
  Reviewer says ................ fix; our checks say fix

Saved drafts/pending/kesar-crumb-bakehouse.json (a DRAFT, waiting for a human: nothing was sent)
...
1 problem(s) found: read the scorecard before you approve anything.
```

The tone is fine (`positive 0.7084`); the claim is not. Now the clean run, with CrewAI's full log:

```bash
python outreach_crew.py --replay
```

In the log, find the two Reviewer tool calls. The tone box shows what the real local model said about this draft:

```text
│  Tool: check_tone                                                            │
│  Output: {"label": "positive", "scores": {"negative": 0.0059, "neutral":     │
│  0.2734, "positive": 0.7208}, "engine": "twitter-roberta-sentiment (local    │
│  ONNX)", "cold_sentences": [], "pushy_phrases": [], "exclamation_marks": 0,  │
│  "words": 89, "verdict": "ok", "reasons": []}                                │
```

and the end:

```text
The Reviewer's typed answer: {
  "status": "pass",
  "tone": "[scripted] positive 0.7208, ok",
  "numbers_outside_brief": [],
  "fixes": []
}
  check_tone ................... 1 call(s)
  check_against_brief .......... 1 call(s)
  Reviewer says ................ pass; our checks say pass

Saved drafts/pending/kesar-crumb-bakehouse.json (a DRAFT, waiting for a human: nothing was sent)
```

`Reviewer says ... our checks say ...` is our code checking the checker: the Reviewer is an agent too, so we recompute both checks in Python and compare. The saved draft's recipient comes from the lead file, and our code appended the opt-out line.

Live with `python outreach_crew.py`: the same shape, Gemini's words. If Gemini's email mentions the customers' problem bluntly, `check_tone` may name that sentence as cold and the verdict becomes `fix`: useful information for you in Step 7.

> [!WARNING]
> **Common errors**
> - `PROBLEM: the Reviewer gave a verdict without calling both tools` - it answered from its impression. The draft still waits for you; run again.
> - `Reviewer says pass; our checks say fix   PROBLEM: they disagree` - trust the tools, not the verdict text.

## Step 7 - The human approval gate

**Drafts only.** A junior writes the letter; the manager signs and posts it. This is the Episode 05 lesson in code: OWASP's Top 10 for LLM Applications (2025) lists **LLM06, Excessive Agency**, giving a model more tools and permissions than its job needs, and suggests the fewest tools and a human approval for high-impact actions ([genai.owasp.org](https://genai.owasp.org/llm-top-10/), checked 2026-10-23).

First, what excessive agency would look like. `excessive_agency.py` gives a Writer a `send_email` tool: a harmless stand-in that only writes the call into a Python list.

```bash
python excessive_agency.py
```

Exact output (the model is scripted to do what any model may do once it holds such a tool):

```text
PART 1 - a Writer WITH a send tool (scripted model, stand-in tool, nothing leaves this computer)

  The agent's final answer:     [scripted] Done, the email has been sent.
  send_email calls:             1
  Human approvals before it:    0
  Address the model typed:      nandini@kesarcrumb.example   (the lead file says nandini.apte@example.com)
  What it promised:             [scripted] Bakeries like yours get 50% more orders with Pickup Parrot. Act now!
  With a real send tool, that was a real email: wrong address, a made-up 50%, and pressure.

PART 2 - an audit of OUR outreach crew (the real agents and tools, no model is called)

  Lead Researcher    tools: read_lead_file
  Outreach Writer    tools: (no tools)
  Email Reviewer     tools: check_tone, check_against_brief

  Tools that could send an email: 0
  The send step lives OUTSIDE the crew: approve.py (a human decides), then your own mail app
  (you press Send). The model never types the address: our code copies it from the lead file.
```

Now the gate, `approve.py`. It shows each pending draft and waits for you:

```python
choice = ask("[a]pprove  [e]dit  [r]eject  [s]kip ? ").strip().lower()[:1]
if choice == "e":
    draft = edit(path, draft, ask, drafts)        # you change a text file, then decide
    ...
if choice == "a":
    saved = approve(path, draft, drafts)          # writes drafts/approved/<lead>.md - and sends NOTHING
    print(mailto_link(draft))                     # opens YOUR mail app with the email filled in
```

```bash
python approve.py
```

Expected output, after the Step 6 run, when you type `a` (exact; the time is yours, and the long link is shortened here):

```text
1 draft(s) waiting in drafts/pending/ (written by outreach_crew.py)

======================================================================
Draft 1 of 1: kesar-crumb-bakehouse
To:        Nandini Apte <nandini.apte@example.com>   (from leads/kesar-crumb-bakehouse.md, not from the model)
Subject:   [scripted] Pick-up messages for Kesar Crumb's festival boxes
Written:   scripted replay (no AI model), 2026-10-26T14:54:34
----------------------------------------------------------------------
Dear Nandini,
...
Warm regards,
Neha Rao
Partnerships, Pickup Parrot

If this is not useful, just reply 'no thanks' and we will not write again.
----------------------------------------------------------------------
Reviewer:  pass   tone: [scripted] positive 0.7208, ok   fixes: none
[a]pprove  [e]dit  [r]eject  [s]kip ? a
Approved. Saved drafts/approved/kesar-crumb-bakehouse.md
Click this link to open it in your mail app; you press Send there, or not at all:
mailto:nandini.apte@example.com?subject=%5Bscripted%5D%20Pick-up%20messages%20for%20Kesar%20Crumb%27s%20festival%20boxes&body=Dear%20Nandini%2C%0A%0A...

Saved 1 draft. Nothing was sent.
```

`e` writes the draft to `drafts/editing/<lead>.txt`: change it in VS Code, save, press Enter, then approve, reject or skip (a skipped draft keeps your edit in `drafts/pending/`). After an edit our code checks the numbers again (people make mistakes too) and puts the opt-out line back if you deleted it. The address `nandini.apte@example.com` is reserved for examples, so even if you press Send in your mail app, no real person receives it.

> [!WARNING]
> **Common errors**
> - `No drafts waiting in drafts/pending/` - run Step 6 first.
> - The `mailto:` link opens nothing - your computer has no default mail app. Copy the subject and body from `drafts/approved/<lead>.md` instead.
> - Looking for an "auto-approve" switch? There is none, on purpose.

## Step 8 - Run it for every lead

**`kickoff_for_each`, and crew memory with a local embedder.** One crew, three leads: `kickoff_for_each(inputs=[...])` runs the same crew once per input dictionary.

```python
results = crew.kickoff_for_each(inputs=inputs)   # one run per lead, same crew
...
# CrewAI 1.15.28 gives each kickoff_for_each run a running TOTAL (we measured 1, 2, 3 requests
# for three one-request runs), so we subtract the previous total: CrewAI's own delta_since().
usage = result.token_usage if previous is None else result.token_usage.delta_since(previous)
```

That comment is a real finding: CrewAI copies the crew for each run, and the copies of each agent's model share one usage counter, so the second lead reports leads 1 + 2. `tests/test_crews_fake.py` pins this behaviour with pretend token numbers (outputs `7, 14, 21` requests for three 7-request runs; after `delta_since`: `7, 7, 7`), so we notice if a new CrewAI version changes it.

```bash
python outreach_crew.py --all --replay --quiet
```

Exact output (the end; each lead's brief, draft and verdict print above it):

```text
Summary (3 leads, one kickoff_for_each)
  lead                     traced   outside  reviewer  requests  tokens  draft
  dhoop-chhaon-textiles    6 of 6   0        pass      0         0       drafts/pending/dhoop-chhaon-textiles.json
  kayal-learning-circle    6 of 6   0        pass      0         0       drafts/pending/kayal-learning-circle.json
  kesar-crumb-bakehouse    6 of 6   0        pass      0         0       drafts/pending/kesar-crumb-bakehouse.json
Usage for all leads: requests 0   prompt tokens 0   completion tokens 0   reasoning tokens 0   total 0
(A scripted model reports no tokens, so the usage is 0. A live run shows real numbers here.)

All checks passed. Next: a human reads each draft:  python approve.py
```

Live (`python outreach_crew.py --all`), each lead also gets its own line `Usage for <lead>: requests <n> ...`. Expect about 7 requests per lead (Researcher 3, Writer 1, Reviewer 3), sometimes one more when CrewAI has to ask again to fit an answer into the shape. We checked the counting on CrewAI's real Gemini class with Google's replies replaced by canned ones (no real Gemini answer): one lead sent 7 requests, `--all` sent 21, and the per-lead lines said 7, 7 and 7. Your token numbers come from your real run.

**Crew memory, briefly.** A shared office notebook the next shift can search. In CrewAI 1.15.28 (read from the installed code, 2026-10-26), `Crew(memory=True)`:

1. **before every task** searches memory with the task text and pastes up to 5 "Relevant memories" into the agent's prompt (a long search text costs a model call to plan the search),
2. **after every task** makes extra model calls: one pulls short "memories" out of the answer, one more per memory files it (scope, categories, importance),
3. turns each memory into numbers with an embeddings model and stores it (LanceDB, in a `memory` folder in CrewAI's data folder on your computer),
4. gives every agent two extra tools, `Search memory` and `Save to memory`.

Its embeddings model is OpenAI's `text-embedding-3-large`, which needs an OpenAI key, **unless** you choose one: `Crew(memory=True, embedder={"provider": "onnx"})` uses the free local all-MiniLM-L6-v2 of Episode 13. Our outreach crew keeps `memory=False`, and `memory_demo.py` shows why, using CrewAI's memory store directly with the local embedder and **no model at all**:

```bash
python memory_demo.py
```

The first run downloads the MiniLM model once (about 80 MB, with a progress bar after step 1) unless Episode 13 already did. Exact output:

```text
Crew(memory=True, embedder={'provider': 'onnx'}) builds:
  Memory with storage LanceDBStorage, scope /crew/crew, embedder ONNXMiniLM_L6_V2
  analysis model: the first agent's model (ScriptedLLM here, Gemini in a real run)
  ... and without embedder=...: embedder None -> OpenAI's default, which needs an OpenAI key

1. Save each lead's facts as memories, one scope per lead (scope, categories and
   importance given by us, so no model is needed):
   /lead/dhoop-chhaon-textiles    3 memories
   /lead/kayal-learning-circle    3 memories
   /lead/kesar-crumb-bakehouse    3 memories

2. The run for kayal-learning-circle searches memory: 'parents or customers miss a time that changed'
   0.658  kayal-learning-circle    Problem he wrote on the centre's notice board: parents miss classes th
   0.637  kayal-learning-circle    How reminders work today: the founder sends each class change to paren
   0.622  kesar-crumb-bakehouse    Problem she mentioned at a Pune bakers' meet-up: some customers came a   <- ANOTHER lead's fact

3. The same search inside one lead's scope:
   0.658  Problem he wrote on the centre's notice board: parents miss classes th
   0.637  How reminders work today: the founder sends each class change to paren
   0.585  Recent news: started weekend online batches in June 2026

Memory helps when later work SHOULD build on earlier work. For independent cases, like one
email per lead, it can carry one customer's details into another's email. We keep it off.
```

The bakery's Diwali problem came back in the tuition centre's search. With `memory=True`, CrewAI runs a search like this before every task and pastes the top 5 results into the prompt, so it could end up in the wrong email. (The score mixes meaning, recency and importance, so it is not a plain similarity.) Then `python approve.py` again for the three drafts: we typed `a`, `r` and `s` and got `Saved 1 draft. Nothing was sent.` and `(1 rejected, 1 still pending)`.

> [!WARNING]
> **Common errors**
> - `429 RESOURCE_EXHAUSTED` halfway through `--all` - three leads are about 21 requests; `max_rpm=10` spreads them out, but the free tier also has a daily limit. Wait, then run the missing lead with `--lead <name>`.
> - The second lead's usage looks twice as big - you printed `result.token_usage` of a `kickoff_for_each` run without `delta_since()`. See above.
> - `RuntimeError: Memory requires an embedder for vector search but initialization failed: The CHROMA_OPENAI_API_KEY environment variable is not set.` - memory is on without an `embedder`, so CrewAI reached for OpenAI. Add `embedder={"provider": "onnx"}`.

---

# Part 2 · The research desk (education only)

## Step 9 - Generate synthetic prices

**Seeded, fake data for learning.** Real prices change daily, belong to data vendors and tempt people to treat a lesson as a tip. Invented prices are the same on every computer, so your numbers match the video. `make_prices.py` moves each price by a small random percentage every weekday (a random walk), adds a few jumps, and writes an invented headline for each jump:

```python
def make_prices(seed: int = SEED, last_day: str = LAST_DAY, days: int = DAYS):
    dates = pd.bdate_range(end=last_day, periods=days)          # weekdays only
    rng = np.random.default_rng(seed)                           # seed 14: same numbers everywhere
    for ticker, (_, first_price, drift, wobble) in COMPANIES.items():
        changes = rng.normal(drift, wobble, size=days)          # one random % change per day
        ...
        closes = first_price * np.cumprod(1 + changes)
```

```bash
python make_prices.py
```

Exact output:

```text
prices.csv, 3 tickers x 250 days   (2025-11-10 to 2026-10-23)
news.csv, 5 invented headlines
Invented sample data, not real companies or prices.

  DEMO1  Invented Solar Pumps Co.     first close  420.00   last close  519.55
  DEMO2  Invented Tea Trucks Co.      first close  180.00   last close  228.48
  DEMO3  Invented Cloud Kitchens Co.  first close   95.00   last close   43.79
```

Each file starts with the label line `# invented sample data, not real companies or prices. Made by make_prices.py, seed 14.` The last day is fixed (2026-10-23), so the data stays the same whenever you run it; today's date is passed to the crew separately.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'pandas'` - Step 0.2.
> - Different numbers - you changed `SEED`, `DAYS` or `LAST_DAY`. Every number in Part 2 changes with them, which is fine, but your output will not match this page.

## Step 10 - The indicator tool

**Returns, 20- and 50-day averages, volatility and max drawdown, in pandas.** A cricket commentator reads the scoreboard; he does not guess the run rate. The Analyst is the commentator, `tools/indicators.py` is the scoreboard:

```python
def max_drawdown(close: pd.Series) -> tuple[float, str, str]:
    """The worst fall from a previous high, in %, with the dates of that high and of the low."""
    running_high = close.cummax()
    drawdown = close / running_high - 1                 # 0 at a new high, negative below it
    trough = drawdown.idxmin()
    peak = close.loc[:trough].idxmax()
    return pct(drawdown.min()), str(peak.date()), str(trough.date())


def annual_volatility(close: pd.Series) -> float:
    """Standard deviation of daily returns x square root of 252, in %."""
    return pct(close.pct_change().std() * math.sqrt(TRADING_DAYS_PER_YEAR))
```

(252 trading days a year is a common convention; ours has 250 invented weekdays.) Every number the tool returns is written into a `Ledger`, the evidence for Step 14.

```bash
python -m tools.indicators DEMO1
```

Exact output:

```text
DEMO1  (invented sample data, not real companies or prices)
2025-11-10 to 2026-10-23, 250 trading days

  first_close ...................... 420.0
  last_close ....................... 519.55
  period_return_pct ................ 23.7
  return_last_20_days_pct .......... 14.98
  average_20_days .................. 490.06
  average_50_days .................. 459.55
  volatility_pct_per_year .......... 26.82
  trading_days_per_year_used ....... 252
  max_drawdown_pct ................. -19.83
  max_drawdown_from_high_on ........ 2026-05-12
  max_drawdown_low_on .............. 2026-07-30
  biggest_up_day ................... +5.09% on 2026-03-18
  biggest_down_day ................. -8.73% on 2026-06-15
```

The biggest down day, 2026-06-15, is the day of an invented recall headline in `data/news.csv`. `DEMO2` and `DEMO3` give different stories (DEMO3: `period_return_pct -53.91`, `max_drawdown_pct -63.61`, `volatility_pct_per_year 40.89`). `tests/test_indicators.py` checks the maths on tiny series by hand, for example `100, 120, 90, 130, 104` has a max drawdown of exactly -25.0% (120 down to 90).

> [!WARNING]
> **Common errors**
> - `prices.csv not found. Make it first:  python make_prices.py` - Step 9.
> - `unknown ticker 'AAPL'` - by design: the tool knows only the invented tickers in `data/prices.csv`.

## Step 11 - The paper backtest tool

**A moving-average crossover versus buy-and-hold, on synthetic data.** The rule: be in the stock while the 20-day average is above the 50-day average, otherwise stay in cash. The baseline: be in the stock every day. Two details honest backtests need:

```python
above = fast_avg > slow_avg                           # True on days the short average is higher
window = close.iloc[slow - 1:]                        # both start on the first day the slow average exists
daily = window.pct_change().iloc[1:]
in_market = above.iloc[slow - 1:].shift(1).iloc[1:].astype(bool)   # decided at YESTERDAY's close: no peeking
rule_growth = (1 + daily.where(in_market, 0.0)).prod()  # cash days earn 0%
```

`tests/test_backtest.py` proves the "no peeking" rule: on `10, 10, 10, 20` the price doubles on the last day, and the rule (which only sees the jump at that day's close) correctly earns 0%, not 100%.

```bash
python -m tools.backtest DEMO1
```

Exact output:

```text
DEMO1  PAPER backtest on synthetic data (invented sample data, not real companies or prices)
Rule: in the stock while the 20-day average is above the 50-day average, else cash
Test: 2026-01-16 to 2026-10-23, 200 days, no trading costs

  Crossover rule (paper) ...... +10.86%
  Buy-and-hold (paper) ........ +12.27%

  Difference: -1.41 percentage points.  Days in the stock: 118 of 200.  Switches in or out: 5.

Synthetic data, paper only: not a forecast, not advice.
```

Try the others: on `DEMO2` the rule made -2.02% against +5.76%; on `DEMO3` -9.26% against -46.29%. Worse twice, much better once, on three invented years: a lesson about testing, not a verdict on the rule.

> [!WARNING]
> **Common errors**
> - `need more than 50 prices` - the 50-day average needs 50 days before the test can start.
> - A rule that looks brilliant - check for peeking first (the `shift(1)`), then remember the missing trading costs.

## Step 12 - The sequential research desk

**Agents quote tool numbers exactly; today's date is passed in.** Three agents, one after the other: the Market Analyst (`price_indicators`, `read_data_file` for the headlines, `calculator`), the Strategy Explainer (`paper_backtest`, `calculator`) and the Risk Reviewer, who returns a typed `DeskReport` (what moved, what the numbers mean, the paper backtest, the risks). There is no buy, sell or hold field. Every task ends with the same rules:

```python
NUMBER_RULES = ("Rules: copy every number exactly as a tool gives it, written as digits. If you need a "
                "number no tool gave, get it from the calculator tool. Never say whether anyone should "
                "buy, sell or hold, and never give a price target or a position size.")
```

The `calculator` is Episode 11's safe calculator (no `eval`, only digits and `+ - * / ( )`), copied into `tools/calculator.py` because Episode 11's file imports LangChain, which lives in the other environment. Its answers go into the Ledger too.

Then our code writes `reports/DEMO1.md`: a header, the agents' text between two invisible markers, a table of the tool numbers (copied by code), and the disclaimer on top and at the end.

```bash
python desk_crew.py --ticker DEMO1 --replay
```

Exact output (the end; CrewAI's log boxes come first; the seconds depend on your computer):

```text
======================================================================
Saved reports/DEMO1.md and reports/DEMO1.numbers.json
Tool calls: price_indicators, read_data_file, paper_backtest
Usage: requests 0   total tokens 0   seconds 0.1

Numbers in the model's text ..... 23
Not from any tool ............... 0
Dates or years not from a tool .. 0
Advice phrases .................. 0
Disclaimer ...................... present

Result: 0 unsupported numbers, disclaimer present

Read it:  reports/DEMO1.md   Check it again any time:  python claims_check.py reports/DEMO1.md
```

And the start of `reports/DEMO1.md` (exact; the scripted agents' sentences say `[scripted]`):

```text
# DEMO1 research note (invented company: Invented Solar Pumps Co.)

> **Education only. Synthetic data. Not investment advice.**

| | |
|---|---|
| Data | 2025-11-10 to 2026-10-23, 250 trading days, invented sample data |
| Today | 2026-10-26 (passed in by our code) |
| Written by | scripted replay (no AI model), sequential crew |

<!-- model text starts -->
## What moved

[scripted] DEMO1 closed at 519.55 on 2026-10-23, up 23.7% from 420.0 on 2025-11-10. Its biggest one-day fall was -8.73% on 2026-06-15, the day of an invented headline: 'DEMO1 recalls one pump model for a safety check'. ...
```

Live (`python desk_crew.py --ticker DEMO1`): Gemini writes the four sections in its own words; the numbers must still come from the tools, and the last lines tell you whether they did. Expect about 6 requests (Analyst 3, Strategy Explainer 2, Risk Reviewer 1), more if an agent uses the calculator: our canned check on CrewAI's real Gemini class sent exactly 6, and never a `temperature`.

> [!WARNING]
> **Common errors**
> - `The desk stopped with an error: ValueError: the Risk Reviewer's answer did not fit DeskReport` - for example only one risk. Run again.
> - `Not from any tool ... 1 ['18.5%']` - an agent did maths in its head or rounded oddly. Step 14 explains the check.
> - `Today` in the report is your computer's date. Use `--today 2026-10-23` to pin it (for example, to compare with this page).

## Step 13 - The hierarchical variant, measured

**An explicit manager model; measure the cost.** Sequential is an assembly line. Hierarchical is a supervisor who hands out every job and reviews it: CrewAI adds a manager agent with two tools, `Delegate work to coworker` and `Ask question to coworker`. More control, more model calls. So we run the same desk both ways and count (`compare_process.py`):

```python
def manager_agent(llm) -> Agent:          # in desk_crew.py
    return Agent(role="Desk Manager", goal="Get the research note on {ticker} done by the right desk member, ...",
                 backstory="... You never write numbers yourself and never give advice.",
                 llm=llm, allow_delegation=True, max_iter=6, verbose=False)

MANAGER_MODELS = {"gemini": "gemini-3.8-flash"}     # at least as strong as the workers; no temperature
```

Only the manager may delegate (every worker has `allow_delegation=False`), and `max_iter` and `max_rpm` cap every agent. Tool calls are counted from CrewAI's own events (`ToolUsageFinishedEvent`).

> **Status (2026-10-26):** the hierarchical process with a real Gemini manager has not been tested in this course yet; we teach and use the sequential crew. The replay below runs CrewAI's real hierarchical code with scripted models, so it shows the mechanism, not Gemini's real costs.

```bash
python compare_process.py --ticker DEMO1 --replay
```

Exact output (only the `Seconds` column depends on your computer):

```text
Research desk on DEMO1, run twice   today: 2026-10-26   SCRIPTED models (replay.py)

Running the sequential desk ...
Running the hierarchical desk ...

Process       Manager model  LLM calls  Tool calls  Delegations  Tokens          Seconds  Same report quality?
------------  -------------  ---------  ----------  -----------  --------------  -------  ------------------------------
sequential    -              6          3           0            n/a (scripted)  0.0      yes: 0 unsupported, disclaimer
hierarchical  scripted       12         6           3            n/a (scripted)  0.1      yes: 0 unsupported, disclaimer

Reports: reports/DEMO1_sequential.md and reports/DEMO1_hierarchical.md
Scripted runs: the counts show HOW a manager adds calls (one delegation and one answer per
task), not what Gemini would really do. Tokens and seconds only mean something in a live run.
```

Twice the model calls for the same report: the manager made one delegation and one answer per task. Live (`python compare_process.py --ticker DEMO1`), the table fills with Gemini's real requests, tokens and seconds; record them before you decide which process you need.

We also ran both desks on CrewAI's real Gemini class with canned replies (still no real Gemini answer): 6 requests sequential, 12 hierarchical, 3 delegations. Two details showed up there. Because our tasks keep their `agent=`, CrewAI 1.15.28 offers the manager that agent's tools as well as its two delegation tools, and lets it delegate only to that agent: asking for anyone else came back as `Error executing tool. coworker mentioned not found, it must be one of the following options: - risk reviewer`. Remove `agent=` from the tasks and the manager gets only the two delegation tools and may pick any coworker (we checked that with a scripted model): more freedom, and more room for ping-pong.

**Delegation ping-pong.** A manager that is never satisfied can bounce a task back and forth. `max_iter` stops it:

```bash
python compare_process.py --pingpong
```

```text
Delegation ping-pong (scripted manager that is never satisfied), manager max_iter=3:
  delegations made:        3
  manager model calls:     4
  worker model calls:      3
  last manager call:       CrewAI forced a final answer (the cap was reached)
  final answer starts:     'Thought: [scripted] Send it back again'
  The cap stopped the bill, not the problem: the stubborn manager still asked for a tool, and
  CrewAI used that text as the answer. Read the final answer; a cap is not a quality check.
  Without a cap, CrewAI's default max_iter is 25 laps, and every lap costs model calls.
```

> **Seen an older tutorial?** It used `Process.hierarchical` with `manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)` and `allow_delegation=True` on every agent, and never compared it with sequential. Here the manager is a `crewai.LLM` chosen on purpose, only the manager delegates, and the cost is measured.

> [!WARNING]
> **Common errors**
> - `Attribute manager_llm or manager_agent is required` - hierarchical needs a manager.
> - `Manager agent should not have tools` - CrewAI gives the manager its delegation tools itself; do not add your own.
> - The live hierarchical run fails or loops - it is untested in this course (status above). Use the sequential desk.

## Step 14 - The claims check and the disclaimer

**Every number traced to a tool; the disclaimer by code.** A model can write "volatility is about 18.5%" in a calm voice when no tool said 18.5. `claims_check.py` reads only the text between the two markers (the parts a model wrote), finds every number and date, and looks for it in the Ledger saved next to the report. Rounding is fine (12.3 matches a tool's 12.34; "fell 8.2%" matches -8.21). A stand-alone year like 2024 that no tool returned is flagged too: the "frozen calendar" of a model that thinks it is still an older year.

```python
def is_supported(number_text: str, allowed: list[float]) -> bool:
    """True when some tool number, rounded the way the text rounds it, gives this number."""
    value, decimals = as_value(number_text)
    return any(round(abs(v), decimals) == value for v in allowed)


def stamp_disclaimer(markdown: str) -> str:
    """Put the banner on top (under the title) and the footer at the end. Running it twice changes nothing."""
```

Check the report from Step 12:

```bash
python claims_check.py reports/DEMO1.md
```

Exact output:

```text
Checking reports/DEMO1.md against 21 numbers and 11 dates from reports/DEMO1.numbers.json

Numbers in the model's text ..... 23
Not from any tool ............... 0
Dates or years not from a tool .. 0
Advice phrases .................. 0
Disclaimer ...................... present

Result: 0 unsupported numbers, disclaimer present
```

Now watch it catch one. `python desk_crew.py --ticker DEMO1 --replay --plant --quiet` makes the scripted Analyst "guess" the volatility as 18.5% instead of the tool's 26.82%. The check of that report:

```text
Numbers in the model's text ..... 23
Not from any tool ............... 1 ['18.5%']
Dates or years not from a tool .. 0
Advice phrases .................. 0
Disclaimer ...................... present

Result: 1 unsupported numbers, disclaimer present
```

`python claims_check.py --demo` shows the same on a five-line report, no files needed. Run Step 12 again without `--plant` to get the clean report back.

Honest limits: the checker reads **digits** ("twelve percent" slips through, so the tasks ask for digits), and its advice list is a crude word list (read every hit yourself). It proves where numbers came from; it does not prove the explanation is wise.

> [!WARNING]
> **Common errors**
> - `Missing reports/DEMO1.md or reports/DEMO1.numbers.json` - run Step 12 first.
> - `PROBLEM: no model text found between the markers` - the report was edited by hand or not made by `desk_crew.py`.
> - `Advice phrases .. 1 ['strong buy']` - the model slipped into advice language. The report is not safe to share; run again.

## Step 15 - Test it without internet

**Both crews end to end with `FakeCrewAILLM`.** The tests replace Gemini with the scripted model; everything else is real.

```bash
python -m pytest tests/test_crews_fake.py -vv
```

```text
collecting ... collected 18 items

tests/test_crews_fake.py::test_one_lead_end_to_end_saves_a_pending_draft PASSED [  5%]
tests/test_crews_fake.py::test_the_writer_sees_the_brief_and_the_reviewer_sees_the_draft PASSED [ 11%]
tests/test_crews_fake.py::test_a_made_up_number_is_caught_and_the_draft_waits_with_a_fix PASSED [ 16%]
tests/test_crews_fake.py::test_kickoff_for_each_writes_one_draft_per_lead PASSED [ 22%]
tests/test_crews_fake.py::test_each_lead_gets_its_own_page_in_the_case_file PASSED [ 27%]
tests/test_crews_fake.py::test_kickoff_for_each_reports_a_running_total_so_we_subtract PASSED [ 33%]
tests/test_crews_fake.py::test_the_scorecard_catches_a_made_up_line_and_a_made_up_number PASSED [ 38%]
tests/test_crews_fake.py::test_the_researcher_cannot_read_your_env_file PASSED [ 44%]
tests/test_crews_fake.py::test_the_inputs_match_the_placeholders PASSED  [ 50%]
tests/test_crews_fake.py::test_each_agent_gets_its_own_llm_object PASSED [ 55%]
tests/test_crews_fake.py::test_without_a_key_every_crew_prints_the_free_key_help PASSED [ 61%]
tests/test_crews_fake.py::test_excessive_agency_replay_shows_the_unapproved_send PASSED [ 66%]
tests/test_crews_fake.py::test_the_desk_writes_a_checked_report_with_the_disclaimer PASSED [ 72%]
tests/test_crews_fake.py::test_a_guessed_number_in_the_desk_report_is_caught PASSED [ 77%]
tests/test_crews_fake.py::test_hierarchical_adds_a_manager_and_its_calls PASSED [ 83%]
tests/test_crews_fake.py::test_max_iter_caps_a_delegation_ping_pong PASSED [ 88%]
tests/test_crews_fake.py::test_the_desk_has_no_trading_tool PASSED       [ 94%]
tests/test_crews_fake.py::test_shared_memory_carries_one_leads_facts_into_another_run PASSED [100%]

============================== 18 passed in 2.91s ==============================
```

Then everything (the tone, file, indicator, backtest, claims and approval tests too):

```bash
python -m pytest
```

```text
.........................................................s..ssss........ [ 92%]
......                                                                   [100%]
73 passed, 5 skipped in 5.44s
```

The 5 `s` are **skipped** tests, not failures: they check LangChain model packages, which live in the other environment. You see 6 or 7 skipped if a local model is not downloaded yet (Steps 3 and 8 download them). `tests/test_approve.py` also reads every Python file of the project and fails if any of them imports a mail or web library of its own, or loads CrewAI or a local model before `privacy`.

> [!WARNING]
> **Common errors**
> - `Tests must not use the internet` - a test reached for a real model. Pass `llm=ScriptedLLM(...)`.
> - `FakeCrewAILLM ran out of scripted responses` - CrewAI asked the model once more than the script has (for example, you added a tool call). Add one reply in `replay.py`.

## Fix-It Corner

| You see | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'crewai'` or `'pandas'` | Activate `.venv-crewai`, then `pip install -r requirements.txt` (Step 0). |
| `Error importing native provider: Missing credentials ... OPENAI_API_KEY` | An `Agent(...)` without `llm=` somewhere: CrewAI fell back to OpenAI. Our builders always pass `llm=`. |
| `RuntimeError: Memory requires an embedder for vector search ... CHROMA_OPENAI_API_KEY ... is not set` | Memory on without an embedder. Add `embedder={"provider": "onnx"}`, or keep memory off. |
| `ValueError: Missing required template variable 'today'` | An input is missing. Run `python inputs_check.py`. |
| `numbers outside the brief .... 1 ['40%']` | The Writer invented a number. The draft waits for you: edit or reject it in `approve.py`. |
| `Result: 1 unsupported numbers` | An agent guessed a number. Do not share the report; run Step 12 again. |
| Usage for the second lead is twice the first | `kickoff_for_each` reports running totals: use `delta_since()` (Step 8). |

## Final challenge

Give the desk a `today` **tool** and measure whether the stale year disappears from the claims check, then add a fourth lead written in Hindi and make the tone check work for it. Details, a checklist and hints: [CHALLENGE.md](CHALLENGE.md).

## What you learned

- One crew, many cases: `{placeholders}`, `kickoff_for_each`, and a check that no input is missing or silently unused. Today's date is an input, because models have no clock.
- A tool earns trust with known inputs before an agent uses it. Our first sentiment model failed that test; a stub that always says "positive" would fail it worse.
- `context` hands each answer to the next task, `output_pydantic` makes it a typed object, and our own scorecard checks the agents against what the tools really returned.
- The send step lives outside the crew: drafts only, the address copied by code, a human gate, a `mailto:` link. That is excessive agency, avoided by design.
- Models explain, code computes: every number in the research note comes from pandas or the calculator, a claims checker traces it, and code stamps the disclaimer. Education only, invented data, no advice.
- Hierarchical crews add a manager and its calls: measure before you choose, and cap every loop with `max_iter` and `max_rpm`.

## Next episode

**Episode 15: One Agent, Many Tools: Calculator, Read-Only SQL and Crawl Search With LangChain create_agent** - back in the LangChain `.venv`: one agent with a calculator, a read-only database and a crawl-search tool, graded against ground truth. -> [ep15-mandi-tool-belt](../ep15-mandi-tool-belt)
