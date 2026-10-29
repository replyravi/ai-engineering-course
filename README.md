# AI Engineering From Zero

**Build LLM apps, RAG and agents in 2026: sixteen episodes, free tools, and baby steps you run yourself.**

The companion code for the YouTube series by **Ravi Shanker Singh** (TechWithRavi).
Every episode has its own folder with small, commented programs, a README that walks you through
each step with the output you should see, offline tests, and a challenge.

- Code: https://github.com/replyravi/ai-engineering-course
- Playlist: the link goes here when Episode 1 is published

## Start here

1. Do **[SETUP.md](SETUP.md)** once (about 20 minutes): Python 3.13, VS Code, one virtual environment, a free Gemini key.
2. Open **[ep01-four-kinds-of-ai](ep01-four-kinds-of-ai/)** and follow its README step by step.
3. Every later episode starts with a two-minute "Step 0" that reuses the same environment and key.

## Episodes

| # | Episode | Folder | Status |
|---|---|---|---|
| 01 | What Is AI, Really? AI vs ML vs Deep Learning vs GenAI, Plus Your Free Gemini Setup | [ep01-four-kinds-of-ai](ep01-four-kinds-of-ai/) | ready |
| 02 | How Transformers Read: Tokens, Embeddings and Self-Attention, Step by Step | [ep02-attention-xray](ep02-attention-xray/) | ready |
| 03 | How GPT Writes: The Decoder, Masking and Next-Token Prediction | [ep03-next-token-lab](ep03-next-token-lab/) | ready |
| 04 | Foundation Models Explained: One Big Model, Many Jobs (Text, Images, Audio) | [ep04-model-zoo](ep04-model-zoo/) | ready |
| 05 | How to Choose an LLM in 2026: Benchmarks, Your Own Eval and AI Safety Basics | [ep05-model-scorecard](ep05-model-scorecard/) | ready |
| 06 | LangChain in 2026 for Beginners: Models, Prompts, Pipes and Your First Agent | [ep06-helpdesk-ladder](ep06-helpdesk-ladder/) | ready |
| 07 | RAG Explained and Built: A Q&A Bot That Cites Its Sources (Free Gemini + Chroma) | [ep07-ask-the-handbook](ep07-ask-the-handbook/) | ready |
| 08 | Vector Databases and Chunking, Tested: Which Way of Splitting Documents Finds Answers Best? | [ep08-chunk-lab](ep08-chunk-lab/) | ready |
| 09 | Prompt Engineering You Can Measure: Zero-Shot, Few-Shot, Chain of Thought and JSON Output | [ep09-prompt-lab](ep09-prompt-lab/) | ready |
| 10 | Load Any File and Give Your Chatbot Memory: Document Loaders + LangChain Memory (2026) | [ep10-counter-bot](ep10-counter-bot/) | ready |
| 11 | LangChain Chains the Modern Way: Pipes, Parallel Steps, and Letting Python Do the Maths | [ep11-party-planner-chains](ep11-party-planner-chains/) | ready |
| 12 | One Assistant, Three Paths: RAG, Routing and Safe Text-to-SQL in LangChain | [ep12-dabbaloop-help-desk](ep12-dabbaloop-help-desk/) | ready |
| 13 | AI Agents and Multi-Agent Crews Explained: Build Your First CrewAI Team | [ep13-shelf-sifter-crew](ep13-shelf-sifter-crew/) | ready |
| 14 | CrewAI Projects: Email Drafts With Human Approval and an Education-Only Stock Research Desk | [ep14-crew-office](ep14-crew-office/) | ready |
| 15 | One Agent, Many Tools: Calculator, Read-Only SQL and Crawl Search With LangChain create_agent | [ep15-mandi-tool-belt](ep15-mandi-tool-belt/) | ready |
| 16 | Hugging Face for Beginners (2026): Run AI Models on Your Own Laptop | [ep16-local-ai-toolbox](ep16-local-ai-toolbox/) | ready |

## How this repo works

```text
ai-engineering-course/
├── README.md                  you are here
├── SETUP.md                   one-time setup for Windows, macOS and Linux
├── .env.example               copy ONCE to .env (here, at the repo root) and paste your key
├── .gitignore                 keeps .env, .venv*, logs, indexes and caches out of git
├── requirements-base.txt      the LangChain + Hugging Face stack (every episode except 13-14)
├── requirements-crewai.txt    the CrewAI stack (episodes 13-14, in a second environment)
├── _template/                 the shared helpers every episode starts from
├── tools/sync_template.py     copies the shared helpers into episode folders, reports drift
└── epNN-<name>/               one folder per episode
```

- **One environment, one key.** Episodes 1-12 and 15-16 share one virtual environment, `.venv`, at the repo root.
  Episodes 13-14 use a second one, `.venv-crewai`, because CrewAI pins older versions of some libraries.
  There is one `.env` file, also at the repo root. Each episode's `llm.py` finds it from inside the episode folder.
- **One-line provider switch.** `LLM_PROVIDER=gemini` (default, free key), `openai`, `anthropic` or `ollama`
  (free, runs on your computer) in `.env`. Nothing else changes.
- **Gemini on screen.** The videos use Google Gemini with a free Google AI Studio key. Embeddings default to a
  free local model (`sentence-transformers/all-MiniLM-L6-v2`), so many steps also run offline.
- **Tests run without internet.** Every episode has `python -m pytest`. The tests use fake AI models, so they need
  no key, cost nothing and give the same result every time.
- **Inside each episode folder:** `README.md` (the baby steps), the programs, `data/` (small invented data),
  `llm.py` and `testing/` (copies of `_template/`), `tests/`, and `CHALLENGE.md`.

## Requirements

- **Python 3.13** recommended. 3.11 and 3.12 also work. Not 3.14 or 3.15 yet (no PyTorch wheels for 3.15, and
  CrewAI does not install on 3.14+). SETUP.md explains how to get the right one.
- Windows 10/11 (64-bit), macOS 14 or newer on Apple Silicon, or 64-bit Linux.
  Intel Macs cannot install the current PyTorch; use Google Colab for the episodes that need it.
- A Google account for the free Gemini key. No credit card or paid account is needed for anything shown on screen.
- Some free disk space: PyTorch is the biggest one-time download, and the local embeddings model is about 90 MB.

## Safety rules used in every episode

- Keys live only in `.env`, which git ignores. Never paste a key into code, a screenshot or a chat.
- On the free tier, Google may use your prompts to improve its products. Do not send private data.
- Every data file in this repo is invented for teaching. Email drafts are never sent without a human click,
  the finance episode is education only, and scraping uses practice sites that allow it.

## For maintainers

```bash
python tools/sync_template.py --check --all     # report any episode whose llm.py or testing/ drifted
python tools/sync_template.py ep02-attention-xray  # copy the shared helpers into one episode
```

## Licence

The code in this repository is released under the [MIT licence](LICENSE) (c) 2026 Ravi Shanker Singh.
You may use, change and share it, including in your own projects. The videos are not covered by this licence.
