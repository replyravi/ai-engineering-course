# Episode 06: LangChain in 2026 for Beginners: Models, Prompts, Pipes and Your First Agent

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep06-helpdesk-ladder

## What you will build

A help desk assistant for **Dabbaloop**, the meal-box delivery service in Pune that we invented for this course
(Episode 07 gives it a full handbook). The assistant climbs a ladder, one small script per rung:
plain messages, a template pipe, answers from a handbook you write, typed answers, and an agent that calls a
holiday tool on its own. At the end, `ladder.py` asks five sample questions and prints which rung solved each one.

This is the layout of the report. The cells come from **your** run, because Gemini's answers decide them; the
ones below are the result we expect, written by hand, not a recorded run:

```text
#  Question                                      Prompt         Context        Tools          Cheapest rung that worked
-----------------------------------------------------------------------------------------------------------------------
1  Explain in two sentences what a meal-box ...  solved         -              -              Prompt
2  What is the last time I can skip today's ...  I don't know   solved         -              Context
3  How many days until the next kitchen holi...  I don't know   I don't know   solved (tool)  Tools
4  A customer wrote: 'Thanks, the dal was lo...  solved         -              -              Prompt (no fine-tune needed)
5  Is there a student discount?                  I don't know   I don't know   I don't know   none: says I don't know (correct)
```

**You will learn:**

- Why a framework helps when providers and models keep changing, and when it is overkill.
- Chat models and **messages** (system, human, AI, tool), reading `.text` and the token usage, and switching provider with one `.env` line.
- **LCEL**, the pipe: `prompt | model | parser`, and the three ways to run it: `invoke`, `stream`, `batch`.
- **Retrieve first, then answer**: pick the top 3 handbook paragraphs with local embeddings, cite them, and say "I don't know" when the handbook is silent.
- **Typed answers** with `with_structured_output` and a Pydantic class.
- Your first **agent** with `@tool` and `create_agent`, and how to read its message trace.
- The ladder: prompt, context, tools, fine-tune. Climb only as high as the question needs.

> **Seen an older tutorial?** It used `LLMChain(llm=llm, prompt=prompt).run(...)` and `initialize_agent(...)` or
> `AgentExecutor`. In 2026 we use the pipe `prompt | llm | StrOutputParser()` with `.invoke()`, and
> `langchain.agents.create_agent`. Same ideas, current API. The old classes moved to the separate
> `langchain-classic` package and are deprecated: in LangChain 1.x, `from langchain.chains import LLMChain`
> fails with `ModuleNotFoundError: No module named 'langchain.chains'`.

## Prerequisites

- Finished [Episode 01](../ep01-four-kinds-of-ai) (the shared `.venv`, the root `.env` with your **free** Gemini key, `get_llm()`, and what a hallucination is). [Episode 04](../ep04-model-zoo) helps: Step 7 reuses its embeddings and cosine-similarity ranking.
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- About 50 minutes. Nothing new to download if you did Episode 01, 04 or 07: the local embeddings model is already on your computer.
- Models and licences:
  - [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (about 90 MB, Steps 7-9 and 12) is released under the **Apache-2.0** licence: free to use, also commercially (keep its licence notice if you share the model files).
  - Optional, only if you try the local switch in Step 3: `qwen3.5:4b` through Ollama. Its model card ([Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)) gives the **Apache-2.0** licence (checked 2026-10-23). Check the download size on https://ollama.com/library/qwen3.5 before you pull it.

## The LangChain family in 2026

LangChain gives you one way to talk to many model providers, plus building blocks: prompts, parsers, runnables,
retrievers, tools and agents. It is not the only option. For one small script that will only ever call one
provider, that provider's own SDK is fine. A framework pays off when you switch models, add retrieval, tools
or tests, and when you want every step traced. The price is one more dependency that changes fast, so we pin
every version and read the migration guide before upgrading.

| Package (version in this course) | What it is for | Used in this episode |
|---|---|---|
| `langchain` (1.4.7) | `init_chat_model`, `create_agent`, `@tool` | Steps 2, 10 |
| `langchain-core` (1.7.0) | messages, prompt templates, output parsers, runnables (the pipe) | every step |
| `langchain-google-genai` (4.4.2), `langchain-openai`, `langchain-anthropic`, `langchain-ollama` | one package per provider | Steps 2-3 (through `get_llm()`) |
| `langchain-huggingface` (1.2.3) | the local embeddings model | Steps 7-9 (through `get_embeddings()`) |
| `langgraph` (1.2.16) | runs stateful agent loops, single or multi-agent; `create_agent` is built on it | Step 10 |
| `langsmith` (0.14.12, comes with `langchain-core`) | tracing and evaluation in the browser | Step 14 (optional) |
| `langchain-classic` (1.0.8) | the legacy chains and memory classes, for old code only | not used |
| `langchain-community` (0.4.2) | loaders, FAISS and more; the project was archived on 2026-06-19 ([announcement](https://github.com/langchain-ai/langchain-community/issues/674)), so the course pins version 0.4.2 | not used here |

> **Common Mix-up.** LangSmith is not "the single-agent framework". It records and evaluates runs (traces).
> LangGraph is what runs agent loops, one agent or many. And **Groq** (a company that serves open models fast)
> is not **Grok** (xAI's model family). This course uses neither.

## What is in this folder

```text
ep06-helpdesk-ladder/
├── data/
│   ├── handbook.md           12 short Dabbaloop rules, one per paragraph (sample by Ravi: write your own)
│   └── holidays.csv          5 invented kitchen holidays
├── common.py                 shared bits: file paths, the house rules, friendly error messages
├── rung1_messages.py         Steps 2-3: system + human messages, .text, token usage
├── rung2_template.py         Steps 4-5: prompt | model | parser, invoke, stream, batch
├── rung3_context.py          Steps 6-8: no context, top 3 paragraphs as context, the "I don't know" rule
├── rung4_structured.py       Step 9:    a typed HelpAnswer
├── rung5_tool.py             Steps 10-11: the days_until_holiday tool, the agent, the message trace
├── ladder.py                 Step 12:   which rung solves which question
├── llm.py, testing/          shared helpers (synced from _template, do not edit here)
├── tests/                    offline tests (fake chat model, no key, no internet)
├── requirements.txt          points to the shared ../requirements-base.txt
└── CHALLENGE.md
```

The five `rung` scripts are the steps you climb as a builder. `ladder.py` then groups them into the rungs a
help desk answer can use: **Prompt** (what rung 1 and 2 do), **Context** (rung 3), **Tools** (rung 5), with
**Fine-tune** as the fourth rung that we never need here. Every file is already in the folder. Type the code
yourself as we go, or read along and run it.

## Step 0 - Get ready (about 2 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the repo root (`ai-engineering-course/`). You created them once in Episode 1 ([SETUP.md](../SETUP.md)). New here? Do SETUP.md first, then come back.

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd ep06-helpdesk-ladder
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep06-helpdesk-ladder
```

Expected: your prompt starts with `(.venv)` and ends in `ep06-helpdesk-ladder`.

> [!WARNING]
> **Common errors**
> - `.venv\Scripts\Activate.ps1 cannot be loaded` / `No such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

### 0.2 Make sure the libraries are up to date

Nothing new to install if you already ran Episode 1: this episode uses the shared `requirements-base.txt`. If you pulled a newer version of the repo, update from this folder:

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
> - `ModuleNotFoundError: No module named 'langchain_google_genai'` (or `langgraph`) - the `.venv` is not active in this terminal. Activate it again (0.1).

---

## Step 1 - Check the setup

Make sure your key is found and Gemini answers. Your key lives in `ai-engineering-course/.env`. `llm.py` looks in this folder and then in each parent folder, so it finds the root file by itself. Do not copy `.env` into episode folders.

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

Without a key you get help instead of a crash (real output with no `.env` file at all, first lines; when a `.env` exists, the first line shows its path instead):

```text
Settings file: no .env found
Provider: gemini   Model: gemini-3.5-flash-lite


No API key found for LLM_PROVIDER=gemini.
Looked for: GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.

Get a FREE Gemini key (no credit card needed):
  1. Open https://aistudio.google.com/apikey
...
```

Every script in this folder prints that same help when the key is missing.

> [!WARNING]
> **Common errors**
> - `No API key found for LLM_PROVIDER=gemini` - the `.env` file is missing, sits in the wrong folder (it belongs in `ai-engineering-course/`), is named `.env.txt`, or still says `paste-your-key-here`. The last line of the help tells you which `.env` file was found.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute (per-minute limit) or until the daily reset. Your limits: https://aistudio.google.com/rate-limit

## Step 2 - Send system and human messages

A chat model does not read one loose string. It reads a **conversation**, like a transcript with a speaker label
on every line, and we hand it the whole transcript on every call. The **system** message sets the rules, the
**human** message asks, and the reply comes back as an **AI** message. (Agents add a fourth role, **tool**, in Step 11.)

Our rules live in `common.py`, because every rung uses them:

```python
HOUSE_RULES = (
    "You are the help desk assistant of Dabbaloop, a meal-box delivery service in Pune. "
    "Start every reply with 'Namaste!'. Be warm and brief: at most three short sentences. "
    "Write numbers as digits."
)
```

`rung1_messages.py` (the important lines):

```python
from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm

messages = [
    SystemMessage(HOUSE_RULES),
    HumanMessage("Explain in two sentences what a meal-box subscription is."),
]
llm = get_llm()                      # Gemini, from LLM_PROVIDER in the root .env
reply = llm.invoke(messages)         # an AIMessage

print(reply.text)                    # .text, not .content (Gemini 3 returns a list of blocks)
print(reply.response_metadata.get("model_name"))   # the model id the provider reports
print(reply.usage_metadata)          # a plain dict of token counts
```

```bash
python rung1_messages.py
```

Expected output (Gemini writes its own sentences, so your wording, token numbers and time will differ):

```text
Sending 2 messages:
  system You are the help desk assistant of Dabbaloop, a meal-box delivery service in Pune. Start every reply with 'Namaste!'. Be warm and brief: at most three short sentences. Write numbers as digits.
  human  Explain in two sentences what a meal-box subscription is.

Provider: gemini   Model: gemini-3.5-flash-lite

Reply (ai):
Namaste! <two sentences written by Gemini>

Model id reported by the provider: gemini-3.5-flash-lite
Token usage: {'input_tokens': <N>, 'output_tokens': <N>, 'total_tokens': <N>, 'input_token_details': {'cache_read': <N>}}
Time: <seconds> s
```

Tokens are the small pieces of text a model reads and writes, and free-tier limits are counted in them. The
input count includes the system message: the rules travel with every single call. If the model "thought" before
answering, you also see `'output_token_details': {'reasoning': <N>}`: thinking tokens are counted as output tokens.

**What `get_llm()` does, in plain LangChain.** For Gemini it builds a `ChatGoogleGenerativeAI` model. The
one-line LangChain way is `init_chat_model` with a **provider prefix**:

```python
from langchain.chat_models import init_chat_model
llm = init_chat_model("google_genai:gemini-3.5-flash-lite")
# others: "openai:gpt-6-luna", "anthropic:claude-haiku-5-5", "ollama:qwen3.5:4b"
```

Always write the prefix. Without it, LangChain guesses **Vertex AI** (Google Cloud) for a Gemini name (real output):

```text
DeprecationWarning: Inferred `model_provider='google_vertexai'` from 'gemini-3.5-flash-lite'. This default will change to 'google_genai' in the next major release. ...
ImportError: Initializing ChatVertexAI requires the langchain-google-vertexai package. Please install it with `pip install langchain-google-vertexai`
```

`get_llm()` also handles provider quirks for you: Gemini 3.5 Flash-Lite, 3.6 Flash and later releases ignore
`temperature`, and some Claude models reject it, so `get_llm()` only sends it to local Ollama models.

> **Classic vs 2026.** Older tutorials imported model classes from the main package (`from langchain.llms import OpenAI`)
> and called `llm.predict(...)`. In LangChain 1.x, `langchain.llms` no longer exists (`ModuleNotFoundError: No module named 'langchain.llms'`).
> Use a provider package or `init_chat_model("provider:model")`, call `.invoke()`, and print `.text`.

> [!WARNING]
> **Common errors**
> - You see `[{'type': 'text', 'text': ...}]` - you printed `reply.content`. Print `reply.text`.
> - `Token usage: None` - some providers or local models report no usage. The answer is still fine.
> - `404 ... model not found` - `LLM_MODEL` in `.env` names a retired or misspelled model. Leave `LLM_MODEL=` empty to use the default.

## Step 3 - Switch provider with one line

This is the first thing LangChain buys you. Open the root `.env`, change **one** line, and run the **same** script:

```text
LLM_PROVIDER=ollama
```

```bash
python rung1_messages.py
```

With Ollama installed and the model pulled (`ollama pull qwen3.5:4b`), expect the same layout with a local model
(written by hand, not a recorded run; the wording, numbers and model id string come from your run):

```text
...
Provider: ollama   Model: qwen3.5:4b

Reply (ai):
<two sentences from the model on your laptop>

Model id reported by the provider: qwen3.5:4b
...
```

It works with Wi-Fi off: the model runs on your computer. If Ollama is **not** running, you get this instead (real output):

```text
...
Provider: ollama   Model: qwen3.5:4b

Could not reach Ollama (ConnectError: [Errno 61] Connection refused).
Ollama runs models on your own computer, and it is not running yet:
  1. Install it from https://ollama.com/download and start it
  2. In a terminal:  ollama pull qwen3.5:4b
  3. Run this script again
```

No Ollama? Try `LLM_PROVIDER=openai` or `anthropic` with a key, or with no key at all to see the help for that provider:

```text
...
Provider: openai   Model: gpt-6-luna

No API key found for LLM_PROVIDER=openai.
Looked for: OPENAI_API_KEY in your .env file.
...
```

Put it back to `LLM_PROVIDER=gemini` before Step 4. "Just change one line" is true for the code, but not for
the models: each provider needs its own key and package, and models differ in tool calling, structured output and speed.

> [!WARNING]
> **Common errors**
> - Windows words the error differently (for example `[WinError 10061]`) - same meaning: nothing is listening on port 11434. Start the Ollama app.
> - An error that says the model `qwen3.5:4b` is not found - you have Ollama but not the model. Run `ollama pull qwen3.5:4b`.
> - A very slow first answer on Ollama - the model is loading into memory. The second call is faster.
> - The reply starts with `<think>` and a long block of reasoning - `qwen3.5` is a "thinking" model, and by default LangChain may show its thinking inside the text. While you use Ollama, change `get_llm()` to `get_llm(reasoning=False)` in `rung1_messages.py` (an Ollama-only setting: change it back before you switch to Gemini).

## Step 4 - Template, model and parser in a pipe

Think of a juice counter: fruit goes in, the blender works, a strainer catches the bits, and a glass comes out.
In LangChain the three parts snap together with `|`:

- `ChatPromptTemplate` fills named **slots** like `{topic}` and makes the messages (dict in, messages out).
- the **model** answers (messages in, AI message out).
- `StrOutputParser` keeps only the text (AI message in, plain string out).

`rung2_template.py`:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    ("system", HOUSE_RULES),
    ("human", "Explain {topic} to {audience} in two short sentences."),
])


def build_chain(llm=None):
    """prompt -> model -> plain text. Pass llm=fake_llm(...) in tests."""
    if llm is None:
        llm = get_llm()
    return PROMPT | llm | StrOutputParser()


chain = build_chain()
print(chain.invoke({"topic": "a meal-box subscription", "audience": "a first-time customer"}))
```

First look at the filled prompt. No model call, so no key is needed:

```bash
python rung2_template.py --show-prompt
```

Expected output (exact):

```text
Template slots: audience, topic

----- system message -----
You are the help desk assistant of Dabbaloop, a meal-box delivery service in Pune. Start every reply with 'Namaste!'. Be warm and brief: at most three short sentences. Write numbers as digits.
----- human message -----
Explain a meal-box subscription to a first-time customer in two short sentences.
```

Now run the chain:

```bash
python rung2_template.py
```

Expected shape (Gemini's wording will differ):

```text
LangSmith tracing: off

Namaste! <two short sentences about a meal-box subscription>

(invoke: the whole answer arrived after <seconds> s)
```

> **Classic vs 2026.** Older tutorials wrote `LLMChain(llm=llm, prompt=prompt)` and called `.run(topic=...)`.
> That class now lives only in `langchain-classic`, and it warns you when you use it (real output):
>
> ```text
> LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 2.0.0. Use `RunnableSequence, e.g., `prompt | llm`` instead.
> LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain-classic 0.1.0 and will be removed in 2.0.0. Use `invoke` instead.
> ```
>
> The pipe you just wrote, run with `.invoke()`, **is** that replacement.

> [!WARNING]
> **Common errors**
> - `KeyError: "Input to ChatPromptTemplate is missing variables {'audience'}. ..."` - every `{slot}` in the template needs a value in the dict you pass.
> - `ModuleNotFoundError: No module named 'langchain.prompts'` - the old import path. Use `from langchain_core.prompts import ChatPromptTemplate`.
> - `ValueError: Model 'gemini-3.5-flash-lite' does not support model prefilling. The final request turn must be a user message or a function response.` - your prompt ends with an AI message. Gemini 3.5 Flash-Lite and 3.6 Flash reject that before sending anything. Always end with the human turn.

## Step 5 - Stream and batch the same chain

The chain is a **runnable**, so the same object also streams and batches. Nothing in `build_chain` changes:

```python
for piece in chain.stream({"topic": topic, "audience": AUDIENCE}):
    print(piece, end="", flush=True)        # print each piece the moment it arrives

answers = chain.batch([{"topic": t, "audience": AUDIENCE} for t in TOPICS])   # 3 inputs, 3 answers, same order
```

```bash
python rung2_template.py --stream
python rung2_template.py --batch
```

Expected shape (your wording and timings will differ):

```text
LangSmith tracing: off

Namaste! <the answer appears piece by piece while you watch>

(stream: first piece after <seconds> s, whole answer after <seconds> s, <N> pieces)
```

```text
LangSmith tracing: off

[a meal-box subscription]
Namaste! <answer>

[a delivery slot]
Namaste! <answer>

[a refundable deposit]
Namaste! <answer>

(batch: 3 answers after <seconds> s)
```

Compare the two timers of the stream run with the invoke run of Step 4: a stream shows the first words before
the whole answer is done, which is why chat apps stream. Batch runs the three calls at the same time (in
parallel threads), so compare its time with three single runs.

> [!WARNING]
> **Common errors**
> - The stream prints everything at once - the terminal is buffering. Keep `flush=True` in the `print`.
> - `429 RESOURCE_EXHAUSTED` during `--batch` - three calls at once can hit the per-minute limit. Wait a minute and run it again.

## Step 6 - Ask a handbook question with no context

Dabbaloop does not exist, so no model was trained on its rules. Ask a rule question with only the house rules:

```python
NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HOUSE_RULES),
    ("human", "{question}"),
])
```

```bash
python rung3_context.py --no-context
```

Expected shape:

```text
Question: What is the last time I can skip today's lunch?

<Gemini's answer: often a confident time that is NOT our rule, or a polite "I can't know that">

(No handbook text was sent with this question. Any time, price or rule in the answer above is a guess, not Dabbaloop's rule. The real rule is in data/handbook.md.)
```

Whatever it says, nothing in it comes from our handbook. A model writes the most likely next words, so a
fluent answer can be completely invented. That is a **hallucination**. Label it as one on screen and move on.

> [!WARNING]
> **Common errors**
> - The model says it cannot know Dabbaloop's rules - that is honest, and still not an answer. Step 7 gives it the rule.

## Step 7 - Add the top three paragraphs as context

Answering a customer from memory is risky; answering with the handbook open on your desk is not. So **our code
searches first**, then hands the model the text. The model does not search anything by itself.

1. Split `data/handbook.md` into numbered paragraphs (`handbook.md#1`, `#2`, ...).
2. Embed every paragraph and the question with the free local model (`get_embeddings()`).
3. Rank by **cosine similarity**, the same function as Episode 04, and keep the top 3.
4. Paste those 3 into the prompt, with a rule to answer **only** from them and cite the tag.

`rung3_context.py` (the important lines):

```python
def load_paragraphs(path=HANDBOOK_FILE):
    """Every paragraph of the handbook with a tag like 'handbook.md#4'."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)        # drop comments
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text)]
    paragraphs = [block for block in blocks if block and not block.startswith("#")]
    return [{"tag": f"{path.name}#{number}", "text": " ".join(paragraph.split())}
            for number, paragraph in enumerate(paragraphs, start=1)]


def cosine_similarity(a, b):
    """How closely two vectors point the same way: 1.0 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank(question, paragraphs, embeddings, top=3):
    vectors = embeddings.embed_documents([p["text"] for p in paragraphs])
    question_vector = embeddings.embed_query(question)
    scored = [(cosine_similarity(question_vector, v), p) for v, p in zip(vectors, paragraphs)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top]


GROUNDED_RULES = """Answer the customer's question using ONLY the handbook paragraphs below.
1. End every sentence with the tag of the paragraph it came from, like [handbook.md#N] (N is the paragraph number).
2. Never use outside knowledge, and never guess a time, a price or a date."""
```

```bash
python rung3_context.py
```

The search part runs on your computer and needs no key, so this part is exact (real output; your timings will
differ, and the very first run is slower: about 30 s to a minute while Python and the model load for the first time):

```text
Question: What is the last time I can skip today's lunch?

Top 3 handbook paragraphs (cosine similarity, higher = closer in meaning):
  0.619  [handbook.md#4]  Cut-off times: to skip, pause or change a meal, do it in the app befor...
  0.602  [handbook.md#9]  Missed meal: if your meal has not arrived 45 minutes after your slot e...
  0.450  [handbook.md#3]  Delivery slots: lunch comes Monday to Saturday, from 12:00 pm to 12:45...
(on your computer: model loaded in 4.4 s, 12 paragraphs ranked in 0.12 s)
```

Then Gemini answers. Expected shape (written by hand, not a recorded reply; the time and the tag should match):

```text
Answer:
Namaste! <skip today's lunch in the app before 9:30 am> [handbook.md#4].

Cited: handbook.md#4  (all from the 3 paragraphs we sent)
```

Look at the scores: the right rule (`#4`, 0.619) only just beats the missed-meal rule (`#9`, 0.602). Both talk
about a meal, a slot and the app. That is why we send three paragraphs, not one. To read the whole prompt the
model gets, run `python rung3_context.py --show-prompt` (no key needed).

This is **R**etrieval **A**ugmented **G**eneration in its smallest form: retrieve, put the text in the prompt,
generate. Episode 07 builds the full version with a vector database.

> [!WARNING]
> **Common errors**
> - First run without internet: about a minute of `Retrying in 1s [Retry 1/5]` lines, then `Could not download the free local embeddings model (about 90 MB) from huggingface.co.` - the model downloads once. Connect and run again.
> - `NOTE: Redirects are currently not supported in MacOs`, a `Loading weights` progress bar and `Warning: You are sending unauthenticated requests to the HF Hub` - all harmless. A free `HF_TOKEN` in `.env` silences the last one; `HF_HUB_OFFLINE=1` in `.env` skips the update check once the model is downloaded.
> - No `[handbook.md#N]` tag in the answer - small models sometimes skip rule 1. Ask again, or try `LLM_MODEL=gemini-3.8-flash` in `.env` (smarter, slower, uses more free quota).
> - You rewrote `handbook.md` and the numbers changed - that is expected. Tags follow the paragraph order.

## Step 8 - Add an "I don't know" rule

Grounding helps only if the model is allowed to say "the handbook does not say". Add rule 3:

```python
REFUSAL_RULE = (f"3. If the paragraphs do not contain the answer, say \"{REFUSAL}\" "
                "and offer to connect the customer to a support agent in the in-app chat.")
```

`REFUSAL` is `"I don't know based on the handbook."` (in `common.py`). Now ask something the handbook does not cover,
first without the rule, then with it:

```bash
python rung3_context.py --no-rule "Is there a student discount?"
python rung3_context.py "Is there a student discount?"
```

The search is exact for both (real output):

```text
Question: Is there a student discount?

Top 3 handbook paragraphs (cosine similarity, higher = closer in meaning):
  0.244  [handbook.md#7]  Delivery fee: delivery is free on every weekly, monthly and quarterly ...
  0.177  [handbook.md#6]  Kitchen holidays: on a kitchen holiday there is no lunch and no dinner...
  0.143  [handbook.md#8]  The steel meal box: a new customer pays a refundable deposit of 350 ru...
```

The best score is only 0.244 (compare 0.619 in Step 7): none of these paragraphs is about discounts. Expected
shape of the answers (Gemini's wording will differ):

```text
--no-rule:   <may squeeze an answer out of the delivery-fee paragraph, or guess>
with rule:   Namaste! I don't know based on the handbook. <offer of the in-app chat>
             Cited: nothing
```

A polite "I don't know" plus a way forward beats a confident guess about money.

> [!WARNING]
> **Common errors**
> - The model still answers with outside facts - check that rules 2 and 3 are in the prompt (`--show-prompt`) word for word.
> - Every question gets "I don't know" - the handbook paragraphs did not load (check the `Top 3` lines), or your own handbook has no answer to that question.

## Step 9 - Typed answers

Free text is fine for a person, but code needs fields it can trust. A Pydantic class is a printed form with
boxes, and `with_structured_output` makes the model fill it in (Gemini enforces the schema):

```python
class HelpAnswer(BaseModel):
    """A help-desk answer that our code can read field by field."""

    answer: str = Field(description="The reply to the customer, at most three short sentences.")
    source: str = Field(description="The tag of the handbook paragraph the answer comes from, like "
                                    "handbook.md#N (N is the paragraph number). Write none if the handbook "
                                    "does not answer it.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high if one paragraph states the answer directly, medium if you had to combine "
                    "paragraphs, low if the handbook does not answer the question.")


chain = PROMPT | llm.with_structured_output(HelpAnswer)
result = chain.invoke({"context": format_context(hits), "question": question})   # a HelpAnswer object
```

`Literal["high", "medium", "low"]` means only these three words are allowed, and `Field(description=...)` is the
hint printed next to each box. The field names, types and descriptions **are** the instructions the model sees.
Look at the form, and at what happens to a badly filled one (no key needed):

```bash
python rung4_structured.py --show-schema
```

Expected output (exact, the end):

```text
  "required": [
    "answer",
    "source",
    "confidence"
  ],
  "title": "HelpAnswer",
  "type": "object"
}

A badly filled form is rejected, not guessed:
2 validation errors for HelpAnswer
source
  Field required [type=missing, input_value={'answer': 'Before 9:30 a...onfidence': 'very sure'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing
confidence
  Input should be 'high', 'medium' or 'low' [type=literal_error, input_value='very sure', input_type=str]
    For further information visit https://errors.pydantic.dev/2.14/v/literal_error
```

Now the real thing:

```bash
python rung4_structured.py
```

Expected shape (the `Context sent` line is exact; Gemini fills in the fields):

```text
Question: What is the last time I can skip today's lunch?

Context sent: handbook.md#4, handbook.md#9, handbook.md#3

HelpAnswer
  answer     : Namaste! <the cut-off rule in Gemini's words>
  source     : handbook.md#4
  confidence : high

Source check: handbook.md#4 is one of the 3 paragraphs we sent. OK
```

The last line is the point of typed output: our code can **check** the answer. Try
`python rung4_structured.py "Is there a student discount?"` and expect `source: none`, `confidence: low`.
Use structured output whenever code, not a person, reads the answer. One honest warning: `confidence` is the
model's own label, filled in like any other box. It is not a measured probability, which is why our code checks
`source` against the paragraphs it really sent.

> [!WARNING]
> **Common errors**
> - `Source check: WARNING` - the model named a paragraph we never sent. Do not trust that answer; ask again or use `gemini-3.8-flash`.
> - `The model call failed: OutputParserException: ...` (Gemini) or `ValidationError` (OpenAI, Claude and the offline tests) - the model's answer did not fit your form, for example a required number the handbook does not have. Make that field optional or describe it better.
> - On Anthropic models `with_structured_output` uses tool calling instead of a JSON schema. You do not need to change anything.

## Step 10 - A holiday tool and an agent

"How many days until the next kitchen holiday?" needs **today's date** and **our holiday list**. No prompt and no
handbook paragraph can know the first one. So we give the model a **tool**: a receptionist who checks the
holiday calendar before answering, instead of guessing.

`rung5_tool.py` (the important lines):

```python
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def days_until_holiday(holiday: str = "") -> str:
    """Count the days from today until a Dabbaloop kitchen holiday (no deliveries that day).

    Leave holiday empty for the next one, or give part of its name, like 'family day'.
    Read-only: it only reads the holiday list.
    """
    return holiday_answer(holiday)        # plain Python: read holidays.csv, count the days


MAX_AGENT_STEPS = 10                      # our own safety limit for the loop


def build_agent(llm=None):
    if llm is None:
        llm = get_llm()
    return create_agent(model=llm, tools=[days_until_holiday], system_prompt=AGENT_RULES)


result = build_agent().invoke(
    {"messages": [{"role": "user", "content": QUESTION}]},
    {"recursion_limit": MAX_AGENT_STEPS},  # stop a loop that never answers
)
print(result["messages"][-1].text)        # the final AI message is the answer
```

`@tool` is a decorator: the line above the function turns a plain Python function into a tool an agent can call.
The model never sees our Python code. It sees only the tool's **name**, its **description** (the docstring) and
its **typed arguments**. Look, and run the tool yourself (no key needed; `--today` pins the date so you get exactly this):

```bash
python rung5_tool.py --tool-only --today 2026-10-23
```

Expected output (exact):

```text
What the model sees:
  name:        days_until_holiday
  description: Count the days from today until a Dabbaloop kitchen holiday (no deliveries that day).
               Leave holiday empty for the next one, or give part of its name, like 'family day'.
               Read-only: it only reads the holiday list.
  arguments:   {"holiday": {"default": "", "title": "Holiday", "type": "string"}}

Today is Friday 23 October 2026.
days_until_holiday()              -> Kitchen deep-clean day is on Tuesday 17 November 2026: 25 days from today (Friday 23 October 2026).
days_until_holiday('family day')  -> Staff family day is on Monday 28 December 2026: 66 days from today (Friday 23 October 2026).
days_until_holiday('Diwali')      -> No kitchen holiday called 'Diwali' in the list. Known holidays: Kitchen deep-clean day, Staff family day, Annual kitchen check, Monsoon repair day, Dabbaloop anniversary.
```

Without `--today` it uses your computer's date. Now let the agent decide:

```bash
python rung5_tool.py
```

Expected shape (the number of days depends on the day you run it; it must match `--tool-only`):

```text
Question: How many days until the next kitchen holiday?

Namaste! <the next kitchen holiday is Kitchen deep-clean day, N days from today>

(tools the agent used: days_until_holiday)
```

**Safety Note.** Give a tool only the power it needs. Ours is read-only and reads one file. A tool that can send,
pay or delete must ask a human first (Episode 05's "excessive agency" risk). A tool should **return** its
errors as text, as `holiday_answer` does: an exception inside a tool stops the whole agent run. And set your own
limit on the loop: without `recursion_limit`, `create_agent` allows 9,999 steps, so a model that keeps calling
tools could use up your free quota. With our limit of 10 it stops after 5 model calls.

> **Classic vs 2026.** Older tutorials built agents with `initialize_agent(tools, llm, agent=...)` or `AgentExecutor`.
> In LangChain 1.x both imports fail (`ImportError: cannot import name 'initialize_agent' from 'langchain.agents'`).
> `create_agent` replaces them, and it runs on LangGraph: the agent it returns is a LangGraph graph with two nodes, `model` and `tools`.

> [!WARNING]
> **Common errors**
> - `(tools the agent used: none)` and a guessed number - the model skipped the tool. Make the question clearly about kitchen holidays, or try `LLM_MODEL=gemini-3.8-flash`.
> - `Error: could not read the holiday list (...)` - `data/holidays.csv` is missing or a date is not written as `YYYY-MM-DD`.
> - `No upcoming kitchen holiday in the list` - every date in `holidays.csv` is in the past. Add new dates.
> - `The agent stopped: it used all its steps without a final answer` - the model kept calling the tool and never answered, so our `MAX_AGENT_STEPS` limit stopped it. Ask a clearer question, or try `LLM_MODEL=gemini-3.8-flash`.

## Step 11 - Print the agent's message trace

An agent is a loop: the model decides (answer, or call a tool?), the tool runs, the result goes back, and the
model decides again. The loop stops when the model answers instead of calling a tool. Every turn is a message:

```python
def print_trace(messages):
    for number, m in enumerate(messages, start=1):
        if m.type == "ai" and m.tool_calls:
            calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)
            print(f"  {number}. ai     asks for a tool: {calls}")
        elif m.type == "tool":
            print(f"  {number}. tool   {m.name} -> {m.text}")
        else:
            print(f"  {number}. {m.type:<6} {m.text}")
```

```bash
python rung5_tool.py --trace --today 2026-10-23
```

Expected shape (message 3 is our own Python, so it is exact; messages 2 and 4 come from Gemini):

```text
Question: How many days until the next kitchen holiday?

Message trace (4 messages):
  1. human  How many days until the next kitchen holiday?
  2. ai     asks for a tool: days_until_holiday({})
  3. tool   days_until_holiday -> Kitchen deep-clean day is on Tuesday 17 November 2026: 25 days from today (Friday 23 October 2026).
  4. ai     Namaste! <the answer, with 25 days>
Tool calls: 1   Model calls: 2
```

Human, AI with a tool call, tool result, final AI: that is the whole agent. (The model may pass
`{"holiday": ""}` instead of `{}`; both mean "the next one".)

> [!WARNING]
> **Common errors**
> - Only 2 messages (human, ai) - the model answered without the tool. See Step 10.
> - More than 4 messages - the model called the tool again (for example once per holiday name). That is allowed; the loop ends when it answers, or at the latest when `MAX_AGENT_STEPS` is reached.

## Step 12 - Run the ladder report

Now climb only as high as each question needs. `ladder.py` asks five sample questions. For each one it tries
**Prompt**, then **Context**, then **Tools**, and stops at the first rung whose answer passes a check written
in plain Python (a time that must appear, the right number of days, the house greeting, an honest "I don't know").
A model never grades itself here.

```python
def climb(item, llm, embeddings, paragraphs, days):
    """Try the rungs cheapest first and stop at the first one that passes the check."""
    for rung in RUNGS:                                    # ["Prompt", "Context", "Tools"]
        answer, calls, used = try_rung(rung, item["question"], llm, embeddings, paragraphs)
        status = grade(item, answer, days)                # 'solved', "I don't know", 'wrong' or 'answered?'
        ...
        if status.startswith("solved"):
            break
```

```bash
python ladder.py --today 2026-10-23
```

The answer key is worked out by our code before any model call, so it is exact (real output):

```text
Answer key (checked by our code, not by a model):
  1. any answer that is not 'I don't know'
  2. must mention 9:30 (handbook.md#4)
  3. must say 25 (days to Kitchen deep-clean day)
  4. must use the house greeting Namaste
  5. must say I don't know (the handbook has no discount)
```

Then the report. Expected shape (the layout is exact; the cells are the result we expect, and your run decides them):

```text
Help Desk Ladder report   model: gemini-3.5-flash-lite (gemini)   today: 2026-10-23

#  Question                                      Prompt         Context        Tools          Cheapest rung that worked
-----------------------------------------------------------------------------------------------------------------------
1  Explain in two sentences what a meal-box ...  solved         -              -              Prompt
2  What is the last time I can skip today's ...  I don't know   solved         -              Context
3  How many days until the next kitchen holi...  I don't know   I don't know   solved (tool)  Tools
4  A customer wrote: 'Thanks, the dal was lo...  solved         -              -              Prompt (no fine-tune needed)
5  Is there a student discount?                  I don't know   I don't know   I don't know   none: says I don't know (correct)

Fine-tune (the 4th rung): not tried. The house tone came from the prompt (question 4),
and fine-tuning is a weak way to teach facts that change, like dates and rules.
Model calls: 11

Answer from the last rung tried:
  1. Namaste! <...>
  ...
```

How to read it: general knowledge needs only a good **prompt**; a Dabbaloop rule needs **context**; today's
count needs a **tool**; and the house tone, which people often want to fine-tune for, already came from the
system prompt. Fine-tuning is for style and behaviour at scale; it is a poor way to teach facts that change.
A `-` means a cheaper rung already solved it, so we never paid for the next one.

> [!WARNING]
> **Common errors**
> - A cell says `wrong` - the answer was fluent but missed the check (for example "10 am" instead of 9:30). That is the ladder doing its job: it climbs to the next rung.
> - Row 5 says `check it: ... answered without a source` - a rung invented an answer the handbook does not have. Read it, then tighten the refusal rule.
> - `429 RESOURCE_EXHAUSTED` halfway - the report makes about 11 model calls. Wait a minute and run it again.

## Step 13 - Run the offline tests

The tests use a **fake chat model** with scripted answers (`fake_llm`), scripted tool calls (`tool_call`) and
scripted typed answers (`structured_reply`), plus small offline embeddings. They need no key and block the
internet on purpose.

```bash
python -m pytest
```

Expected output (real run):

```text
.....................s...........................................        [100%]
64 passed, 1 skipped in 4.52s
```

An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes.

How a test checks the agent loop without any AI (slightly shortened from `tests/test_agent.py`):

```python
def test_the_loop_is_human_ai_tool_ai(fixed_today):
    llm = fake_llm(
        tool_call("days_until_holiday", {}),                       # model call 1: "run the tool, please"
        "Namaste! The next kitchen holiday is in 25 days.",        # model call 2: the answer
    )
    messages = ask_agent("How many days until the next kitchen holiday?", llm)
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert messages[2].text == TOOL_RESULT                          # our real Python ran
```

The fake model is scripted, but the tool, the loop and our checks are real code.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'rung1_messages'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called a real model. Pass `llm=fake_llm(...)`.
> - `FakeChatModel ran out of scripted replies` - your code made one more model call than the test scripted. Add a reply, or find the extra call.
> - A test marked `s` with "Local model not downloaded yet" - run `python rung3_context.py` once with internet, then test again.

## Step 14 - Optional: turn on LangSmith tracing

LangSmith records every step of a run (the prompt, the model call, the parser, tool calls, timings and tokens)
so you can see it in the browser. It needs a free account at https://smith.langchain.com and its own key.
Add these lines to the **root** `.env` (never on screen, never in code):

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=paste-your-langsmith-key-here
LANGSMITH_PROJECT=ep06-helpdesk-ladder
```

No code changes: LangChain reads these settings by itself. Run any chain again:

```bash
python rung2_template.py
```

The first line now says `LangSmith tracing: ON (project: ep06-helpdesk-ladder) - open https://smith.langchain.com to see the run`.
Open the project in the browser and click the newest run: you should see the prompt, the model call and the
parser as three steps, with the exact messages and the token counts. Turn it off again with `LANGSMITH_TRACING=false`.

If you set `LANGSMITH_TRACING=true` but forget the key, the script warns you (real output):

```text
LangSmith tracing: ON, but LANGSMITH_API_KEY is missing in .env (no trace will be saved)
```

> **Classic vs 2026.** Older tutorials used `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`. The current
> `langsmith` package still reads those, but the `LANGSMITH_...` names are the ones to use now.

**Safety Note.** A trace stores your prompts and answers on LangSmith's servers. Do not trace private customer data.
Your account is in another region (the EU, for example)? Also set `LANGSMITH_ENDPOINT=` to the API address shown in your LangSmith settings.

> [!WARNING]
> **Common errors**
> - `LangSmith tracing: off (write LANGSMITH_TRACING=true in small letters, not True)` - LangSmith switches on only for the exact word `true`. `True` or `TRUE` records nothing.
> - No runs appear - check the spelling of `LANGSMITH_TRACING=true` and that the key is in the root `.env`.
> - `401` / `403` errors in the terminal - the LangSmith key is wrong or belongs to another region.
> - `Failed to multipart ingest runs: ... Please confirm your internet connection` - tracing is on, but you are offline (for example the Wi-Fi-off test of Step 3). The script still answers; only the trace upload fails. Set `LANGSMITH_TRACING=false` for offline runs.

## Fix-It Corner

| You see | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'langchain.chains'` | Old tutorial code. Use the pipe (Step 4). |
| `ImportError: cannot import name 'initialize_agent'` | Use `create_agent` (Step 10). |
| `ImportError: Initializing ChatVertexAI requires ...` | Add the provider prefix: `google_genai:gemini-3.5-flash-lite` (Step 2). |
| `[{'type': 'text', 'text': ...}]` printed | Print `.text`, not `.content`. |
| `Connection refused` with `LLM_PROVIDER=ollama` | Start Ollama and `ollama pull qwen3.5:4b` (Step 3). |
| The agent guesses instead of calling the tool | Clearer question, or `LLM_MODEL=gemini-3.8-flash` (Step 10). |

## Final challenge

Give the agent a **second tool** (a lookup in a timetable CSV you write, or a unit converter) and make it pick the
right tool for five test questions. Details in [CHALLENGE.md](CHALLENGE.md).

## What you learned

- A framework gives you one interface over many providers: the same script ran on Gemini and switches to Ollama with one `.env` line.
- Messages have roles; print `.text`; `usage_metadata` shows the tokens you spend.
- `prompt | model | parser` is a runnable that can `invoke`, `stream` and `batch`. It replaced `LLMChain`.
- Retrieve first, then answer: our code picks the top 3 paragraphs, the model cites them, and an "I don't know" rule stops guesses.
- `with_structured_output` returns a typed object your code can check.
- An agent is a loop of model decisions and tool calls; `create_agent` builds it on LangGraph, the trace shows every step, and you set the step limit.
- Climb the cheapest rung that works: prompt, context, tools, and fine-tune last.

## Next episode

**Episode 07: RAG Explained and Built** - Step 7 searched 12 paragraphs in a Python list. What happens with a whole
handbook of files and a PDF? We build a real RAG bot on a vector database that cites its sources. -> [ep07-ask-the-handbook](../ep07-ask-the-handbook)

---

Dabbaloop, its rules, holidays and prices are invented for teaching. Any match with a real business is a coincidence.
