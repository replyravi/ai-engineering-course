# Episode 07: RAG Explained and Built: A Q&A Bot That Cites Its Sources

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep07-ask-the-handbook

## What you will build

`ask.py`, a terminal bot that answers questions about **Dabbaloop**, a meal-box delivery service in Pune that we invented for this course. It reads the Dabbaloop handbook first (4 Markdown files and 1 PDF), answers in plain words, ends every sentence with a citation like `[delivery.md#8]`, and says **"I don't know based on the handbook."** when the handbook has no answer.

```text
You: What is the refund window for a missed lunch?
Bot: Report a missed lunch in the app within 6 hours of the end of your slot to get a full wallet credit [delivery.md#8]. A meal counts as missed when it has not reached you 45 minutes after your slot ends [refunds.md#3].
     cited  delivery.md#8      section: Late or missed meal  (distance 0.437)
     cited  refunds.md#3       section: Missed or late meal  (distance 0.449)
            refunds.md#12      section: What we cannot refund  (distance 0.450)

You: What is the best laptop for coding?
Bot: I don't know based on the handbook.
```

(The answer sentences on this page are written by hand to show the format; they are not a recorded Gemini reply, and Gemini will word its answer in its own way. The sources and distances come from a real run of the search, so you should see the same ones.)

**You will learn:**

- Why RAG beats a bare model for private, fresh and checkable facts, and the two pipelines: **index once** (load, split, embed, store) and **ask many times** (retrieve, augment, generate).
- The difference between tokens (what the chat model reads) and embeddings (what the search uses).
- How to split documents with overlap, build a Chroma index that never duplicates, and search it with distances.
- How to write a grounded prompt with citations and a refusal rule, and wire it as an LCEL chain that returns the answer **and** its sources.
- How to measure retrieval with a hit rate on golden questions, and see what chunk size and k do to it.

> **Seen an older tutorial?** It used `RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=...)`. In 2026 we build the same pipeline as an **LCEL chain**: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`.
> Same idea, but you can see the prompt, add sources and test each part. ("Stuff" only means: paste all retrieved chunks into one prompt.) Both `RetrievalQA` and the newer `create_retrieval_chain` still live in the legacy `langchain-classic` package: `RetrievalQA` is deprecated, and `create_retrieval_chain` is a legacy helper. The other modern pattern, a RAG **agent** that calls a search tool, comes later in the course.

## Prerequisites

- Finished Episode 06, folder `ep06-helpdesk-ladder` (or know the `prompt | llm | parser` pipe). Episodes 02 and 04 help with tokens, embeddings and cosine similarity.
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- The shared `.venv` and root `.env` with your **free** Gemini key from Episode 1 ([SETUP.md](../SETUP.md)).
- On the free tier, Google may use your prompts to improve its products. RAG pastes your documents into every prompt, so only use files you are allowed to share. Our handbook is invented, so it is safe.
- About 45 minutes. The local embeddings model is about 90 MB; if you did Episode 01, 04 or 06, it is already on your computer.
- That model, [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), is released under the **Apache-2.0** licence: free to use, also commercially (keep its licence notice if you share the model files). It is the only model this episode downloads.

## What is in this folder

```text
ep07-ask-the-handbook/
├── handbook/                 the Dabbaloop handbook (invented company, invented rules)
│   ├── refunds.md  delivery.md  plans.md  support.md
│   └── food_safety.pdf       3 pages (made with tools/make_food_safety_pdf.py)
├── step02_bare_model.py      Step 2: ask the model WITHOUT the handbook
├── ingest.py                 Steps 3, 4, 6: load, split, embed, store
├── step05_embed_three.py     Step 5: three sentences, three similarity scores
├── search.py                 Step 7: top-k chunks with distances
├── rag_chain.py              Steps 8-11: prompt, chain, answer + sources, refusal
├── ask.py                    Step 12: the chat loop
├── eval_retrieval.py         Step 14: hit rate at k
├── golden.json               8 golden questions + 3 out-of-scope questions
├── tools/make_food_safety_pdf.py   how the PDF was made (you do not need to run it)
├── llm.py, testing/          shared helpers (synced from _template)
├── tests/                    offline tests (fake chat model, no key, no internet)
├── requirements.txt          points to the shared ../requirements-base.txt
├── CHALLENGE.md
└── index/                    made by ingest.py (git-ignored)
```

Every file is already here. Type the code yourself as we go, or read along and run it.

## Step 0 - Get ready (about 2 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the repo root (`ai-engineering-course/`). You created them once in Episode 1 ([SETUP.md](../SETUP.md)). New here? Do SETUP.md first, then come back.

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd ep07-ask-the-handbook
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep07-ask-the-handbook
```

Expected: your prompt starts with `(.venv)` and ends in `ep07-ask-the-handbook`.

> [!WARNING]
> **Common errors**
> - Windows: `The term '.\.venv\Scripts\Activate.ps1' is not recognized` / macOS, Linux: `.venv/bin/activate` with `no such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

### 0.2 Make sure the libraries are up to date

Nothing new to install if you already ran Episode 1: this episode uses the shared `requirements-base.txt` (Chroma, the local embeddings model, pypdf and LangChain are all in it). If you pulled a newer version of the repo, update from this folder:

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
> - `ModuleNotFoundError: No module named 'langchain_chroma'` (or `pypdf`) - the `.venv` is not active in this terminal. Activate it again (0.1).

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

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama`. Nothing else changes.

> [!WARNING]
> **Common errors**
> - `No API key found for LLM_PROVIDER=gemini` - the scripts print this friendly help instead of crashing. The `.env` file is missing, sits in the wrong folder (it belongs in `ai-engineering-course/`), is named `.env.txt`, or still says `paste-your-key-here`. The last line tells you which `.env` file was found.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute (per-minute limit) or until the daily reset. Your limits: https://aistudio.google.com/rate-limit

## Step 2 - Ask the bare model a handbook question

Dabbaloop does not exist, so no model was trained on its rules. Let's see what a model does when it cannot know.

`step02_bare_model.py` (the important lines):

```python
QUESTION = "Dabbaloop is a meal-box delivery service in Pune. What is its refund window for a missed lunch?"
llm = get_llm()
print(llm.invoke(QUESTION).text)   # .text, not .content (Gemini 3+ returns a list of blocks)
```

```bash
python step02_bare_model.py
```

Expected: an answer that sounds sure of itself but is **not** Dabbaloop's rule (the real rule is in `handbook/refunds.md`). Some models admit they do not know this company. Either way, nothing in the answer is grounded in our handbook. That gap is what RAG fills: private data, fresh data, and answers you can check.

> [!WARNING]
> **Common errors**
> - You see the key help text - finish Step 1 first.
> - The model says it cannot browse the internet - that is fine and honest. It still does not know the rule.

## Step 3 - Load the handbook files

A **Document** is text plus a label (metadata). Our label is the file name, so every answer can point back to its source. Markdown files become one Document each; the PDF becomes one Document per page.

`ingest.py`:

```python
def load_handbook(folder=HANDBOOK_DIR):
    """Read every .md file (one Document each) and every .pdf (one Document per page)."""
    docs = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": path.name}))
        elif path.suffix.lower() == ".pdf":
            for page_number, page in enumerate(PdfReader(path).pages, start=1):
                docs.append(Document(
                    page_content=page.extract_text(),
                    metadata={"source": path.name, "page": page_number},
                ))
    return docs
```

(LangChain's `TextLoader` and `PyPDFLoader` do the same job; Episode 10 tours the loaders. Note that `PyPDFLoader` counts pages from 0, while we count from 1 like a human.)

```bash
python ingest.py
```

This one command runs the whole indexing pipeline (Steps 3, 4 and 6). For now, read only the first lines; Steps 4 and 6 explain the rest of its output. The first run on a new computer also downloads the local embeddings model.

Expected output (first lines):

```text
Loaded 5 files as 7 documents (14,869 characters)
  delivery.md      1 document(s)  3,062 characters
  food_safety.pdf  3 document(s)  3,378 characters
  plans.md         1 document(s)  2,692 characters
  refunds.md       1 document(s)  3,094 characters
  support.md       1 document(s)  2,643 characters
```

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'pypdf'` - the venv is not active.
> - Extra lines such as `Loading weights: 100%`, `NOTE: Redirects are currently not supported in MacOs` or a Hugging Face `unauthenticated requests` warning - harmless. They come from the embeddings model (more in Step 5).
> - Text from a PDF is never perfect. Our PDF comes out as `T ake a photo` on page 3. Real PDFs have worse surprises (headers, footers, columns), so always print what your loader read.

## Step 4 - Split into chunks of 400 characters with 60 overlap

We search small pieces, not whole files: a small piece matches one question better, costs fewer tokens and keeps the key line near the top. `split_documents` keeps each chunk's metadata, and we add a chunk number and the Markdown section.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,          # characters per chunk (at most)
    chunk_overlap=60,        # characters that neighbouring chunks may share
    add_start_index=True,    # remember where each chunk starts
)
```

The splitter first tries to cut at blank lines, then at line breaks, then at spaces. The overlap carries the last small piece of one chunk into the next, so a sentence near a cut is not lost.

Run `python ingest.py` again. Expected output (the split part):

```text
Split into 54 chunks (chunk_size=400, chunk_overlap=60)
  Example chunk metadata: {'source': 'delivery.md', 'start_index': 662, 'chunk': 3, 'section': 'Delivery slots'}
Neighbouring chunks that share text: 21 of 49 pairs (15 of them share just a heading line)
  food_safety.pdf#9 ends with:    ...teel box in a microwave. Move the food to a microwave-safe bowl and   «heat it until it is steaming hot all the way through.»
  food_safety.pdf#10 starts with:  «heat it until it is steaming hot all the way through.» - Curd and raita should not stay out of the fridge for more than 2 ho...
  Shared text: 53 characters
```

Look at the numbers: in our Markdown files, paragraphs are short, so most cuts land on a blank line and the only thing carried over is a heading. Inside the PDF page the text has no blank lines, so the overlap is a real sentence.

> [!WARNING]
> **Common errors**
> - `from langchain.text_splitter import ...` fails - that is the old path. Use `from langchain_text_splitters import RecursiveCharacterTextSplitter`.
> - A different chunk count - you changed a handbook file or the settings. That is fine; re-run Step 6 so the index matches.
> - Chunk size is counted in **characters** here, not tokens. Episode 08 counts in tokens.

## Step 5 - Embed three sentences locally

An embedding is ONE list of numbers for a whole piece of text, made by a separate model, used only for **search**. (Tokens and token IDs are a different thing: they are how the chat model reads text.) Two sentences with the same meaning get vectors that point the same way, even with different words.

`step05_embed_three.py`:

```python
SENTENCES = [
    "My lunch box never arrived today.",
    "The rider did not deliver my midday meal.",
    "Fridges must stay at 5 degrees C or colder.",
]
embeddings = get_embeddings()          # free local all-MiniLM-L6-v2, no key
vectors = embeddings.embed_documents(SENTENCES)
```

```bash
python step05_embed_three.py
```

Expected output:

```text
dim 384  (every sentence becomes 384 numbers)
first 5 numbers of sentence 1: [0.0189, 0.0918, 0.0815, 0.0575, 0.0203]

0.428  "My lunch box never arrived today."  vs  "The rider did not deliver my midday meal."
0.225  "My lunch box never arrived today."  vs  "Fridges must stay at 5 degrees C or colder."
0.110  "The rider did not deliver my midday meal."  vs  "Fridges must stay at 5 degrees C or colder."
```

The scores are cosine similarities (Episode 04): 1.0 means the two vectors point the same way, around 0 means unrelated. The first two sentences share only the word "my", yet they score highest. That is meaning, not keywords.

> [!WARNING]
> **Common errors**
> - First run without internet: `OSError: We couldn't connect to 'https://huggingface.co'` - the model downloads once (about 90 MB). Connect and run again.
> - `Warning: You are sending unauthenticated requests to the HF Hub` and, on macOS, `NOTE: Redirects are currently not supported in MacOs` - both are harmless. A free `HF_TOKEN` in `.env` silences the first one.
> - Wi-Fi off and every run waits about 20 seconds? The model library is trying to check for updates. Add `HF_HUB_OFFLINE=1` to `.env` to use the downloaded copy straight away (each run then prints one harmless `Could not reach the Hub ... Using cached commit hash` line).

## Step 6 - Build the Chroma index with stable ids

Chroma is a small vector database: it stores vectors and finds the closest ones fast. It runs inside Python, with no server and no account. We store every chunk and its vector in a Chroma collection saved in `index/`. The trap: `Chroma.from_documents(...)` gives every chunk a random id, so running it three times stores 54, then 108, then 162 chunks, and every answer starts to repeat itself. Instead we give each chunk an id made from its file, chunk number and text. Same chunk, same id, no duplicates.

```python
def chunk_id(chunk):
    """A stable id: the same file + chunk number + text always gives the same id."""
    key = f"{chunk.metadata['source']}|{chunk.metadata['chunk']}|{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def build_index(chunks, embeddings=None, index_dir=INDEX_DIR):
    store = open_index(embeddings, index_dir)     # Chroma with cosine distance, saved in index/
    wanted = {chunk_id(chunk): chunk for chunk in chunks}
    existing = set(store.get(include=[])["ids"])

    old_ids = sorted(existing - set(wanted))              # chunks that are no longer in the files
    new_ids = [i for i in wanted if i not in existing]     # chunks we have not stored yet
    if old_ids:
        store.delete(ids=old_ids)
    if new_ids:
        store.add_documents([wanted[i] for i in new_ids], ids=new_ids)
    ...
```

You already ran `ingest.py` in Steps 3 and 4, so the index exists. To watch a first build, delete the `index` folder, then run the pipeline twice.

**Windows (PowerShell):**

```powershell
Remove-Item -Recurse -Force index
python ingest.py
python ingest.py
```

**macOS / Linux:**

```bash
rm -rf index
python ingest.py
python ingest.py
```

Expected output, first run (last lines):

```text
Embedding new chunks and saving them to index/ ...
Indexed 54 chunks (new this run: 54, removed: 0)
```

Second run:

```text
Embedding new chunks and saving them to index/ ...
Indexed 54 chunks (new this run: 0, removed: 0)
```

Edit a handbook file and run it again: only the changed chunks are removed and re-added.

> [!WARNING]
> **Common errors**
> - The count keeps growing - you used `Chroma.from_documents` without `ids=`. Delete the `index/` folder and run `python ingest.py`.
> - An error about the embedding **dimension** after changing `EMBEDDINGS_PROVIDER` or `EMBEDDINGS_MODEL` - vectors from different models cannot be mixed. Delete `index/` and run `python ingest.py` again.
> - Never commit `index/`. It is in this folder's `.gitignore`, and anyone can rebuild it in seconds.

## Step 7 - Search with scores

Now the question pipeline starts. The question is embedded with the **same** model, and Chroma returns the closest chunks. The score is a cosine **distance**, which is 1 minus the cosine similarity of Step 5: lower means closer in meaning.

```python
def search(question, k=3, store=None):
    """Return the k closest chunks as (Document, distance) pairs, closest first."""
    if store is None:
        store = load_index()
    return store.similarity_search_with_score(question, k=k)
```

```bash
python search.py "What is the refund window for a missed lunch?"
```

Expected output:

```text
Question: What is the refund window for a missed lunch?

1. distance 0.437  [delivery.md#8]  section: Late or missed meal
   ## Late or missed meal The app shows your rider's live location from the moment your meal leaves the kitche...
2. distance 0.449  [refunds.md#3]  section: Missed or late meal
   ## Missed or late meal A meal counts as missed when it has not reached you 45 minutes after your delivery s...
3. distance 0.450  [refunds.md#12]  section: What we cannot refund
   The steel meal box deposit is not a refund. It is returned separately when you hand back your boxes; see th...
```

Notice three real-world things. The best match is `delivery.md#8`, the short summary of the missed-meal rule in the **delivery** handbook, not the refunds policy itself. Result 3, `refunds.md#12`, is noise: "The steel meal box deposit is not a refund" shares the words *refund* and *meal* with the question but says nothing about a missed lunch. And the main policy paragraph, `refunds.md#4` ("Report a missed lunch or dinner within 6 hours..."), only comes 8th (distance 0.564). Its heading, "Missed or late meal", ended up in the chunk before it. See it yourself with `python search.py "What is the refund window for a missed lunch?" -k 8`. Episode 08 tests splitting by headings to fix this kind of miss.

Try the PDF too:

```bash
python search.py "Is the kitchen nut-free?"
```

```text
Question: Is the kitchen nut-free?

1. distance 0.376  [food_safety.pdf#6]  page 2
   that is usually served on Fridays. Our kitchen is not nut-free. Peanuts and cashews are used in the same ki...
2. distance 0.628  [food_safety.pdf#5]  page 2
   3. Allergens in our food Our menu labels show five allergens. Milk: paneer, curd, ghee and butter. Gluten: ...
3. distance 0.701  [food_safety.pdf#4]  page 1
   deliveries. Dabbaloop Kitchen Food Safety and Allergen Guide - page 1 of 3
```

Result 3 is mostly the PDF's page footer: repeated headers and footers become noisy chunks.

> [!WARNING]
> **Common errors**
> - `The index is empty. Build it first with:  python ingest.py` - run Step 6 first.
> - "Distances around 0.4 to 0.6 look bad" - they are normal for this small model. Compare them with each other, not with zero.
> - A good distance does not prove the right chunk came back. `python search.py "my dabba never came today, what now"` scores 0.530, but its top hit, `support.md#1`, is just the title chunk of the support page (so is the 3rd, `refunds.md#1`). The model splits "dabba" into the same word pieces that start "Dabbaloop", so chunks with the company name look close. The real rule, `refunds.md#4`, is 2nd. Ask "my lunch never came today, what now" and the title chunks disappear. Always read the chunks, not only the scores.

## Step 8 - Write and print the grounded prompt

**Augment** means: paste the retrieved chunks into the prompt, each with its tag, plus clear rules. The prompt is plain text. The chat model never sees vectors.

`rag_chain.py`:

```python
SYSTEM_RULES = """You are the help-desk assistant of Dabbaloop, a meal-box delivery service in Pune.
Answer the customer's question using ONLY the handbook extracts below.

Rules:
1. End every sentence with the tag of the extract it came from, for example [refunds.md#3].
2. If the extracts do not contain the answer, reply with exactly this sentence and nothing else:
   I don't know based on the handbook.
3. Never use outside knowledge and never guess a number, a time or a price.
4. Keep the answer short: at most three sentences.

Handbook extracts:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    ("human", "{question}"),      # the last message is the human turn (Gemini needs this)
])


def format_docs(docs):
    """Paste the retrieved chunks into one block of text, each with its citation tag."""
    blocks = [f"[{source_tag(doc)}] ({source_where(doc)})\n{doc.page_content}" for doc in docs]
    return "\n\n".join(blocks)
```

```bash
python rag_chain.py --show-prompt "What is the refund window for a missed lunch?"
```

No key is needed for this one. Expected output (the end):

```text
Handbook extracts:
[delivery.md#8] (section: Late or missed meal)
## Late or missed meal

The app shows your rider's live location from the moment your meal leaves the kitchen. If the meal still has not arrived 45 minutes after your slot ends, it is treated as missed. Report it in the app within 6 hours of the slot ending to get a full wallet credit. The full rules are in the Refunds and Credits Policy.

## Rain, traffic and bandh days

[refunds.md#3] (section: Missed or late meal)
...
----- human message -----
What is the refund window for a missed lunch?
-----
Prompt: 1,458 characters, about 364 tokens
Budget: about 364 prompt tokens + up to ~100 reply tokens = about 464 tokens of the model's context window
```

The printout adds the prompt and the reply into one budget, the classic way to think about a context window. Our Gemini model actually has two separate limits: about 1 million tokens for what it reads and about 65,000 tokens for what it writes (Google's model page, checked 2026-10-23). Our prompt is tiny next to that, so RAG here is about precision, cost and speed, not about fitting. ("About 4 characters per token" is a rule of thumb; Gemini's exact count comes back with each real reply.)

> [!WARNING]
> **Common errors**
> - `KeyError: 'context'` - a `{placeholder}` name in the prompt does not match the keys you pass in.
> - A prompt that ends with an AI message - Gemini 3.5 Flash-Lite and newer reject it. Always end with the human turn.

## Step 9 - Build the LCEL RAG chain

Retrieve, augment, generate in one pipe. This is what `RetrievalQA` used to hide inside one class.

Quick recap from Episode 06: `prompt | llm | StrOutputParser()` is an **LCEL** pipe (LangChain Expression Language). Each step hands its output to the next one: the prompt turns a dict into messages, the model replies with a message, and the parser turns that reply into plain text.

The new part is the dict at the front of the pipe. It runs both of its entries on the same question: `retriever | format_docs` finds the closest chunks (the retriever is Step 7's search, packed by `as_retriever` so it fits in a pipe) and pastes them into one text, and `RunnablePassthrough()` passes the question through unchanged.

```python
def build_simple_chain(store=None, llm=None, k=TOP_K):
    """question -> retrieve k chunks -> fill the prompt -> model -> plain text answer."""
    if store is None:
        store = load_index()
    if llm is None:
        llm = get_llm()
    retriever = store.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
```

```bash
python rag_chain.py --simple "What is the refund window for a missed lunch?"
```

Expected shape (written by hand, not a recorded reply; Gemini's wording will differ, but the tag should be `[delivery.md#8]` and maybe `[refunds.md#3]`):

```text
Report a missed lunch in the app within 6 hours of the end of your slot to get a full wallet credit [delivery.md#8]. A meal counts as missed when it has not reached you 45 minutes after your slot ends [refunds.md#3].
```

Compare this with Step 2: same model, same question. The only difference is that it read first.

> [!WARNING]
> **Common errors**
> - No `[file#chunk]` tag in the answer - small models sometimes skip rule 1. Ask again, or try `LLM_MODEL=gemini-3.8-flash` in `.env` (smarter, slower, uses more free quota).
> - `No API key found` - Step 1.
> - The answer is a Python list like `[{'type': 'text', ...}]` - you printed `message.content`. Keep `StrOutputParser()` at the end of the pipe (or print `.text`).

## Step 10 - Return the answer and its sources together

A user (and you, when debugging) wants to see **where** an answer came from. A parallel step builds the answer and the source list from the same retrieved chunks.

```python
answer_and_sources = RunnableParallel(
    answer=(
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | PROMPT
        | llm
        | StrOutputParser()
    ),
    sources=lambda x: describe_sources(x["docs"]),
)
```

```bash
python rag_chain.py "What is the refund window for a missed lunch?"
```

Expected shape (the answer text is written by hand and Gemini's will differ; the sources come from the real search and should match):

```text
{'answer': 'Report a missed lunch in the app within 6 hours of the end of your slot to get a full '
           'wallet credit [delivery.md#8]. A meal counts as missed when it has not reached you 45 '
           'minutes after your slot ends [refunds.md#3].',
 'sources': [{'tag': 'delivery.md#8', 'where': 'section: Late or missed meal', 'distance': 0.437},
             {'tag': 'refunds.md#3', 'where': 'section: Missed or late meal', 'distance': 0.449},
             {'tag': 'refunds.md#12', 'where': 'section: What we cannot refund', 'distance': 0.45}]}
```

> [!WARNING]
> **Common errors**
> - `KeyError: 'docs'` - the dict step before the parallel step must produce both `docs` and `question`. Compare with `build_chain` in `rag_chain.py`.

## Step 11 - Ask an out-of-scope question

Two safety nets make the bot say **"I don't know based on the handbook."**

1. **The gate:** if even the closest chunk is further than `MAX_DISTANCE = 0.80`, retrieval is weak and we refuse **without calling the model** (free and instant).
2. **The prompt rule:** if the chunks are on topic but do not hold the answer, the model must reply with the refusal sentence, and we then drop the sources.

```python
def retrieval_is_weak(inputs):
    docs = inputs["docs"]
    return not docs or docs[0].metadata["distance"] > max_distance

return (
    {"docs": RunnableLambda(retrieve), "question": RunnablePassthrough()}
    | RunnableBranch((retrieval_is_weak, RunnableLambda(refuse)), answer_and_sources)
    | RunnableLambda(drop_sources_if_refused)
)
```

Two new pieces in this pipe: `RunnableLambda(fn)` lets a normal Python function sit in the pipe, and `RunnableBranch((condition, a), b)` works like `if condition: a else: b`.

```bash
python rag_chain.py "What is the best laptop for coding?"
python rag_chain.py "Who founded Dabbaloop?"
```

Expected for both. The first is stopped by the gate at distance 0.851, so this output is exact. The second reaches the model, which should follow rule 2:

```text
{'answer': "I don't know based on the handbook.", 'sources': []}
```

Why 0.80? We measured it. Clearly off-topic questions (laptops, cricket, quantum computing) had a best distance of 0.73 to 0.86, so a few of them still get past the gate (a Gmail password question scored 0.73); real handbook questions were 0.20 to 0.55, and a question in Hinglish ("the delivery guy never showed up with my khana") was 0.69. A tighter gate would refuse that customer, so the gate only stops the obvious cases and the prompt rule handles the rest. If you change the chunk size or the embeddings model, measure again.

> [!WARNING]
> **Common errors**
> - The model answers anyway with outside facts - make sure rule 2 and rule 3 are still in the prompt, word for word.
> - Every question gets "I don't know" - you set `MAX_DISTANCE` too low, or the index is from a different embeddings model (delete `index/`, run `python ingest.py`).

## Step 12 - Wrap it in ask.py

```python
def chat(chain, read=input, write=print):
    write("Ask the Dabbaloop handbook a question. Type exit to quit.")
    while True:
        try:
            question = read("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in EXIT_WORDS:
            break
        if not question:
            continue
        try:
            write(format_reply(chain.invoke(question)))
        except Exception as error:  # e.g. no internet, or the free-tier limit
            write(f"Bot: the model call failed: {error}")
            write("     (429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and ask again)")
    write("Bye!")
```

```bash
python ask.py
```

Expected: the session shown at the top of this page. It starts with `Ask the Dabbaloop handbook a question. Type exit to quit.`, keeps answering until you type `exit`, and then prints `Bye!`. Sources the answer actually cites are marked `cited`. For a single question: `python ask.py "Do you deliver on Sunday?"`

> [!WARNING]
> **Common errors**
> - `Bot: the model call failed: ... 429 RESOURCE_EXHAUSTED` - free-tier limit; wait a minute. The loop keeps running.
> - Nothing happens when you press Enter on an empty line - by design; type a question or `exit`.

## Step 13 - Test it without internet

The tests use a **fake chat model** with scripted answers, fake embeddings for plumbing tests, and the real local MiniLM model (already downloaded) for search tests. They need no key and block the internet on purpose.

```bash
python -m pytest
```

Expected output:

```text
...............................s..........                               [100%]
41 passed, 1 skipped in 5.25s
```

Your time will differ. An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes.

How a test checks the refusal gate (from `tests/test_refusal.py`):

```python
def test_off_topic_question_is_stopped_by_the_gate(real_store):
    llm = fake_llm()      # no scripted replies: calling the model would raise an error

    result = build_chain(real_store, llm=llm).invoke("What is the best laptop for coding?")

    assert result["answer"] == REFUSAL
    assert llm.received == []       # the model was never called
```

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'ingest'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called a real model. Pass `llm=fake_llm(...)`.
> - More than one `s` (for example `36 passed, 6 skipped`) - run `python -m pytest -rs` to see why. "Local model not downloaded yet" means: run `python step05_embed_three.py` once with internet, then test again.

## Step 14 - Measure retrieval

Is the right chunk in the top k? We wrote 8 golden questions with the file and a phrase that must be retrieved (`golden.json`). A question is a **hit** when one of the top-k chunks comes from the right file and contains that phrase.

```bash
python eval_retrieval.py
```

Expected output:

```text
Index: 54 chunks (chunk_size=400, chunk_overlap=60)

HIT   best 0.437 [delivery.md#8     ] What is the refund window for a missed lunch?
HIT   best 0.364 [delivery.md#2     ] Which areas are in delivery zone C?
HIT   best 0.397 [plans.md#4        ] How much does the monthly Jain thali plan cost?
HIT   best 0.198 [plans.md#6        ] How many days a month can I pause my plan?
HIT   best 0.389 [support.md#2      ] What are the chat support hours on Sunday?
HIT   best 0.376 [food_safety.pdf#6 ] Is the kitchen nut-free?
HIT   best 0.371 [food_safety.pdf#9 ] How long can I keep the delivered food before eating it?
HIT   best 0.301 [delivery.md#6     ] What is the deposit for the steel meal box?

Hit rate at k=3: 8/8 = 100%   (about 234 context tokens per question)

Out-of-scope questions (gate: refuse without the model when best distance > 0.8):
  best 0.436  Who founded Dabbaloop?  -> sent to the model, the prompt rule must refuse
  best 0.699  Is there a student discount?  -> sent to the model, the prompt rule must refuse
  best 0.851  What is the best laptop for coding?  -> refused by the gate
```

Now make the chunks tiny:

```bash
python eval_retrieval.py --chunk-size 120
```

```text
Index: 218 chunks (chunk_size=120, chunk_overlap=60)
...
Hit rate at k=3: 6/8 = 75%   (about 70 context tokens per question)
```

Tiny chunks cut answers away from the words that match the question. And the k trade-off:

```bash
python eval_retrieval.py --k 1 2 3 4 5 6
```

```text
Index: 54 chunks (chunk_size=400, chunk_overlap=60)

  k   hit rate      context sent to the model
  1   7/8 =  88%   about 83 tokens per question
  2   7/8 =  88%   about 158 tokens per question
  3   8/8 = 100%   about 234 tokens per question
  4   8/8 = 100%   about 308 tokens per question
  5   8/8 = 100%   about 375 tokens per question
  6   8/8 = 100%   about 451 tokens per question
```

The hit rate climbs and levels off at k=3, while the tokens keep climbing. More chunks means more noise for the model to read and more cost, so we keep k=3.

> [!WARNING]
> **Common errors**
> - Your numbers differ slightly - you edited the handbook or `golden.json`. That is the point of the scoreboard: re-run it after every change.
> - A hit rate is about **retrieval**. It does not prove the model's answer is right. Spot-check a few answers against their cited chunks too.

## Fix-It Corner: where RAG answers go wrong

| Symptom | Check first |
|---|---|
| Wrong answer | Print the retrieved chunks (`python search.py "..."`). Is the right text there? |
| Right chunk missing | Chunk size or k (Step 14). |
| Right chunk present, wrong answer | The prompt rules, or try `gemini-3.8-flash`. |
| Answer adds outside facts | The grounding rules (rule 2 and rule 3). |
| The same source appears twice | Duplicate chunks: stable ids (Step 6). |
| Out-of-scope question gets an answer | The refusal path (Step 11). |

## Final challenge

Add a new policy file (a festival holiday schedule), re-index without duplicates, keep the hit rate at 100 percent, and make the bot cite the new file. Details in [CHALLENGE.md](CHALLENGE.md).

## What you learned

- RAG = retrieve, augment, generate. The model answers from text it read a second ago, so it can cite it.
- Indexing runs once per change (load, split, embed, store). The question pipeline runs per question.
- Embeddings drive the search; the chat model only ever reads plain text.
- Stable ids keep the index clean; a distance gate and a refusal rule keep the bot honest.
- Measure retrieval with golden questions before you tune anything.

## Next episode

**Episode 08: Vector Databases and Chunking, Tested** - `refunds.md#4` held the main refund rule but lost its heading and missed the top 3. Which way of splitting finds answers best? We stop guessing and measure. -> `ep08-chunk-lab` (coming soon)

---

Dabbaloop, its people, prices and rules are invented for teaching. Any match with a real business is a coincidence.
