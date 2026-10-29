# Episode 11: LangChain Chains the Modern Way: Pipes, Parallel Steps, and Letting Python Do the Maths

> Part of **AI Engineering From Zero** by Ravi Shanker Singh (TechWithRavi).
> Video: link added when the episode goes live  |  Code: https://github.com/replyravi/ai-engineering-course/tree/main/ep11-party-planner-chains

## What you will build

`party.py`, a **Party Planner** pipeline. You type one sentence, and it gives you back a menu, a typed grocery
list, a shopping list costed from **your own** `prices.csv`, a budget check, cheaper swaps when the bill is too
high, and an invite plus a host checklist written side by side. The model does the writing. Python does every
rupee of maths.

```text
$ python party.py "South Indian lunch for 12 guests, budget 6000 rupees"
Party request: South Indian lunch for 12 guests, budget 6000 rupees
Read by Python: 12 guests, budget Rs 6,000

== Menu (model) ==
<four dishes written by Gemini>

== Grocery list (model, typed): <N> items -> shopping list (Python) ==
Shopping list for 12 guests (prices: data/prices.csv, SAMPLE prices)
Item                 Per guest    To buy       Price                      Cost
<item>               <amount>     <amount>     Rs <price> / <unit>     Rs <cost>
...
Total for <N> items                                                  Rs <exact total>

== Budget check (Python) ==
Within budget: Rs <left> left (bill Rs <total>, budget Rs 6,000).
...
Keys in the result: request, guests, budget, menu, items, bill, check, swaps, invite, checklist
Model calls this run: 4   (<seconds> s; every rupee above was added up by Python)
```

(This is the layout, not a recorded Gemini run. The lines Python writes, like `Read by Python` and the
`Keys` line, come out exactly like this. The menu, the grocery list and so the rupee totals depend on what
Gemini writes in your run.)

**You will learn:**

- What a **chain** is: a pipe of runnables that all share `invoke`, `stream` and `batch`, and when a pipe beats one big prompt.
- `prompt | llm | StrOutputParser()`, invoked with a **dict of named slots**, and why the parser matters.
- How to hand step one's output to step two by **name**, and keep every step's result with `RunnablePassthrough.assign`.
- How to run independent steps side by side with `RunnableParallel`, and put plain Python in the pipe with `RunnableLambda`.
- Why a model's arithmetic can be wrong, and how a **calculator tool** with `create_agent` fixes it, without ever calling `eval()`.
- How to count model calls and test a whole pipeline offline with a fake model.

> **Seen an older tutorial?** It used the classic chain classes. In 2026 we build the same ideas with LCEL
> (LangChain Expression Language: the `|` pipe) and tools:
>
> | Classic class | What it did | 2026 way (this episode) |
> |---|---|---|
> | `LLMChain` | prompt + model | `prompt \| llm \| StrOutputParser()` (Step 2) |
> | `SimpleSequentialChain` | step one's text feeds step two | `{"menu": menu_chain} \| notes_chain` (Step 4) |
> | `SequentialChain` | several named inputs and outputs | `RunnablePassthrough.assign(...)`, `RunnableParallel(...)` (Steps 7, 9) |
> | `TransformChain` | plain Python between steps | `RunnableLambda(my_function)` (Step 6) |
> | `LLMMathChain` | model writes an expression, numexpr computes it | a `@tool` calculator + `create_agent` (Step 12) |
>
> In LangChain 1.x all five live only in the separate legacy package `langchain-classic`:
> `from langchain.chains import LLMChain` fails with `ModuleNotFoundError: No module named 'langchain.chains'`.

## Prerequisites

- Finished [Episode 06](../ep06-helpdesk-ladder) (the pipe `prompt | llm | parser`, `invoke`, `stream`, `batch`, and your first `create_agent`).
- [Episode 09](../ep09-prompt-lab) (structured output) and [Episode 10](../ep10-counter-bot) (reading a CSV) help. This episode recaps both in a line or two, so you can follow it straight after Episode 06.
- **Python 3.13** (3.11 and 3.12 also work). **Do not use Python 3.14 or 3.15 yet** - some course libraries (PyTorch, CrewAI) do not support them.
- The shared `.venv` and root `.env` with your **free** Gemini key from Episode 1 ([SETUP.md](../SETUP.md)).
- About 45 minutes. **No model is downloaded** in this episode, so there is no model licence to check. A full run of `party.py` makes 4 or 5 model calls on your free quota.

## What is in this folder

```text
ep11-party-planner-chains/
├── data/
│   └── prices.csv            24 SAMPLE grocery prices written by Ravi (edit them to your shop's prices)
├── chains.py                 Steps 2-5 and 9: the menu, notes, ingredients, swaps, invite and checklist chains
├── costing.py                Steps 6 and 10: read prices.csv, scale, cost and budget-check (plain Python)
├── party.py                  Steps 7, 8 and 10: the whole pipeline with RunnablePassthrough.assign
├── math_experiment.py        Step 11: the model versus Python on five party sums
├── calculator.py             Step 12: a safe calculator @tool and an agent that uses it
├── llm.py, testing/          shared helpers (synced from _template, do not edit here)
├── tests/                    offline tests (fake chat model, no key, no internet)
├── requirements.txt          points to the shared ../requirements-base.txt
└── CHALLENGE.md
```

Every file is already here. Type the code yourself as we go, or read along and run it.

## Step 0 - Get ready (about 2 minutes)

The whole course shares **one** virtual environment (`.venv`) and **one** `.env` file, both at the repo root (`ai-engineering-course/`). You created them once in Episode 1 ([SETUP.md](../SETUP.md)). New here? Do SETUP.md first, then come back.

### 0.1 Open the repo and activate the shared environment

Open the `ai-engineering-course` folder in VS Code, then open a terminal (**Terminal > New Terminal**).

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
cd ep11-party-planner-chains
```

**macOS / Linux:**

```bash
source .venv/bin/activate
cd ep11-party-planner-chains
```

Expected: your prompt starts with `(.venv)` and ends in `ep11-party-planner-chains`.

> [!WARNING]
> **Common errors**
> - Windows: `The term '.\.venv\Scripts\Activate.ps1' is not recognized` / macOS, Linux: `.venv/bin/activate` with `no such file or directory` - you are not in the repo root, or the `.venv` does not exist yet. `cd` to the `ai-engineering-course` folder first; if there is still no `.venv`, follow [SETUP.md](../SETUP.md).
> - Windows: `running scripts is disabled on this system` - allow scripts for this window only, then activate again: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

### 0.2 Make sure the libraries are up to date

Nothing new to install if you already ran Episode 1: this episode uses the shared `requirements-base.txt`
(`numexpr`, which the calculator needs, is already in it). If you pulled a newer version of the repo, update from this folder:

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
> - `ModuleNotFoundError: No module named 'numexpr'` (or `langgraph`) - the `.venv` is not active in this terminal. Activate it again (0.1).

---

## Step 1 - Check the setup

Make sure your key is found and Gemini answers. Your key lives in `ai-engineering-course/.env`. `llm.py` looks in
this folder and then in each parent folder, so it finds the root file by itself. Do not copy `.env` into episode folders.

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

Without a key you get help instead of a crash (real output with no `.env` file at all, first lines):

```text
Settings file: no .env found
Provider: gemini   Model: gemini-3.5-flash-lite


No API key found for LLM_PROVIDER=gemini.
Looked for: GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.

Get a FREE Gemini key (no credit card needed):
  1. Open https://aistudio.google.com/apikey
...
```

Every script in this folder prints that same help when the key is missing. The steps that never call a model
(`--show-prompt`, `--show-schema`, `costing.py`, `--count-calls`, `--answers-only`, `--tool-only` and the tests)
work without any key.

> **Want another AI?** Change ONE line in the root `.env`: `LLM_PROVIDER=openai`, `anthropic` or `ollama`. Nothing else changes.

> [!WARNING]
> **Common errors**
> - `No API key found for LLM_PROVIDER=gemini` - the `.env` file is missing, sits in the wrong folder (it belongs in `ai-engineering-course/`), is named `.env.txt`, or still says `paste-your-key-here`. The last line of the help tells you which `.env` file was found.
> - `429 RESOURCE_EXHAUSTED` - free-tier limit. Wait a minute (per-minute limit) or until the daily reset. Your limits: https://aistudio.google.com/rate-limit

## Step 2 - The menu chain

Picture a dosa counter. One person pours the batter, the next one runs the tawa, the next adds the filling, the
last one plates it. Each station has one job and passes the plate on. A **chain** is the same idea in code: a
row of small steps, joined with `|`, where each step's output is the next step's input.

| Station | Takes in | Hands on |
|---|---|---|
| `ChatPromptTemplate` | a dict of named slots | messages |
| the model (`get_llm()`) | messages | an AI message: text **plus** metadata |
| `StrOutputParser()` | an AI message | plain text |

Every station is a **runnable**, and so is the whole row. That is why the finished chain gets `invoke`,
`stream` and `batch` for free (Episode 06), and why you can swap one station without touching the others.
Use a chain when the job has clear stages you want to see, test or reuse. One question with one answer
needs no chain at all.

`chains.py` (the important lines):

```python
MENU_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MENU_SYSTEM),      # "You plan home-cooked party menus in India. ... exactly four dishes ..."
    ("human", "Party: {request}\nGuests: {guests}"),
])


def build_menu_chain(llm=None):
    """{'request', 'guests'} -> four dishes as plain text (Step 2)."""
    return MENU_PROMPT | _model(llm) | StrOutputParser()


menu_chain = build_menu_chain()
print(menu_chain.invoke({"request": "South Indian lunch", "guests": 12}))
```

The prompt has **two** slots, so we pass a dict with two keys. First look at the filled prompt (no model call, no key needed):

```bash
python chains.py menu --show-prompt
```

Expected output (exact):

```text
Template slots: guests, request
Input dict:     {'request': 'South Indian lunch', 'guests': 12}

----- system message -----
You plan home-cooked party menus in India. Reply with exactly four dishes, one per line, written as 'Dish - a few words about it'. No prices, no quantities, no extra text.
----- human message -----
Party: South Indian lunch
Guests: 12
```

Now run the chain:

```bash
python chains.py menu
```

Expected shape (Gemini writes its own dishes, so your wording and time will differ):

```text
Input dict: {'request': 'South Indian lunch', 'guests': 12}

<Dish 1 - a few words>
<Dish 2 - a few words>
<Dish 3 - a few words>
<Dish 4 - a few words>

(1 model call, <seconds> s)
```

Plan your own party with `python chains.py menu --request "Kerala sadya for a housewarming" --guests 20`.

**Why always a dict?** A prompt with exactly **one** slot also accepts a bare value: LangChain wraps
`chain.invoke("a picnic")` into `{"request": "a picnic"}` for you. With two slots it cannot guess, and it
stops (real output, also in the Common errors box). Pass a dict every time and the question never comes up.

> **Classic vs 2026.** Older tutorials wrote `LLMChain(llm=llm, prompt=prompt)` and called `.run(...)`. That class
> now lives only in `langchain-classic`, and it warns you (real output):
>
> ```text
> LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 2.0.0. Use `RunnableSequence, e.g., `prompt | llm`` instead.
> ```
>
> The pipe you just wrote **is** that replacement.

> [!WARNING]
> **Common errors**
> - `TypeError: Expected mapping type as input to ChatPromptTemplate. Received <class 'str'>.` - you passed a bare string to a prompt with two slots. Pass a dict: `{"request": ..., "guests": ...}`.
> - `KeyError: "Input to ChatPromptTemplate is missing variables {'guests'}.  Expected: ['guests', 'request'] Received: ['request']` - every `{slot}` needs a key in the dict.
> - `ModuleNotFoundError: No module named 'langchain.prompts'` - the old import path. Use `from langchain_core.prompts import ChatPromptTemplate`.

## Step 3 - Remove and restore the parser

What does the parser actually do? Take it out and look. Without it, the chain ends at the model, so you get the
model's whole **AI message**: the text in an envelope covered in stamps (metadata).

```python
raw_chain = MENU_PROMPT | llm                       # no parser
message = raw_chain.invoke(inputs)
print(type(message).__name__)                       # AIMessage
print(type(message.content).__name__)               # list on Gemini 3: a list of content blocks

handed_on = NOTES_PROMPT.invoke({"menu": message})  # what the NEXT prompt would receive
text = StrOutputParser().invoke(message)            # the parser on its own: the text, nothing else
```

```bash
python chains.py raw
```

Expected shape (the labels are exact; the dishes, the id and the metadata come from your run):

```text
Without a parser:  MENU_PROMPT | llm
  type(result)          -> AIMessage
  type(result.content)  -> list
  metadata it carries   -> finish_reason, model_name, model_provider, safety_ratings, <maybe more>

Hand that object to the next prompt and it gets ALL of this (first 300 characters):
  Menu: content=[{'type': 'text', 'text': '<the four dishes>', 'extras': {'signature': '<a long code>'}}] additional_kwargs={} response_metadata={'finish_reason': 'STOP', ... ...

With the parser:  StrOutputParser().invoke(result)
  isinstance(..., str)  -> True   (type name TextAccessor: LangChain's own kind of str)
  <the four dishes, as plain text>

(1 model call, <seconds> s)
```

Look at the middle block. Without a parser, the second prompt would not get "the menu". It would get the whole
envelope printed as text, signature, metadata and all. A model may still cope, but you are paying tokens for
noise and hoping for luck. The parser peels the envelope so only the text travels on.

The type name `TextAccessor` is just LangChain's own kind of string: `isinstance(text, str)` is `True`, and you
use it exactly like any string. Other providers may put a plain `str` in `.content` instead of a `list`;
`StrOutputParser` and `.text` give you the text either way.

> [!WARNING]
> **Common errors**
> - You see `[{'type': 'text', 'text': ...}]` on screen - you printed `.content`. Print `.text`, or end the chain with `StrOutputParser()`.
> - `AttributeError: 'TextAccessor' object has no attribute 'content'` - the chain already ends with a parser, so the result is plain text. Print it directly.

## Step 4 - Hand the menu to a second chain

A relay race where the baton has a **name** written on it. The first runner (the menu chain) writes the menu;
the baton says `menu`; the second runner (the notes chain) has a slot called `{menu}` and grabs exactly that.

```python
NOTES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NOTES_SYSTEM),     # "You are a friendly home cook. For each dish in the menu, write one line ..."
    ("human", "Menu:\n{menu}"),
])

menu_then_notes = {"menu": build_menu_chain(llm)} | build_notes_chain(llm)
print(menu_then_notes.invoke({"request": "South Indian lunch", "guests": 12}))
```

The dict `{"menu": menu_chain}` is a step of its own: it runs `menu_chain` with the input and puts the text it
gets back under the key `menu`. Because we parsed the menu in Step 3, the notes chain receives plain text.

```bash
python chains.py notes --show-prompt
```

Expected output (exact):

```text
----- system message -----
You are a friendly home cook. For each dish in the menu, write one line: the dish name, then one practical tip, like what to start first or prepare the day before.
----- human message -----
Menu:
<the menu written by the menu chain>
```

```bash
python chains.py notes
```

Expected shape (Gemini's wording will differ):

```text
Chain: {"menu": menu_chain} | notes_chain

<Dish 1>: <one tip>
<Dish 2>: <one tip>
<Dish 3>: <one tip>
<Dish 4>: <one tip>

(2 model calls, <seconds> s)
```

Two model calls, one after the other, and the notes are for exactly the menu written in **this** run. You cannot
see that menu yet, because the dict step only hands it on. Do not fix that by calling `menu_chain.invoke(...)`
a second time to print it: that is a new call, and a model rarely writes the same menu twice. Step 7 keeps it properly.

> **Classic vs 2026.** Older tutorials used `SimpleSequentialChain(chains=[chain_one, chain_two])`. It still runs
> from `langchain_classic.chains` (version 1.0.8 prints no warning for it), but it lives only in the legacy package.
> `{"menu": chain_one} | chain_two` is the same hand-off, with the baton's name written down.

> [!WARNING]
> **Common errors**
> - `KeyError: "Input to ChatPromptTemplate is missing variables {'menu'}..."` - the dict key and the slot name must match exactly (`"menu"` and `{menu}`).
> - The notes talk about different dishes than you expected - you probably printed a separate `menu_chain.invoke(...)`. That was another call with another menu.

## Step 5 - Typed ingredients

The next step is for **code**, not for a person: Python needs item names and numbers it can trust. So the model
fills in a form (Episode 09 called this structured output): a Pydantic class whose fields, types and
descriptions **are** the instructions. We keep the form in `costing.py`, because that is where Python reads it.

```python
class Ingredient(BaseModel):
    """One grocery item, as the amount ONE guest needs."""

    item: str = Field(description="The item name, copied exactly from the price list")
    qty_per_person: float = Field(description="How much ONE guest needs, in the unit below")
    unit: Literal["g", "ml", "piece"] = Field(
        description="g for solids, ml for liquids, piece for things you count")


class Ingredients(BaseModel):
    """The grocery list for the whole menu."""

    items: list[Ingredient] = Field(description="Every item the menu needs, one entry per item")
```

`Literal["g", "ml", "piece"]` means only those three words are allowed. The chain (`chains.py`) also pastes the
item names from **your** `prices.csv` into the prompt, so the model uses names our costing step can find:

```python
def build_ingredients_chain(llm=None, prices=None):
    """{'menu'} -> an Ingredients object, not text (Step 5)."""
    prompt = INGREDIENTS_PROMPT.partial(price_list=price_list_text(prices or load_prices()))
    return prompt | _model(llm).with_structured_output(Ingredients)
```

`.partial(price_list=...)` fills one slot now and leaves `{menu}` for later. Look at the form and the prompt (no key needed):

```bash
python chains.py ingredients --show-schema
python chains.py ingredients --show-prompt
```

Expected output (exact; the end of the schema, then the start and the end of the prompt):

```text
        "unit": {
          "description": "g for solids, ml for liquids, piece for things you count",
          "enum": [
            "g",
            "ml",
            "piece"
          ],
...
  "title": "Ingredients",
  "type": "object"
}
```

```text
----- system message -----
You turn a party menu into a grocery list. Give the amount ONE guest needs. Use item names from this price list, spelled exactly as written, with the unit shown:
- rice (g)
- toor dal (g)
...
- banana leaf (piece)
- papad (piece)
If the menu needs something that is not in the list, still add it with a short plain name. Leave out water and salt.
----- human message -----
Menu:
<the menu written by the menu chain>
```

Now the real thing:

```bash
python chains.py ingredients
```

Expected shape (the items and amounts come from Gemini):

```text
Chain: {"menu": menu_chain} | ingredients_chain
type(result) -> Ingredients with <N> items (an object, not text)

Item                    Per guest  Unit
<item>                      <amount>  g
<item>                      <amount>  ml
<item>                      <amount>  piece
...

(2 model calls, <seconds> s)
```

The result is an `Ingredients` object: `items.items[0].qty_per_person` is a real number your code can multiply.

> [!WARNING]
> **Common errors**
> - `The model's answer did not fit the Ingredients form (Step 5)` - the reply could not be read as the form. Run it again, or try `LLM_MODEL=gemini-3.8-flash` in `.env`.
> - Odd amounts, like 500 g of rice for one guest - the form checks the **types**, not the sense. The model's numbers are a guess; Step 6 shows them next to the totals so a person can spot them.
> - On Anthropic models `with_structured_output` uses tool calling instead of a JSON schema. You do not need to change anything.

## Step 6 - Scale and cost in Python

At the grocery counter nobody guesses the weight: the scale says 1.44 kg and the bill follows. Here the model's
"grams per guest" go to a plain Python function, and the **prices come from a file you control**. Reading a CSV
(Episode 10) takes one line: `csv.DictReader` turns every row into a dict like `{'item': 'rice', 'unit': 'kg', 'price_rupees': '64'}`.

`data/prices.csv` (the start):

```text
# SAMPLE prices written by Ravi for this course. NOT live market prices. Edit them to match your local shop.
# One row per item: the shop unit (kg, litre or piece) and the price in rupees for ONE unit.
# Lines that start with # are notes and are skipped. Add a row for anything the model asks for that is missing.
item,unit,price_rupees
rice,kg,64
toor dal,kg,160
```

`costing.py` (the important lines):

```python
def scale_and_cost(items, guests: int, prices: dict[str, Price]) -> Bill:
    ...
    to_buy = per_guest * guests / units_per_shop_unit          # 120 g x 12 / 1000 = 1.44 kg
    if shop_unit == "piece":
        to_buy = Decimal(math.ceil(to_buy))                    # nobody sells 1.8 coconuts: round UP
    cost = (to_buy * price.rupees).quantize(PAISA, rounding=ROUND_HALF_UP)
    ...
    total = sum((line.cost for line in lines), Decimal("0"))


def cost_step(state: dict) -> Bill:
    """Read 'items' and 'guests' from the running dict, return the bill."""
    return scale_and_cost(state["items"], state["guests"], load_prices())

step = RunnableLambda(cost_step)        # the same function, now a pipeline step
```

`Decimal` is Python's exact decimal number. With ordinary floats, `0.1 * 3` is `0.30000000000000004`; with
`Decimal` money adds up to the paisa. To run this step with no key at all, `costing.py` uses a grocery list we
**wrote by hand** (`SAMPLE_ITEMS`, a sambar rice, beans poriyal, rasam and semiya payasam lunch; not a model answer):

```bash
python costing.py
```

Expected output (exact):

```text
Price list: 24 items from data/prices.csv

Shopping list for 12 guests (prices: data/prices.csv, SAMPLE prices)
Item                 Per guest    To buy       Price                      Cost
rice                 120 g        1.44 kg      Rs 64 / kg             Rs 92.16
toor dal             35 g         0.42 kg      Rs 160 / kg            Rs 67.20
tamarind             8 g          0.096 kg     Rs 200 / kg            Rs 19.20
sambar powder        6 g          0.072 kg     Rs 480 / kg            Rs 34.56
rasam powder         4 g          0.048 kg     Rs 500 / kg            Rs 24.00
tomato               70 g         0.84 kg      Rs 40 / kg             Rs 33.60
drumstick            40 g         0.48 kg      Rs 120 / kg            Rs 57.60
beans                60 g         0.72 kg      Rs 80 / kg             Rs 57.60
coconut              0.15 piece   2 pieces     Rs 35 / piece          Rs 70.00
coconut oil          12 ml        0.144 litre  Rs 240 / litre         Rs 34.56
curd                 80 g         0.96 kg      Rs 70 / kg             Rs 67.20
vermicelli           20 g         0.24 kg      Rs 120 / kg            Rs 28.80
milk                 120 ml       1.44 litre   Rs 58 / litre          Rs 83.52
jaggery              25 g         0.3 kg       Rs 70 / kg             Rs 21.00
cashew nuts          5 g          0.06 kg      Rs 950 / kg            Rs 57.00
ghee                 8 g          0.096 kg     Rs 650 / kg            Rs 62.40
curry leaves bunch   0.1 piece    2 pieces     Rs 10 / piece          Rs 20.00
banana leaf          1 piece      12 pieces    Rs 6 / piece           Rs 72.00
papad                1 piece      12 pieces    Rs 3 / piece           Rs 36.00
------------------------------------------------------------------------------
Total for 19 items                                                   Rs 938.40
Not costed: asafoetida (no price in data/prices.csv)

RunnableLambda(cost_step).invoke(...) -> total Rs 938.40 (same function, now a step)

Budget check (Python decides):
  Within budget: Rs 5,061.60 left (bill Rs 938.40, budget Rs 6,000).
```

Three details to notice:

- **Coconut**: 0.15 per guest x 12 = 1.8, so Python buys **2** (pieces are rounded up).
- **Asafoetida** has no price yet, so it is listed as `Not costed`, not guessed. Add a row to `prices.csv` and run again.
- The `RunnableLambda` line gives the same Rs 938.40: wrapping a function changes nothing about its maths; it just gives it `invoke`, `batch` and `stream` so it fits in a pipe.

Try `python costing.py --guests 20` (expected total: `Rs 1,539.00`; coconuts become 3, banana leaves 20).

> **Classic vs 2026.** Older tutorials used `TransformChain(input_variables=[...], output_variables=[...], transform=fn)`.
> It still runs from `langchain_classic.chains`. `RunnableLambda(fn)` does the same with any function and no
> extra declarations.

> [!WARNING]
> **Common errors**
> - `Could not read the price list: prices.csv line 8: unit 'kgs' must be kg, litre or piece.` (real output after changing tamarind's unit to `kgs`) - fix that line of `data/prices.csv`. The other messages name the line too: a price that is not a number, a price of 0, an empty name, or an item listed twice.
> - `Not costed: milk (the list says g, prices.csv sells it per litre)` - the model used the wrong unit for that item. Python does not guess a conversion; it tells you.
> - `... the values are separated by ';'. Use commas (in Excel: save as 'CSV UTF-8 (comma delimited)').` - some Excel settings save CSV files with semicolons. Save again as "CSV UTF-8 (comma delimited)".

## Step 7 - Keep every step

A backpack that picks up one thing at every stop and never drops the old ones. `RunnablePassthrough.assign`
passes the whole dict on **and** adds one new key, computed from that dict:

```python
def build_plan_chain(llm=None):
    """Step 7: request -> menu -> items -> bill, keeping every key in one dict."""
    if llm is None:
        llm = get_llm()
    return (
        RunnablePassthrough.assign(menu=build_menu_chain(llm))           # model: 1 call
        | RunnablePassthrough.assign(items=build_ingredients_chain(llm))  # model: 1 call, typed
        | RunnablePassthrough.assign(bill=RunnableLambda(cost_step))      # Python: no call
    )
```

Python reads the guest count and the budget straight from your sentence (`read_request` in `party.py`), so the
model never touches those numbers. Write them as digits.

```bash
python party.py --plan-only
```

Expected shape (the first two lines and the `Keys` line are exact; the rest comes from Gemini and your prices):

```text
Party request: South Indian lunch for 12 guests, budget 6000 rupees
Read by Python: 12 guests, budget Rs 6,000

== Menu (model) ==
<four dishes>

== Grocery list (model, typed): <N> items -> shopping list (Python) ==
Shopping list for 12 guests (prices: data/prices.csv, SAMPLE prices)
<one line per item, like the Step 6 table>
Total for <N> items                                                  Rs <exact total>

Keys in the result: request, guests, budget, menu, items, bill
Model calls this run: 2   (<seconds> s; every rupee above was added up by Python)
```

Now the menu, the grocery list and the bill all come from **one** run, and you can print any of them.

**Tip.** To watch every step go in and out, add two lines before `invoke`:
`from langchain_core.globals import set_debug` and `set_debug(True)`. LangChain then prints each step's input and
output in colour (long, but great when a step surprises you). Episode 06 Step 14 showed LangSmith, the same view in a browser.

> **Classic vs 2026.** Older tutorials promised named inputs and outputs with `SequentialChain(chains=[...],
> input_variables=[...], output_variables=[...])`. `RunnablePassthrough.assign` gives you named outputs with
> nothing to declare: the key you write is the name.

> [!WARNING]
> **Common errors**
> - `How many guests? Write it as digits, like 'for 12 guests', or add --guests 12.` - Python reads digits, not words. Real output for `python party.py "Lunch for twelve, budget 6000 rupees"`.
> - `KeyError: 'items'` inside `cost_step` - the steps are in the wrong order. A step can only read keys that an earlier step added.

## Step 8 - Count model calls

Every model call costs time and free quota, so count them. Our fake model (the one the tests use) remembers
every prompt it received in `.received`, so counting is just `len(llm.received)`. This step runs on scripted
replies, so it needs no key and gives the same result every time:

```bash
python party.py --count-calls
```

Expected output (exact):

```text
Counting model calls with a SCRIPTED fake model (no key, no internet).
Every model reply below is scripted by us, not written by an AI.

A) Print each stage by re-running the chain up to it (what older tutorials did):
   print the menu           -> 1 model call    running total 1   (menu used: scripted menu 1)
   print the grocery list   -> 2 model calls   running total 3   (menu used: scripted menu 2)
   print the bill           -> 2 model calls   running total 5   (menu used: scripted menu 3)
   total: 5 model calls, and the three printouts came from 3 different menus

B) One run of build_plan_chain() with RunnablePassthrough.assign:
   keys: request, guests, budget, menu, items, bill
   total: 2 model calls, one menu all the way through (bill Rs 938.40)

C) The whole party pipeline, build_party_chain():
   budget Rs 6,000  Within budget: Rs 5,061.60 left (bill Rs 938.40, budget Rs 6,000).
                    4 model calls: menu, grocery list, invite, checklist
   budget Rs 800    Over budget by Rs 138.40 (bill Rs 938.40, budget Rs 800).
                    5 model calls: menu, grocery list, swaps, invite, checklist
```

Read A against B. Re-running the chain to print each stage paid for 5 calls instead of 2, **and** the menu you
printed was not the menu the bill was built from. With a real model that is just as true: each new call can
write a different menu. With `assign`, each model step runs once and every printout belongs to the same run.

In real runs, `party.py` and `chains.py` count calls with a tiny callback (`CallCounter` in `chains.py`) and print `Model calls this run`.

> [!WARNING]
> **Common errors**
> - `FakeChatModel ran out of scripted replies` - in your own tests, your code made one more model call than you scripted. That is exactly the kind of extra call this step is about.

## Step 9 - Invite and checklist in parallel

The invite and the host checklist both need the menu, but **not each other**. So they do not have to wait in a
line: `RunnableParallel` sends both at the same time and gives back one dict with both answers.

```python
def build_extras(llm=None):
    """Step 9: invite and checklist do not need each other, so they run side by side."""
    llm = _model(llm)
    return RunnableParallel(invite=build_invite_chain(llm), checklist=build_checklist_chain(llm))


extras = RunnablePassthrough.assign(menu=build_menu_chain(llm)) | build_extras(llm)
```

```bash
python chains.py parallel --show-prompt     # both prompts, no key needed
python chains.py parallel
```

Expected shape (Gemini writes the text):

```text
Chain: RunnablePassthrough.assign(menu=menu_chain) | RunnableParallel(invite=..., checklist=...)
Keys in the result: invite, checklist

== invite (printed only, never sent) ==
<a short invite with [date], [time] and [address] left blank>

== checklist ==
[ ] <six short lines, from the day before to serving time>

(3 model calls, <seconds> s)
```

Both calls are sent at the same moment, so you wait for the slower of the two instead of one after the other.
The time in brackets is your whole run, menu included, and it changes from run to run.

In `party.py` the same idea is one line: `RunnablePassthrough.assign(invite=..., checklist=...)`. Two keys in
one `assign` also run side by side, because `assign` builds a `RunnableParallel` inside.

**Safety Note.** The invite is printed text. Nothing in this project sends a message to anyone: you copy it,
fill in the blanks, and send it yourself. The prompt also asks the model not to invent names.

> [!WARNING]
> **Common errors**
> - `429 RESOURCE_EXHAUSTED` - two calls at the same moment can hit the per-minute limit. Wait a minute and run again.
> - The invite contains a made-up date or address - small models sometimes ignore "leave blanks". Edit it before you share it, or try `LLM_MODEL=gemini-3.8-flash`.

## Step 10 - Budget check: Python decides, the model suggests

Is Rs 938.40 more than Rs 800? That is a subtraction, not a question for a language model. `budget_check` in
`costing.py` decides. Only when the bill is **over**, the pipeline asks the model for cheaper swaps, and it
tells the model the gap and the three costliest items, worked out by Python:

```python
def swaps_if_over_budget(state):
    # Python decides. A Runnable returned from a RunnableLambda is run next with the
    # same input; plain text is just the answer. (Episode 12 builds a full router this way.)
    return swaps_chain if state["check"].is_over else NO_SWAPS
```

The swaps prompt ends with **"Do NOT write any prices, amounts or totals: our code does all the maths."**

First the decision on its own (no key):

```bash
python costing.py --budget 800
```

Expected output (exact, the last two lines):

```text
Budget check (Python decides):
  Over budget by Rs 138.40 (bill Rs 938.40, budget Rs 800).
```

Then the whole pipeline. With a real model your bill depends on its grocery list, so pick a budget **below the
total your Step 7 run printed** to see the swaps:

```bash
python party.py "South Indian lunch for 12 guests, budget 800 rupees"
```

Expected shape:

```text
...
== Budget check (Python) ==
Over budget by Rs <gap> (bill Rs <total>, budget Rs 800).

== Cheaper swaps (model, only when over budget) ==
Swap <X> for <Y> - <why it still works>
...

== Invite (model; printed only, never sent) ==
...
Keys in the result: request, guests, budget, menu, items, bill, check, swaps, invite, checklist
Model calls this run: 5   (<seconds> s; every rupee above was added up by Python)
```

Within budget you see `Not needed: the bill fits the budget.` and 4 model calls instead of 5. The swaps are
**suggestions**: Python did not re-cost them. Trying a swap is your job (and a good stretch for the challenge).

> [!WARNING]
> **Common errors**
> - No swaps although you expected some - your bill was below the budget. Look at the `Budget check` line and lower the budget.
> - The swaps contain prices anyway - the model ignored the rule. Treat any number it writes as a guess; the numbers that count are the ones in the Python table.

## Step 11 - The model versus Python on five party sums

A good shopkeeper chats warmly with every customer and still uses a calculator for the bill. Why? A model writes
a number the way it writes a word: as **tokens**, small pieces of text, one after another, each one the most
likely next piece. Nothing in that adds the digits up, so a long or exact sum can drift. Let us test it on our own
party sums. First Python's answers (no key):

```bash
python math_experiment.py --answers-only
```

Expected output (exact; question 1's long list of amounts is shortened here):

```text
1. Shopping list total (19 lines)
   Question:   Add up this shopping list, in rupees: rice 92.16, toor dal 67.20, tamarind 19.20, ... papad 36.00. What is the total?
   Expression: 92.16 + 67.20 + 19.20 + 34.56 + 24.00 + 33.60 + 57.60 + 57.60 + 70.00 + 34.56 + 67.20 + 28.80 + 83.52 + 21.00 + 57.00 + 62.40 + 20.00 + 72.00 + 36.00
   Python:     938.4

2. Caterer quote plus 18% GST
   Question:   A caterer quotes Rs 48,750 before tax. Add 18% GST. What is the total in rupees?
   Expression: 48750 * (100 + 18) / 100
   Python:     57525

3. GST inside a bill
   Question:   A catering bill of Rs 73,986 already includes 18% GST. How many rupees of it are GST?
   Expression: 73986 * 18 / (100 + 18)
   Python:     11286

4. Split among 17 guests
   Question:   Split Rs 73,986 equally among 17 guests. How much does each guest pay, to the paisa?
   Expression: 73986 / 17
   Python:     4352.117647

5. Wedding plates
   Question:   A wedding caterer serves 2,347 plates at Rs 1,869 per plate. What is the total in rupees?
   Expression: 2347 * 1869
   Python:     4386543

The 18% GST is an illustrative rate for practice, not tax advice.
```

Question 1 is the Step 6 shopping list: 19 line costs. Now ask the model the same five, **alone**, with a prompt
that asks for only the final number:

```bash
python math_experiment.py
```

Expected shape (the Python column is exact; the Model column and the Result come from **your** run):

```text
Model: gemini-3.5-flash-lite (gemini), answering alone with no tools.

#  Sum                                 Python (numexpr)      Model said   Result
1  Shopping list total (19 lines)                 938.4         <number>   match | OFF by <n>
2  Caterer quote plus 18% GST                     57525         <number>   ...
3  GST inside a bill                              11286         <number>   ...
4  Split among 17 guests                    4352.117647         <number>   ...
5  Wedding plates                               4386543         <number>   ...

Matches: <n> of 5 (to the paisa). This is ONE run of ONE model, not an accuracy rate.

What the model actually wrote:
  1. <the raw reply>
  ...
```

Whatever your run shows, read it carefully. If some rows say `OFF by`, there it is. If every row says `match`,
that is not proof either: the next run, or the next sum, may differ, and you could not have known which answers
were right without Python checking. One run proves nothing general, so we never quote an accuracy rate from it.
The lesson stands either way: **let the model talk, let Python count.**

> [!WARNING]
> **Common errors**
> - `NO NUMBER` in the Result column - the model replied in words. Look at "What the model actually wrote".
> - A `match` you think is wrong - we compare to the paisa (within 0.01), so 4352.12 matches 4352.117647.

## Step 12 - A calculator tool and an agent

So give the maths back to Python, on the model's request. The model writes an **expression**; a tool computes
the **number**. `calculator.py` has the tool (Episode 15 reuses it):

```python
@tool
def calculator(expression: str) -> str:
    """Calculate an arithmetic expression exactly, like '12 * 154' or '(938.40 / 12) * 17'.

    Use only numbers, + - * / ** and brackets. Use brackets to make the order clear.
    Write a percentage as / 100: 18% of 500 is '500 * 18 / 100'.
    Use this for EVERY sum, product, split or percentage. Never do maths in your head.
    """
    try:
        return format_number(calculate(expression))
    except CalculatorError as error:
        return f"Error: {error}"           # returned, not raised: the agent keeps running
```

`calculate()` first checks the text: **only** digits, `+ - * / ( ) .` and spaces are allowed (a comma inside a
number, like `48,750`, is removed first). Then it turns every
whole number into a decimal (`12` becomes `12.0`) and lets `numexpr` do the arithmetic, with no names and no
Python built-ins available. Run the tool by hand first (no key):

```bash
python calculator.py --tool-only
```

Expected output (exact):

```text
What the model sees:
  name:        calculator
  description: Calculate an arithmetic expression exactly, like '12 * 154' or '(938.40 / 12) * 17'.
               Use only numbers, + - * / ** and brackets. Use brackets to make the order clear.
               Write a percentage as / 100: 18% of 500 is '500 * 18 / 100'.
               Use this for EVERY sum, product, split or percentage. Never do maths in your head.
  arguments:   {"expression": {"title": "Expression", "type": "string"}}

The tool, run by hand (our Python, no model):
  calculator("12 * 154")                       -> 1848
  calculator("938.40 / 12 * 17")               -> 1329.4
  calculator("48,750 * 18 / 100")              -> 8775
  calculator("10 + 5 * 2")                     -> 20
  calculator("(10 + 5) * 2")                   -> 30
  calculator("73986 / 17")                     -> 4352.117647
  calculator("18% of 500")                     -> Error: write a percentage as / 100: 18% of 500 is 500 * 18 / 100.
  calculator("1 / 0")                          -> Error: division by zero.
  calculator("9**9**9")                        -> Error: the result is too big.
  calculator("__import__('os').system('ls')")  -> Error: only numbers, + - * / ** and brackets are allowed (found: ' _ e i l m o p r s t y).
```

Look at the last four. A percent sign gets a hint the model can act on. `9**9**9` as whole numbers would make
Python try to build a number with about 370 million digits (when we tried, it was still busy after two minutes);
as decimals it fails at once. And the last line is not maths at all: it is a Python command that would run `ls`
on your computer. Our check refuses it before anything is computed.

Now the agent. "Five more guests join, what is the new total?" The Rs 938.40 in the question is the Step 6 bill:

```python
def build_math_agent(llm=None):
    """The agent: a model + the calculator tool. Pass llm=fake_llm(...) in tests."""
    if llm is None:
        llm = get_llm()
    return create_agent(model=llm, tools=[calculator], system_prompt=AGENT_RULES)
```

```bash
python calculator.py
```

Expected shape (message 3 is our Python, so its number is exact; messages 2 and 4 come from Gemini):

```text
Question: Our South Indian lunch for 12 guests costs Rs 938.40 in groceries. Five more guests join and the cost per guest stays the same. What is the new total?

Message trace (4 messages):
  1. human  Our South Indian lunch for 12 guests costs Rs 938.40 in groceries. Five more guests join and the cost per guest stays the same. What is the new total?
  2. ai     asks for a tool: calculator({"expression": "<an expression, for example 938.40 / 12 * 17>"})
  3. tool   calculator -> 1329.4
  4. ai     <the answer, with Rs 1,329.40>
Tool calls: 1   Model calls: 2
```

The model may write the expression another way, for example `(938.40 / 12) * (12 + 5)`: any correct expression
gives `1329.4`. If it writes a wrong one, the tool computes the wrong sum exactly, so read line 2 too.

**Safety Note.** Never pass text a model wrote to Python's `eval()`. `eval` runs **any** Python, including the
last line above. `numexpr` is safer: it checks the text first and offers no Python built-ins. But under the hood
it still lets Python read the expression, which is why `calculate()` adds its own stricter check and the decimal
trick on top. And the tool **returns** its errors as text: an exception inside a tool would stop the whole agent run.

> **Classic vs 2026.** Older tutorials used `LLMMathChain.from_llm(llm)`. Inside, the model was asked to write the
> expression in a code block that starts with a ```` ```text ```` line, and `numexpr` computed only what it found in that
> block. A reply that already starts with `Answer:` is passed through **untouched**. We ran the classic chain with
> scripted replies (real output):
>
> ```text
> model wrote a ```text block  -> {'question': '17 guests at 938.40/12 each?', 'answer': 'Answer: 1329.4'}
> model wrote 'Answer: ...'    -> {'question': 'same question', 'answer': 'Answer: 938.40 / 12 * 17'}
> ```
>
> So a custom prompt that makes the model reply "Answer: ..." quietly switches the calculator off: you get the
> expression back, not the number. `LLMMathChain` lives in `langchain_classic`, marked deprecated since 0.2.13,
> and its deprecation note names `langchain.agents.create_agent` with a calculator tool as the replacement. That
> is what you just built, and the trace shows the calculation happening.

> [!WARNING]
> **Common errors**
> - Only 2 messages (human, ai) and no tool call - the model did the sum in its head. Check the answer against `--tool-only`, make the question clearly about a calculation, or try `LLM_MODEL=gemini-3.8-flash`.
> - A `tool` line that starts with `Error:` and then a second tool call - that is the loop working: the model read the error and fixed its expression.
> - `The agent stopped: it used all its steps without a final answer` - the model kept calling the tool. Our limit (`MAX_AGENT_STEPS = 10`) stopped it after 5 model calls.

## Step 13 - Run the tests

The tests use a **fake chat model** with scripted answers (`fake_llm`), scripted typed answers
(`structured_reply`) and scripted tool calls (`tool_call`). They need no key and block the internet on purpose.

```bash
python -m pytest
```

Expected output (real run; your time will differ):

```text
........................................................................ [ 65%]
.................................s....                                   [100%]
109 passed, 1 skipped in 1.63s
```

An `s` is a **skipped** test, not a failure: one check in `tests/test_llm.py` only runs in the CrewAI episodes.

What they check: `scale_and_cost` and `budget_check` on fixed inputs (worked out by hand), every `prices.csv`
error message, the calculator (including refused unsafe expressions and `9**9**9` refused in under a second), the
chain wiring and the call counts, the typed grocery list, the agent loop with a scripted tool call, and the
experiment flagging a wrong answer. How a pipeline test looks (from `tests/test_chains.py`):

```python
def test_plan_chain_keeps_every_key_with_two_calls():
    llm = fake_llm(MENU, structured_reply(SAMPLE))                  # model call 1: menu, call 2: grocery list
    result = build_plan_chain(llm).invoke(PARTY)
    assert list(result) == ["request", "guests", "budget", "menu", "items", "bill"]
    assert result["bill"].total == Decimal("938.40")
    assert len(llm.received) == 2                                   # one call per model step
    assert llm.received[1][-1].text == f"Menu:\n{MENU}"             # the grocery list saw THIS menu
```

The fake model is scripted, but the prompts, the pipeline, the costing and the tool are real code.

> [!WARNING]
> **Common errors**
> - `ModuleNotFoundError: No module named 'costing'` - run pytest from this folder (the one with `pytest.ini`).
> - `Tests must not use the internet` - a test called the real model. Pass `llm=fake_llm(...)` to the function.
> - `FakeChatModel ran out of scripted replies` - the code made one more model call than the test scripted.
> - A costing test fails after you edited `data/prices.csv` - some tests use the course's sample prices on purpose (for example the Rs 938.40 total). Keep a copy of the original file, or update the expected numbers.

## Fix-It Corner

| You see | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'langchain.chains'` | Old tutorial code. Use the pipe (Step 2) or the table at the top. |
| `TypeError: Expected mapping type as input to ChatPromptTemplate` | Pass a dict with one key per slot (Step 2). |
| `[{'type': 'text', 'text': ...}]` printed | End the chain with `StrOutputParser()` or print `.text` (Step 3). |
| The next prompt gets `content=... additional_kwargs=...` | You left out the parser between two chains (Step 3). |
| The printed menu does not match the bill | You re-ran a chain to print it. Use `RunnablePassthrough.assign` (Steps 7-8). |
| `Not costed: ...` | Add that item to `data/prices.csv`, or fix its unit (Step 6). |
| A number in a model answer looks off | Do not trust it: compute it in Python or with the calculator tool (Steps 11-12). |
| `Could not reach the model's server (ConnectError: ...)` | You are offline. The library retries for about half a minute first. Connect and run again. |

## Final challenge

Add a **dietary filter** step with `RunnableLambda` before the costing (Jain or vegan), and a **second-language
invite** inside the parallel block. Details in [CHALLENGE.md](CHALLENGE.md).

## What you learned

- A chain is a pipe of runnables; the whole pipe gets `invoke`, `stream` and `batch`. Pass a dict with one key per slot.
- `StrOutputParser` peels the AI message so only text travels to the next step.
- `{"menu": menu_chain} | notes_chain` hands step one's text to step two by name (the old `SimpleSequentialChain`).
- `RunnablePassthrough.assign` keeps every step's result with one call per step; `RunnableParallel` runs independent steps side by side.
- `RunnableLambda` puts plain, testable Python in the pipe; typed output (`with_structured_output`) makes the hand-off from model to Python safe.
- A model writes numbers as tokens, so let Python count: a `@tool` calculator on `numexpr`, never `eval()`, with `create_agent`.

## Next episode

**Episode 12: One Assistant, Three Paths** - Step 10 sent only over-budget bills to the swaps chain. Next we
route every question to the right chain: the handbook (RAG), a specialist prompt, or a safe, read-only SQL query.
-> [ep12-dabbaloop-help-desk](../ep12-dabbaloop-help-desk)

---

The party, the grocery list and the prices in `data/prices.csv` are sample data written for teaching. They are not
live market prices, and the 18% GST in Step 11 is an illustrative rate, not tax advice.
