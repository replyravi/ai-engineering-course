# Setup (do this once for the whole course)

By the end of this page you will have:

- **Python 3.13** and **VS Code** with the Python extension
- the course code on your computer
- **one virtual environment** (`.venv`) at the repo root with every library the course needs
- a **free Gemini key** saved in **one `.env` file** at the repo root
- a setup check that says `All good`

It takes about 20 minutes, most of it download time. Episode 1 does all of this on screen with you.

> Tested on macOS (Apple Silicon). For Windows and Linux, every pinned package has a ready-made
> wheel for Python 3.11-3.13 (checked on 2026-10-23). If a step fails on your machine, open an issue
> on GitHub with the exact error message.

---

## 1. Install Python 3.13

**Why 3.13 and not the newest?** The big download button on python.org now gives you Python 3.15.
The course libraries are not ready for it yet: PyTorch has no ready-made packages for 3.15,
and CrewAI (Episodes 13-14) does not install on 3.14 or newer. Python **3.13** works with everything.
3.11 and 3.12 also work.

**Windows**

1. Open the **Python 3.13.17** release page, https://www.python.org/downloads/release/python-31317/ , and download "Windows installer (64-bit)".
2. In the installer, tick **Add python.exe to PATH**, then click **Install Now**.
3. Check it in PowerShell:

```powershell
py -3.13 --version
```

(If you use winget: `winget install Python.Python.3.13` does the same.)

**macOS** (Apple Silicon, macOS 14 or newer)

1. Open the **Python 3.13.17** release page, https://www.python.org/downloads/release/python-31317/ , and install the "macOS installer", or run `brew install python@3.13`.
2. Check it in Terminal:

```bash
python3.13 --version
```

**Linux**

- Fedora: `sudo dnf install python3.13`
- Debian / Ubuntu: `sudo apt install python3.13 python3.13-venv` if your release has it.
- Any distro (always works): install **uv**, which downloads Python for you:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
```

Expected output of the version check:

```text
Python 3.13.x
```

> **Intel Mac?** The current PyTorch release no longer ships for Intel Macs. You can follow the
> episodes without PyTorch in Google Colab, or use a Windows or Linux PC.

## 2. Install VS Code and the Python extension

1. Download VS Code from https://code.visualstudio.com/ and install it.
2. Open VS Code, click the **Extensions** icon (or press Ctrl+Shift+X, Cmd+Shift+X on a Mac),
   search for **Python** and install the one published by **Microsoft**.

## 3. Get the course code

**With git** (recommended, so you can pull updates):

```bash
git clone https://github.com/replyravi/ai-engineering-course.git
cd ai-engineering-course
```

**Without git:** on the GitHub page click **Code > Download ZIP**, unzip it, and rename the folder to
`ai-engineering-course` if it is called `ai-engineering-course-main`.

Then in VS Code: **File > Open Folder...** and pick the `ai-engineering-course` folder.
Open a terminal inside VS Code with **Terminal > New Terminal**. Every command below runs in that
terminal, from the repo root (the folder that contains `SETUP.md`).

## 4. Create the virtual environment and install the libraries

A virtual environment is a private box of libraries for this course, so it never clashes with other
Python projects on your computer. We make **one** for the whole course, at the repo root.

**Windows (PowerShell)**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-base.txt
pip check
```

**macOS**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-base.txt
pip check
```

**Linux**

```bash
python3.13 -m venv .venv          # or, with uv:  uv venv --python 3.13 --seed .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # skip this line if you have an NVIDIA GPU
pip install -r requirements-base.txt
pip check
```

**Why the extra PyTorch line on Windows and Linux?** It installs the small CPU-only build of PyTorch.
Without it, pip on Linux downloads the NVIDIA (CUDA) build, which is several GB. Nothing in this course
needs a GPU. On macOS the normal package is already the right one.

Expected: your prompt starts with `(.venv)`, and the last command prints:

```text
No broken requirements found.
```

Finally, tell VS Code to use this environment: press **Ctrl+Shift+P** (Cmd+Shift+P on a Mac), run
**Python: Select Interpreter**, and pick the one that shows `.venv`. New terminals then activate it for you.

> Every time you open a new terminal outside VS Code, activate the environment again from the repo root:
> `.\.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (macOS, Linux).

## 5. Get your free Gemini key

1. Open https://aistudio.google.com/apikey and sign in with your Google account.
2. Accept the terms. Click **Create API key** and copy it. No credit card is needed.
3. Keep it secret. Anyone with your key can use your quota.

Good to know (as of 2026-10-23, re-check on the page):

- The free tier has rate limits per Google Cloud project. See yours at https://aistudio.google.com/rate-limit
  A `429 RESOURCE_EXHAUSTED` error means you hit one: wait a minute, or until the next day.
- On the free tier, Google may use your prompts to improve its products. Never send private or company data.
- Recording a video or sharing your screen? Use a separate key and delete it afterwards in AI Studio.

## 6. Save the key in `.env` (once, at the repo root)

Copy the example file, then open `.env` in VS Code and paste your key after `GEMINI_API_KEY=`
(no quotes, no spaces).

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

The line should look like this, with your own key:

```text
GEMINI_API_KEY=AIza...your-key...
```

Every episode finds this one file by itself (`llm.py` looks in the episode folder, then in each parent folder).
Do not copy `.env` into episode folders. `.gitignore` already lists `.env`, so git never uploads it.

> **Windows tip:** File Explorer hides file extensions, so a file you create there can secretly be
> called `.env.txt`. Create and edit it in VS Code instead.

## 7. Check everything

```bash
python ep01-four-kinds-of-ai/check_setup.py
```

Expected output (your Python patch number and masked key will differ):

```text
AI Engineering From Zero - setup check

Python .................. 3.13.x OK
Virtual env ............. active (.venv)
langchain ............... 1.4.7
langchain-google-genai .. 4.4.2
scikit-learn ............ 1.9.1
pandas .................. 3.1.0
matplotlib .............. 3.11.2
sentence-transformers ... 6.2.0
torch ................... 2.14.1
python-dotenv ........... 1.2.4
.env file ............... found (ai-engineering-course/.env)
Provider ................ gemini
Model ................... gemini-3.5-flash-lite
API key ................. GEMINI_API_KEY found (AIza...xxxx)
.env ignored by git ..... yes

All good. Next: python first_call.py
```

(If you downloaded the ZIP instead of cloning, the last check says `yes (.gitignore lists .env)`.)

Then make your first call. You are still in the repo root, so the command includes the episode folder
(inside `ep01-four-kinds-of-ai` it is just `python first_call.py`, as the check says):

```bash
python ep01-four-kinds-of-ai/first_call.py
```

You should see a two-sentence answer from Gemini and the number of tokens it used.
Episode 1's README explains every line.

## 8. Later: the CrewAI environment (Episodes 13-14 only)

CrewAI pins older versions of some libraries, so it gets its own environment, also at the repo root.
Do this when you reach Episode 13, not now:

**Windows (PowerShell):**

```powershell
py -3.13 -m venv .venv-crewai
.\.venv-crewai\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-crewai.txt
```

**macOS / Linux:**

```bash
python3.13 -m venv .venv-crewai
source .venv-crewai/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-crewai.txt
```

It uses the same root `.env` file.

---

## Fix-it corner

| You see | Why | Fix |
|---|---|---|
| `py is not recognized` / `command not found: python3.13` | Python 3.13 is not installed, or not on PATH | Step 1 again. On Windows, re-run the installer and tick **Add python.exe to PATH**. |
| `running scripts is disabled on this system` | PowerShell blocks the activate script | `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`, then activate again (this window only). |
| `No matching distribution found for torch==2.14.1` | The `.venv` was made with Python 3.14 or 3.15 | Delete the `.venv` folder and redo step 4 with Python 3.13. |
| `Microsoft Visual C++ 14.0 or greater is required` | pip is trying to build from source: wrong Python version | Use Python 3.13 (all packages ship ready-made for it). |
| pip downloads `nvidia-...` files of several GB | Linux without the CPU PyTorch line | Cancel, run the `torch==2.14.1 --index-url .../cpu` line first, then the requirements. |
| `.env file ... NOT FOUND` | `.env` is missing, in the wrong folder, or is `.env.txt` | Step 6, in the repo root, using VS Code. |
| `API key ... MISSING` | The line still says `paste-your-key-here`, or has quotes/spaces | Paste the key right after `GEMINI_API_KEY=`. |
| `ModuleNotFoundError: No module named 'langchain'` | The environment is not active | Activate `.venv` (step 4) or select it in VS Code. |
