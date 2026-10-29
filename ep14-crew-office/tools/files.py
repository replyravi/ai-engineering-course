"""
tools/files.py - Step 4: a file tool with a fence around it.

CrewAI's ready-made FileReadTool reads ANY file path the model writes (inside the working
folder). Our agents only need a few files, so our tool only opens those:

    read_lead_file   (Researcher)  leads/*.md and product.md
    read_data_file   (Analyst)     data/news.csv

Anything else, like ../.env with your API key, is refused. Fewest tools, smallest reach:
the "excessive agency" lesson from Episode 05.

Each line comes back with its file name and line number ("leads/kesar-crumb-bakehouse.md:10: ..."),
so every fact the Researcher writes can carry an exact source, and our code can check it.

    python -m tools.files leads/kesar-crumb-bakehouse.md     (the tool by hand, no model)
    python -m tools.files ../.env                            (refused)

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import sys
from pathlib import Path

from crewai.tools import BaseTool, ToolFailure
from pydantic import BaseModel, ConfigDict, Field

from notebook import CaseFile, Ledger

PROJECT = Path(__file__).resolve().parent.parent      # the episode folder
LEAD_FILES = ["leads/", "product.md"]
DATA_FILES = ["data/news.csv"]


def resolve_inside(path: str, allowed: list[str], root: Path = PROJECT) -> Path:
    """Return the real file for `path`, or raise PermissionError if it is outside the fence.

    allowed holds file names ("product.md") and folders ending in "/" ("leads/").
    resolve() follows "..", "~" tricks and links, so "leads/../../.env" is caught too.
    """
    target = (root / path).resolve()
    root = root.resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"{path!r} is outside this project folder")
    relative = target.relative_to(root).as_posix()
    for entry in allowed:
        if relative == entry or (entry.endswith("/") and relative.startswith(entry)):
            if not target.is_file():
                raise FileNotFoundError(f"there is no file {relative!r}")
            return target
    raise PermissionError(f"{relative!r} is not one of the files this tool may read ({', '.join(allowed)})")


def numbered(relative: str, text: str) -> str:
    """'leads/x.md:3: - City: Pune' - one line per non-empty line, with its number."""
    return "\n".join(f"{relative}:{number}: {line}"
                     for number, line in enumerate(text.splitlines(), start=1) if line.strip())


class ReadFileInput(BaseModel):
    path: str = Field(description="The file to read, relative to the project folder, "
                                  "for example leads/kesar-crumb-bakehouse.md")


class ScopedReadTool(BaseTool):
    """Read one allowed text file and return its lines with line numbers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "read_file"
    description: str = "Read one allowed text file."
    args_schema: type[BaseModel] = ReadFileInput
    allowed: list[str]
    root: Path = PROJECT
    case: CaseFile | None = None      # outreach crew: remember what was read
    ledger: Ledger | None = None      # research desk: remember every date and number in it

    def _run(self, path: str):
        try:
            target = resolve_inside(path, self.allowed, self.root)
            text = target.read_text(encoding="utf-8")
        except (PermissionError, FileNotFoundError, UnicodeDecodeError) as error:
            # ToolFailure = "this call FAILED" (Episode 13). The message tells the model what IS allowed.
            return ToolFailure(message=f"Refused: {error}")

        relative = target.relative_to(self.root.resolve()).as_posix()
        if self.case is not None:
            self.case.remember_file(relative, text.splitlines())
            self.case.note_call(self.name)
        if self.ledger is not None:
            self.ledger.record_text(self.name, text)
        return numbered(relative, text)


def lead_file_tool(case: CaseFile | None = None) -> ScopedReadTool:
    """The Researcher's tool: lead files and the product sheet only."""
    return ScopedReadTool(
        name="read_lead_file",
        description="Read one lead file (leads/<name>.md) or the product sheet (product.md). "
                    "Returns every line as '<file>:<line number>: <text>'. Use the '<file>:<line number>' "
                    "part as the source of each fact. Other files are refused.",
        allowed=LEAD_FILES,
        case=case,
    )


def data_file_tool(ledger: Ledger | None = None) -> ScopedReadTool:
    """The Analyst's tool: the invented news file only."""
    return ScopedReadTool(
        name="read_data_file",
        description="Read data/news.csv: invented headlines (date, ticker, headline) for the invented "
                    "tickers DEMO1, DEMO2 and DEMO3. Returns every line as '<file>:<line number>: <text>'.",
        allowed=DATA_FILES,
        ledger=ledger,
    )


if __name__ == "__main__":
    tool = lead_file_tool()
    path = sys.argv[1] if len(sys.argv) > 1 else "leads/kesar-crumb-bakehouse.md"
    print(f"Tool name:    {tool.name}")
    print(f"Description:  {tool.description}\n")
    result = tool.run(path=path)
    print(f"tool.run(path={path!r}) returned:")
    print(f"  {result.message}" if isinstance(result, ToolFailure) else result)
