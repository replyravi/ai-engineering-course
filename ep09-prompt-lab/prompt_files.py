"""
prompt_files.py - prompts live in versioned files, not inside the code (Step 14).

Every prompt of the lab is one small TOML file in prompts/, for example prompts/balance_rtc_v1.toml:

    name = "balance_rtc"            the prompt family
    version = 1                     bump it whenever you change the text
    technique = "role-task-context"
    output = "transfer lines"       what the answer looks like (techniques.py reads this)
    system = \"\"\"...\"\"\"            role and rules
    human = \"\"\"<note>{note}</note>\"\"\"  the input, inside delimiters
    [[examples]]                    optional worked examples (few-shot)

Why files? You can change a prompt without touching the code, compare two versions on the
same scoreboard, and every score says exactly which prompt text produced it.
The file name must be <name>_v<version>.toml, so a score can never point at the wrong text.

TOML is read with tomllib, which comes with Python 3.11 and newer: nothing extra to install.

Run:  python prompt_files.py                 (list every prompt file, offline)
      python prompt_files.py balance_rtc_v1  (show one prompt's messages, filled with note 1)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from common import PROMPTS_DIR, get_note

REQUIRED_KEYS = ["name", "version", "technique", "output", "human"]


class PromptFileError(ValueError):
    """A prompt file is missing, has a typo, or its name does not match its version."""


@dataclass
class PromptFile:
    """One prompt file, read from prompts/."""

    name: str
    version: int
    technique: str
    output: str
    human: str
    system: str = ""
    changes: str = ""
    examples: list[dict] = field(default_factory=list)
    path: Path | None = None

    @property
    def version_id(self) -> str:
        """balance_rtc + 1 -> 'balance_rtc_v1' (printed next to every score)."""
        return f"{self.name}_v{self.version}"

    def template(self, with_examples: bool = True) -> ChatPromptTemplate:
        """Build the ChatPromptTemplate: system rules, worked examples as message pairs, then the input."""
        messages = []
        if self.system:
            messages.append(("system", self.system))
        if with_examples and self.examples:
            # Each example becomes a human message (the note) and an AI message (the answer we want).
            pair = ChatPromptTemplate.from_messages([("human", self.human), ("ai", "{answer}")])
            messages.append(FewShotChatMessagePromptTemplate(example_prompt=pair, examples=self.examples))
        # The LAST message is always the human input. Gemini 3.5 Flash-Lite and newer reject a
        # request that ends with an AI message (that trick was called "prefill").
        messages.append(("human", self.human))
        return ChatPromptTemplate.from_messages(messages)


def read_prompt(name: str, folder=PROMPTS_DIR) -> PromptFile:
    """Read prompts/<name>.toml, e.g. read_prompt("balance_rtc_v1")."""
    path = Path(folder) / f"{name.removesuffix('.toml')}.toml"
    if not path.is_file():
        known = ", ".join(p.stem for p in sorted(Path(folder).glob("*.toml")))
        raise PromptFileError(f"No prompt file {path.name} in prompts/. Known prompts: {known}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise PromptFileError(f"{path.name} is not valid TOML: {error}") from None

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise PromptFileError(f"{path.name} is missing: {', '.join(missing)}")
    prompt = PromptFile(
        name=data["name"], version=int(data["version"]), technique=data["technique"],
        output=data["output"], human=data["human"], system=data.get("system", ""),
        changes=data.get("changes", ""), examples=data.get("examples", []), path=path,
    )
    if prompt.version_id != path.stem:
        raise PromptFileError(
            f"{path.name} says name={prompt.name!r} and version={prompt.version}, so it must be called "
            f"{prompt.version_id}.toml. Rename the file or fix the version, so scores never point at the wrong text."
        )
    return prompt


def list_prompts(folder=PROMPTS_DIR) -> list[PromptFile]:
    """Every prompt file in prompts/, sorted by name."""
    return [read_prompt(p.stem, folder) for p in sorted(Path(folder).glob("*.toml"))]


# ---------------------------------------------------------------------------
# Delimiters: the user's text is DATA, not instructions (Step 12)
# ---------------------------------------------------------------------------

def as_data(text: str) -> str:
    """Remove <note> and </note> tags from user text.

    Our prompts put the note between <note> and </note>. If a note could contain its own
    </note>, it could "close the box" early and write text that looks like it comes from us.
    """
    return re.sub(r"</?\s*note\s*>", "", text, flags=re.IGNORECASE)


def note_inputs(note_text: str) -> dict:
    """The input dict for every note prompt: {"note": cleaned note text}."""
    return {"note": as_data(note_text)}


def estimate_tokens(messages) -> int:
    """About 4 characters per token: a rule of thumb (the real count comes back with each reply)."""
    return round(sum(len(m.text) for m in messages) / 4)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="List or show the prompt files.")
    parser.add_argument("prompt", nargs="?", help="a prompt to show, e.g. balance_rtc_v1")
    parser.add_argument("--note", default="1", help="the note to fill it with (default 1)")
    args = parser.parse_args(argv)
    try:
        return show(args)
    except ValueError as error:        # an unknown prompt, a TOML typo, or a name/version mismatch
        print(error)
        return 1


def show(args) -> int:
    if args.prompt:
        prompt = read_prompt(args.prompt)
        note = get_note(args.note)
        template = prompt.template()
        if "note" in template.input_variables:
            inputs = note_inputs(note.note)
        else:                                   # the message prompt is filled from settle(), not the note
            from settle import settle
            inputs = {"trip": note.trip, "transfers": "\n".join(settle(note.expenses).lines())}
        messages = template.format_messages(**inputs)
        print(f"{prompt.version_id}  ({prompt.technique}, output: {prompt.output})\n")
        for message in messages:
            print(f"----- {message.type} -----\n{message.text}")
        print(f"-----\n{len(messages)} messages, about {estimate_tokens(messages)} tokens")
        return 0

    prompts = list_prompts()
    print(f"{'prompt version':<22}{'technique':<20}{'output':<26}examples")
    for prompt in prompts:
        print(f"{prompt.version_id:<22}{prompt.technique:<20}{prompt.output:<26}{len(prompt.examples)}")
    print(f"\n{len(prompts)} prompt files in prompts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
