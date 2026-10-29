"""
techniques.py - the prompt techniques, one small function each (Steps 2-6, 8, 9 and 12).

    zero-shot   Step 2   one plain sentence + the note, no examples (the baseline)
    template    Step 3   the same prompt as a ChatPromptTemplate: system + human, note in delimiters
    rtc         Step 4   role, task and context: a careful trip accountant with rules and a format
    few-shot    Step 5   two worked examples as message pairs; labels counted on all ten notes
    structured  Step 6   with_structured_output + the ExpenseList form -> a typed table
    cot         Step 8   chain of thought: direct answer vs step-by-step, both checked by settle()
    cot-hidden  Step 9   the working goes in a hidden field; the group sees only final_answer
    injection   Step 12  a note with an order to the AI inside: naive prompt vs guarded pipeline

Run:  python techniques.py zero-shot                 (needs your free Gemini key)
      python techniques.py template                  (offline: prints the two messages)
      python techniques.py rtc --show-prompt         (offline: the filled prompt, no model call)
      python techniques.py structured --bad-reply    (offline: a scripted broken reply is rejected)
      python techniques.py --help                    (every command)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
import re
from dataclasses import dataclass, field

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from common import RESULTS_DIR, friendly_error, get_note, load_notes, make_llm, model_label
from guard import check_expenses, order_lines
from llm import MissingAPIKeyError
from models import BalanceAnswer, Expense, ExpenseList, Settlement, Transfer, TripNote, rupees
from prompt_files import estimate_tokens, read_prompt, note_inputs
from settle import same_result, settle

SCHEMAS = {"ExpenseList": ExpenseList, "BalanceAnswer": BalanceAnswer}


# ---------------------------------------------------------------------------
# Step 2 - zero-shot: one plain string, no examples
# ---------------------------------------------------------------------------

ZERO_SHOT_SUMMARY = "Summarise this note from our trip group chat.\n\n"


def zero_shot_summary(note_text: str, llm=None) -> str:
    """The simplest prompt there is: an instruction and the note, in ONE user string."""
    llm = llm or make_llm()
    return llm.invoke(ZERO_SHOT_SUMMARY + note_text).text      # .text, not .content (Gemini 3+)


# ---------------------------------------------------------------------------
# Step 3 / 4 - a prompt file as a chat template, piped into the model
# ---------------------------------------------------------------------------

def show_messages(prompt_name: str, note_text: str, with_examples: bool = True) -> list:
    """Fill a prompt file with a note and print the messages the model would get (no model call)."""
    prompt = read_prompt(prompt_name)
    template = prompt.template(with_examples)
    messages = template.format_messages(**note_inputs(note_text))
    word = "message" if len(messages) == 1 else "messages"
    print(f"{prompt.version_id}: slots {template.input_variables}, {len(messages)} {word}\n")
    for message in messages:
        print(f"----- {message.type} -----\n{message.text}")
    print(f"-----\nabout {estimate_tokens(messages)} tokens")
    return messages


def run_text_prompt(prompt_name: str, note_text: str, llm=None, with_examples: bool = True) -> str:
    """prompt file -> ChatPromptTemplate | model | StrOutputParser -> plain text."""
    llm = llm or make_llm()
    chain = read_prompt(prompt_name).template(with_examples) | llm | StrOutputParser()
    return chain.invoke(note_inputs(note_text))


# ---------------------------------------------------------------------------
# Step 5 - few-shot labels, counted on all ten notes
# ---------------------------------------------------------------------------

CATEGORY_LINE = re.compile(
    r"(?P<payer>[A-Za-z]+)\s*\|\s*(?:₹|rs\.?\s*)?(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?:rupees)?\s*\|\s*(?P<label>[A-Za-z]+)",
    re.IGNORECASE,
)


def parse_category_lines(text: str) -> list[tuple[str, float, str]]:
    """'Meera | 1800 | stay' lines -> [('meera', 1800.0, 'stay'), ...]."""
    return [(m["payer"].lower(), float(m["amount"].replace(",", "")), m["label"].lower())
            for m in CATEGORY_LINE.finditer(text.replace("*", ""))]


def score_labels(reply: str, expenses: list[Expense]) -> int:
    """How many payments of the answer key got the right label in the reply."""
    labels = {(payer, amount): label for payer, amount, label in parse_category_lines(reply)}
    return sum(labels.get((e.payer.lower(), e.amount)) == e.category for e in expenses)


def compare_labels(notes: list[TripNote], llm) -> dict:
    """Label every note twice: without the examples (zero-shot) and with them (few-shot)."""
    totals = {"without": 0, "with": 0, "payments": 0, "perfect_without": 0, "perfect_with": 0}
    print(f"{'note':<14}{'without examples':>18}{'with examples':>16}")
    for note in notes:
        right = {}
        for mode, with_examples in (("without", False), ("with", True)):
            reply = run_text_prompt("expense_fewshot_v1", note.note, llm, with_examples)
            right[mode] = score_labels(reply, note.expenses)
            totals[mode] += right[mode]
            totals[f"perfect_{mode}"] += right[mode] == len(note.expenses)
        totals["payments"] += len(note.expenses)
        n = len(note.expenses)
        print(f"{note.id:>2} {note.slug:<11}{right['without']:>14}/{n}{right['with']:>14}/{n}")
    p = totals["payments"]
    print(f"{'total':<14}{totals['without']:>14}/{p}{totals['with']:>14}/{p}")
    print(f"every label right: {totals['perfect_without']} of {len(notes)} notes without examples, "
          f"{totals['perfect_with']} of {len(notes)} with examples")
    return totals


# ---------------------------------------------------------------------------
# Step 6 - structured output: the model fills in a typed form
# ---------------------------------------------------------------------------

def extract_expenses(note_text: str, llm=None, prompt_name: str = "expenses_v1") -> ExpenseList:
    """prompt | llm.with_structured_output(ExpenseList) -> an ExpenseList object, not text."""
    llm = llm or make_llm()
    chain = read_prompt(prompt_name).template() | llm.with_structured_output(ExpenseList)
    return chain.invoke(note_inputs(note_text))


def expense_table(expenses: list[Expense]) -> list[str]:
    rows = [f"  {'payer':<8}{'amount':>8}  {'category':<10}{'item':<26}shared_with"]
    for e in expenses:
        rows.append(f"  {e.payer:<8}{rupees(e.amount):>8}  {e.category:<10}{e.item[:25]:<26}{', '.join(e.shared_with)}")
    return rows


# A broken reply, written by us to show what validation does (it is not a model answer).
BROKEN_REPLY = {"expenses": [{"payer": "Meera", "item": "homestay", "amount": "eighteen hundred",
                              "category": "hotel", "shared_with": []}]}


def bad_reply_demo(note_text: str) -> str:
    """Send a scripted BROKEN reply through the real with_structured_output parser (offline)."""
    from testing.fakes import fake_llm, tool_call       # the course's pretend model

    llm = fake_llm(tool_call("ExpenseList", BROKEN_REPLY))
    try:
        extract_expenses(note_text, llm)
    except (ValidationError, OutputParserException) as error:
        return explain_validation(error)
    return "accepted (this should not happen)"


def explain_validation(error: Exception) -> str:
    """Pydantic's error list as short lines: where, what was wrong."""
    if not isinstance(error, ValidationError):
        return str(error).splitlines()[0]
    lines = [f"{error.error_count()} problems, so the reply was rejected:"]
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"])
        lines.append(f"  {where:<28}{item['msg']} (got {item['input']!r})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Steps 8, 9 and 13 - "who owes whom" from any technique, in one shape
# ---------------------------------------------------------------------------

TRANSFER_PHRASE = re.compile(
    r"(?P<who>[A-Za-z][A-Za-z']*)\s+(?:owes|owe|should pay|must pay|needs to pay|has to pay|pays|->|→)\s+"
    r"(?P<to>[A-Za-z][A-Za-z']*)\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*(?P<amount>\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_transfers(text: str) -> list[Transfer]:
    """Find every 'Arjun owes Meera 230' (also ₹230, Rs 230, 230 rupees) in a free-text reply."""
    transfers = []
    for match in TRANSFER_PHRASE.finditer(text.replace("*", "")):
        amount = float(match["amount"].replace(",", ""))
        if amount > 0:
            transfers.append(Transfer(from_person=match["who"].title(), to_person=match["to"].title(), amount=amount))
    return transfers


ALL_SQUARE = re.compile(r"\b(nobody owes|no one owes|everyone is square|all square)\b", re.IGNORECASE)


def read_transfers(text: str) -> list[Transfer] | None:
    """The transfers in a reply, [] if it says nobody owes anything, None if we cannot read it."""
    transfers = parse_transfers(text)
    if transfers or ALL_SQUARE.search(text):
        return transfers
    return None


def final_answer_part(text: str) -> str:
    """Everything after 'FINAL ANSWER:' (the working above it is not the answer)."""
    parts = re.split(r"final answer\s*:?", text, flags=re.IGNORECASE)
    return parts[-1] if len(parts) > 1 else ""


@dataclass
class Answer:
    """One technique's answer to 'who owes whom', in the same shape for every technique."""

    transfers: list[Transfer] | None      # None = no usable answer
    reply: str = ""                       # what the model wrote (or the typed object, as text)
    hidden: str = ""                      # chain-of-thought working we keep backstage
    held_back: list = field(default_factory=list)   # expenses the guard did not trust
    error: str = ""


def ask_transfers(prompt_name: str, note: TripNote, llm) -> Answer:
    """Ask one prompt file 'who owes whom' for one note. The prompt's output setting decides how to read it."""
    prompt = read_prompt(prompt_name)
    inputs = note_inputs(note.note)

    if prompt.output in SCHEMAS:
        # Typed answer. include_raw=True: a reply that does not fit the form comes back as an error, not a crash.
        chain = prompt.template() | llm.with_structured_output(SCHEMAS[prompt.output], include_raw=True)
        result = chain.invoke(inputs)
        if result["parsed"] is None:
            return Answer(None, error=f"reply did not fit the form: {str(result['parsing_error']).splitlines()[0]}")
        parsed = result["parsed"]
        if isinstance(parsed, BalanceAnswer):
            return Answer(parsed.final_answer, Settlement(transfers=parsed.final_answer).model_dump_json(),
                          hidden=parsed.reasoning)
        # ExpenseList: check every expense against the note, then let Python do the maths.
        kept, held_back = check_expenses(parsed.expenses, note.note, note.people)
        return Answer(settle(kept).transfers, parsed.model_dump_json(), held_back=held_back)

    text = (prompt.template() | llm | StrOutputParser()).invoke(inputs)
    if prompt.output == "working + final answer":
        answer_part = final_answer_part(text)
        working = text[: len(text) - len(answer_part)]
        return Answer(read_transfers(answer_part) if answer_part else None, text, hidden=working)
    return Answer(read_transfers(text), text)


def verdict(answer: Answer, note: TripNote) -> str:
    if answer.transfers is None:
        return "no answer"
    return "match" if same_result(answer.transfers, note.expenses) else "MISMATCH"


def short(transfers: list[Transfer] | None) -> str:
    if transfers is None:
        return "-"
    return ", ".join(f"{t.from_person}->{t.to_person} {rupees(t.amount)}" for t in transfers) or "nobody owes"


def append_log(path, record: dict) -> None:
    """One JSON line per model call, so you can read every reply later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def compare_cot(notes: list[TripNote], llm, log_path=RESULTS_DIR / "cot_log.jsonl") -> dict:
    """Step 8: the direct answer (balance_rtc_v1) vs chain of thought (balance_cot_v1), checked by settle()."""
    counts = {"direct": 0, "cot": 0, "fixed": [], "broke": []}
    print(f"{'note':<14}{'direct':<11}{'chain of thought':<18}Python (settle) says")
    for note in notes:
        result = {}
        for label, prompt_name in (("direct", "balance_rtc_v1"), ("cot", "balance_cot_v1")):
            answer = ask_transfers(prompt_name, note, llm)
            result[label] = verdict(answer, note)
            counts[label] += result[label] == "match"
            append_log(log_path, {"note": note.id, "prompt": prompt_name, "model": model_label(),
                                  "reply": answer.reply, "parsed": short(answer.transfers), "verdict": result[label]})
        if result["direct"] != "match" and result["cot"] == "match":
            counts["fixed"].append(note.id)
        if result["direct"] == "match" and result["cot"] != "match":
            counts["broke"].append(note.id)
        print(f"{note.id:>2} {note.slug:<11}{result['direct']:<11}{result['cot']:<18}{short(settle(note.expenses).transfers)}")
    print(f"\nmatches settle(): direct {counts['direct']} of {len(notes)}, chain of thought {counts['cot']} of {len(notes)}")
    print(f"chain of thought fixed notes: {counts['fixed'] or 'none'}; broke notes: {counts['broke'] or 'none'}")
    print(f"every reply is in {log_path.relative_to(log_path.parent.parent)}")
    return counts


# ---------------------------------------------------------------------------
# Step 12 - prompt injection: naive prompt vs the guarded pipeline
# ---------------------------------------------------------------------------

def injection_demo(note: TripNote, llm_naive, llm_guarded) -> dict:
    """Run the injection note through the bare prompt and through the guarded pipeline."""
    print(f"Note {note.id} ({note.slug}) has {len(order_lines(note.note))} line(s) that read like an order to the AI:")
    for line in order_lines(note.note):
        print(f"  {line!r}")
    python_says = short(settle(note.expenses).transfers)

    naive = ask_transfers("balance_zero_v1", note, llm_naive)
    print("\nA) naive: balance_zero_v1 (note pasted as it is, no rule, free text)")
    print(f"   model wrote: {naive.reply.strip()[:300]}")
    print(f"   read as:     {short(naive.transfers)}  ->  {verdict(naive, note)}")

    guarded = ask_transfers("expenses_v1", note, llm_guarded)
    print("\nB) guarded: expenses_v1 (delimiters + system rule + typed form) -> guard.py -> settle.py")
    for expense, reason in guarded.held_back:
        print(f"   held back for a person: {expense.payer} {rupees(expense.amount)} ({reason})")
    if not guarded.held_back:
        print("   held back for a person: nothing")
    print(f"   settle-up:   {short(guarded.transfers)}  ->  {verdict(guarded, note)}")
    print(f"\nPython on your answer key says: {python_says}")
    return {"naive": verdict(naive, note), "guarded": verdict(guarded, note)}


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def get_model(args):
    """The real model, or None after printing the friendly 'add your free key' help."""
    try:
        return make_llm(getattr(args, "thinking", None))
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return None


def pick_notes(spec: str) -> list[TripNote]:
    """'all' -> every note; '3,8,9' -> those notes."""
    if spec == "all":
        return load_notes()
    return [get_note(part) for part in spec.split(",")]


def run_command(args) -> int:
    note = get_note(args.note) if hasattr(args, "note") else None

    if args.command == "template":
        show_messages("summary_v1", note.note)
        return 0
    if getattr(args, "show_prompt", False):
        names = {"zero-shot": None, "rtc": "summary_v2", "few-shot": "expense_fewshot_v1",
                 "structured": "expenses_v1", "cot": "balance_cot_v1", "cot-hidden": "balance_cot_v2"}
        if args.command == "zero-shot":
            print(ZERO_SHOT_SUMMARY + note.note)
        elif args.command == "injection":
            show_messages("balance_zero_v1", note.note)
            print()
            show_messages("expenses_v1", note.note)
        else:
            sample = note or load_notes()[0]
            messages = show_messages(names[args.command], sample.note)
            if args.command == "few-shot":
                plain = read_prompt("expense_fewshot_v1").template(False).format_messages(**note_inputs(sample.note))
                print(f"the two examples add about {estimate_tokens(messages) - estimate_tokens(plain)} tokens to every call")
        return 0
    if args.command == "structured" and args.bad_reply:
        print("A scripted broken reply (written by us, not by a model):")
        print(f"  {json.dumps(BROKEN_REPLY)}\n")
        print(bad_reply_demo(note.note))
        return 0

    llm = get_model(args)
    if llm is None:
        return 1
    print(f"Model: {model_label()}\n")
    try:
        if args.command == "zero-shot":
            print(zero_shot_summary(note.note, llm))
        elif args.command == "rtc":
            print(run_text_prompt("summary_v2", note.note, llm))
        elif args.command == "few-shot":
            compare_labels(pick_notes(args.notes), llm)
        elif args.command == "structured":
            result = extract_expenses(note.note, llm)
            print(f"{type(result).__name__} with {len(result.expenses)} Expense objects:")
            print("\n".join(expense_table(result.expenses)))
        elif args.command == "cot":
            compare_cot(pick_notes(args.notes), llm)
        elif args.command == "cot-hidden":
            answer = ask_transfers("balance_cot_v2", note, llm)
            print(f"Note {note.id} ({note.slug}), prompt balance_cot_v2. Shown to the group:")
            for transfer in answer.transfers or []:
                print(f"  {transfer.line()}")
            print(f"\nPython check: {verdict(answer, note)} (settle says {short(settle(note.expenses).transfers)})")
            append_log(RESULTS_DIR / "reasoning_log.jsonl", {"note": note.id, "model": model_label(),
                                                             "reasoning": answer.hidden, "final": short(answer.transfers)})
            if args.show_reasoning:
                print(f"\nThe hidden working (never shown to the group):\n{answer.hidden}")
            else:
                print(f"(the working, {len(answer.hidden):,} characters, is in results/reasoning_log.jsonl;"
                      " add --show-reasoning to print it)")
        elif args.command == "injection":
            injection_demo(note, llm, llm)
    except (ValidationError, OutputParserException) as error:
        print(f"The reply did not fit the form.\n{explain_validation(error)}")
        return 1
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt techniques, one command per step.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text, default_note in [
        ("zero-shot", "Step 2: one plain string, no examples", "1"),
        ("template", "Step 3: print the chat template's messages (offline)", "1"),
        ("rtc", "Step 4: role, task and context", "1"),
        ("structured", "Step 6: typed expenses with with_structured_output", "1"),
        ("cot-hidden", "Step 9: chain of thought with the working hidden", "9"),
        ("injection", "Step 12: naive prompt vs guarded pipeline", "10"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--note", default=default_note, help=f"note number or name (default {default_note})")
        if name != "template":
            command.add_argument("--show-prompt", action="store_true", help="print the prompt only (no key needed)")
        if name in ("cot-hidden",):
            command.add_argument("--show-reasoning", action="store_true", help="also print the hidden working")
        if name == "structured":
            command.add_argument("--bad-reply", action="store_true", help="offline: a scripted broken reply is rejected")
    for name, help_text in [("few-shot", "Step 5: labels with and without examples, all ten notes"),
                            ("cot", "Step 8: direct answer vs chain of thought, checked by settle()")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--notes", default="all", help="'all' or numbers like 3,8,9")
        command.add_argument("--show-prompt", action="store_true", help="print the prompt only (no key needed)")
    for command in commands.choices.values():
        command.add_argument("--thinking", choices=["low", "medium", "high"],
                             help="Gemini thinking level (3.8 Flash: low, medium or high)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_command(args)
    except ValueError as error:       # a note or prompt file that does not exist or has a typo
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
