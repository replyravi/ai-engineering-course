"""
counter_bot.py - the Kadak Kettle counter bot (Steps 11 to 15). Needs your free Gemini key.

    python counter_bot.py                       Step 11: buffer memory (every turn), kept in RAM
    python counter_bot.py --memory window       Step 12: only the last 2 exchanges
    python counter_bot.py --memory summary      Step 13: old turns squeezed into one note
    python counter_bot.py --tools               Step 14: menu answers looked up in the cafe's files, with tags
    python counter_bot.py --tools --save        Step 15: chats saved in state/counter_bot.sqlite (survive a restart)
    python counter_bot.py --customer vikram     start as another customer (default: asha)

In the chat, type a message, or one of these commands:
    /history          what is saved for this customer (the model can only see these messages)
    /customer NAME    switch to another customer (another thread_id)
    /forget           delete this customer's conversation
    /help             show the commands
    exit              stop the program

After every reply the token meter shows what the model was sent.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from memory_modes import MODES, build_bot, is_summary  # first: it switches on safe loading before LangGraph loads

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from common import CAFE_NAME, CAFE_RULES, GROUNDING_RULES, INDEX_DIR, STATE_DIR, friendly_error  # noqa: E402
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider  # noqa: E402
from token_meter import TokenMeter  # noqa: E402

STATE_FILE = STATE_DIR / "counter_bot.sqlite"
EXIT_WORDS = {"exit", "quit", "bye"}
MAX_AGENT_STEPS = 10      # safety limit for the agent loop (model -> tool -> model ...), as in Episode 06
HELP = "Commands: /history  /customer NAME  /forget  /help  exit"


# ---------------------------------------------------------------------------
# Where the conversations live: RAM, or a SQLite file
# ---------------------------------------------------------------------------

def open_checkpointer(save=False, path=STATE_FILE):
    """InMemorySaver keeps chats in RAM (gone when the program stops). SqliteSaver keeps them in a file."""
    if not save:
        return InMemorySaver()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)    # LangGraph may use it from another thread
    return SqliteSaver(connection)


def forget(checkpointer, thread_id):
    """Delete one customer's conversation. A chat is personal data: deleting it must really delete it."""
    checkpointer.delete_thread(thread_id)
    if isinstance(checkpointer, SqliteSaver):
        # SQLite only marks deleted rows as free space, so the old text can stay inside the file.
        # VACUUM rewrites the file without it, and the checkpoint empties the write-ahead log (-wal file).
        checkpointer.conn.execute("VACUUM")
        checkpointer.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def thread_id_for(customer):
    """One thread per customer: 'Asha ' and 'asha' are the same thread."""
    return customer.strip().lower()


def config_for(customer, meter=None):
    """The thread_id tells the checkpointer WHICH conversation to load and save."""
    config = {"configurable": {"thread_id": thread_id_for(customer)}, "recursion_limit": MAX_AGENT_STEPS}
    if meter is not None:
        config["callbacks"] = [meter]           # the meter sees every model call, summary calls too
    return config


def saved_messages(bot, customer):
    """The messages the checkpointer holds for this customer (empty list for a new customer)."""
    return bot.get_state(config_for(customer)).values.get("messages", [])


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def shown_path(path):
    """'state/counter_bot.sqlite' instead of the full path, when the file is inside this folder."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def one_line(text, width=100):
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def who(message):
    """A label for a saved message. The summary note is saved as a 'human' message, so we label it ourselves."""
    if is_summary(message):
        return "summary"
    if message.type == "human":
        return "customer"
    if message.type == "tool":
        return "tool result"
    if message.type == "ai" and message.tool_calls:
        return "bot (asks a tool)"
    return "bot"


def history_lines(messages):
    if not messages:
        return ["  (nothing saved for this customer)"]
    lines = []
    for number, message in enumerate(messages, start=1):
        if message.type == "ai" and message.tool_calls:
            text = ", ".join(f"{call['name']}({call['args']})" for call in message.tool_calls)
        else:
            text = message.text
        lines.append(f"  {number:>2}. {who(message):<18} {one_line(text)}")
    return lines


def tool_lines(new_messages):
    """The tools the bot called during this turn, so you can see where a fact came from."""
    lines = []
    for message in new_messages:
        if message.type == "ai" and message.tool_calls:
            for call in message.tool_calls:
                lines.append(f"  tool  | {call['name']}({call['args']})")
    return lines


def after_last_question(messages):
    """The messages this turn added: everything after the last customer message (summaries do not count)."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human" and not is_summary(messages[index]):
            return messages[index + 1:]
    return messages


# ---------------------------------------------------------------------------
# The chat loop
# ---------------------------------------------------------------------------

def chat(bot, checkpointer, customer="asha", read=input, write=print, meter=None, echo=False):
    """Read a line, answer it, show the meter. Repeat until 'exit'. Returns the last customer."""
    meter = meter if meter is not None else TokenMeter()
    count = len(saved_messages(bot, customer))
    if count:
        write(f"Welcome back, {customer}: {count} saved messages in this thread.")
    while True:
        try:
            line = read(f"\n[{customer}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if echo:
            write(line)                         # input from a file or a pipe: show what was "typed"
        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            break
        if line == "/help":
            write(HELP)
            continue
        if line == "/history":
            messages = saved_messages(bot, customer)
            write(f"Saved for {customer} (thread_id '{thread_id_for(customer)}'): {len(messages)} messages")
            for text in history_lines(messages):
                write(text)
            continue
        if line == "/forget":
            forget(checkpointer, thread_id_for(customer))
            write(f"Deleted every saved message of {customer}.")
            continue
        if line.startswith("/customer"):
            name = line[len("/customer"):].strip()
            if not name:
                write("Type a name, for example:  /customer vikram")
                continue
            customer = name
            count = len(saved_messages(bot, customer))
            write(f"Now serving {customer} (thread_id '{thread_id_for(customer)}', {count} saved messages).")
            continue
        if line.startswith("/"):
            write(f"Unknown command {line.split()[0]}. {HELP}")
            continue

        meter.new_turn()
        try:
            result = bot.invoke({"messages": [{"role": "user", "content": line}]}, config_for(customer, meter))
        except Exception as error:      # no internet, free-tier limit, ...: explain and keep the chat going
            write(f"Bot: {friendly_error(error)}")
            continue
        new = after_last_question(result["messages"])
        for text in tool_lines(new):
            write(text)
        write(f"Bot: {result['messages'][-1].text}")      # .text, not .content (Gemini 3+ sends blocks)
        for text in meter.turn_lines():
            write(f"  meter | {text}")
    write("Bye!")
    return customer


def main(argv=None, llm=None, read=input, write=print):
    parser = argparse.ArgumentParser(description=f"The {CAFE_NAME} counter bot.")
    parser.add_argument("--memory", choices=MODES, default="buffer", help="buffer (default), window or summary")
    parser.add_argument("--customer", default="asha", help="who is at the counter (default: asha)")
    parser.add_argument("--tools", action="store_true", help="look facts up in the cafe's files (Step 14)")
    parser.add_argument("--save", action="store_true", help="save chats in a SQLite file (Step 15)")
    parser.add_argument("--state-file", default=str(STATE_FILE), help=argparse.SUPPRESS)   # used by the tests
    args = parser.parse_args(argv)

    try:
        if llm is None:
            llm = get_llm()                 # Gemini by default (LLM_PROVIDER in .env)
            model = f"{get_model_name()} ({get_provider()})"
        else:
            model = type(llm).__name__      # a fake model in the tests
    except (MissingAPIKeyError, ValueError) as error:
        write(str(error).strip())           # friendly help instead of a traceback
        return 1

    tools, prompt = (), CAFE_RULES
    if args.tools:
        if not (INDEX_DIR / "chroma.sqlite3").exists():
            write("The cafe index is not built yet. Build it first with:  python cafe_index.py")
            return 1
        from cafe_tools import TOOLS, warm_up

        warm_up()                           # load the embeddings model now, not during the first question
        tools, prompt = TOOLS, CAFE_RULES + GROUNDING_RULES

    checkpointer = open_checkpointer(args.save, Path(args.state_file))
    bot = build_bot(args.memory, llm=llm, checkpointer=checkpointer, tools=tools, system_prompt=prompt)
    where = shown_path(args.state_file) if args.save else "RAM (gone when you exit)"
    write(f"{CAFE_NAME} counter bot | model: {model} | memory: {args.memory} | "
          f"tools: {'on' if tools else 'off'} | saved in: {where}")
    write(HELP)
    try:
        piped = read is input and not sys.stdin.isatty()     # lines come from a file or a pipe
        chat(bot, checkpointer, args.customer, read=read, write=write, echo=piped)
    finally:
        if isinstance(checkpointer, SqliteSaver):
            checkpointer.conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
