"""
Thread tests (Steps 11 and 15): one thread per customer, saved chats that survive a restart,
and /forget that really deletes. Fake models only; the SQLite files live in pytest's temp folder.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from counter_bot import forget, open_checkpointer
from memory_modes import build_bot
from testing.fakes import fake_llm

EPISODE = Path(__file__).resolve().parent.parent


def say(bot, text, thread):
    return bot.invoke({"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": thread}})


def saved(bot, thread):
    return bot.get_state({"configurable": {"thread_id": thread}}).values.get("messages", [])


def test_two_customers_never_see_each_other_s_messages():
    llm = fake_llm("Hi Asha, one masala chai, no nuts, noted.", "Sorry, I don't know yet.", "You are Asha.")
    bot = build_bot("buffer", llm=llm)

    say(bot, "Hi, I'm Asha. One masala chai, no nuts please.", "asha")
    say(bot, "Who am I?", "vikram")
    say(bot, "Who am I?", "asha")

    vikram_saw = [m.text for m in llm.received[1] if m.type != "system"]
    asha_saw = [m.text for m in llm.received[2] if m.type != "system"]
    assert vikram_saw == ["Who am I?"]                                # nothing from Asha's thread
    assert asha_saw[0] == "Hi, I'm Asha. One masala chai, no nuts please."
    assert len(saved(bot, "asha")) == 4 and len(saved(bot, "vikram")) == 2


def test_a_checkpointer_needs_a_thread_id():
    bot = build_bot("buffer", llm=fake_llm("hi"))
    with pytest.raises(ValueError, match="thread_id"):
        bot.invoke({"messages": [{"role": "user", "content": "Hi"}]})


def test_forget_deletes_one_customer_only():
    checkpointer = open_checkpointer(save=False)
    bot = build_bot("buffer", llm=fake_llm("Hi Asha", "Hi Vikram"), checkpointer=checkpointer)
    say(bot, "I'm Asha", "asha")
    say(bot, "I'm Vikram", "vikram")

    forget(checkpointer, "asha")

    assert saved(bot, "asha") == []
    assert len(saved(bot, "vikram")) == 2


def test_a_saved_chat_survives_a_restart_in_a_new_process(tmp_path):
    state_file = tmp_path / "counter_bot.sqlite"
    checkpointer = open_checkpointer(save=True, path=state_file)
    bot = build_bot("buffer", llm=fake_llm("Hi Asha, one masala chai, no nuts, noted."), checkpointer=checkpointer)
    say(bot, "Hi, I'm Asha. One masala chai, no nuts please.", "asha")
    checkpointer.conn.close()                                         # "kill the program"

    # A brand-new Python process opens the same file and asks "Who am I?"
    restart = (
        "import json, sys\n"
        "from counter_bot import open_checkpointer\n"
        "from memory_modes import build_bot\n"
        "from testing.fakes import fake_llm\n"
        "from pathlib import Path\n"
        "from langgraph.checkpoint.serde import _msgpack\n"
        "llm = fake_llm('You are Asha, and you asked for nothing with nuts.')\n"
        "bot = build_bot('buffer', llm=llm, checkpointer=open_checkpointer(True, Path(sys.argv[1])))\n"
        "bot.invoke({'messages': [{'role': 'user', 'content': 'Who am I?'}]}, {'configurable': {'thread_id': 'asha'}})\n"
        "print(json.dumps({'saw': [[m.type, m.text] for m in llm.received[0]],\n"
        "                  'strict': _msgpack.STRICT_MSGPACK_ENABLED}))\n"
    )
    done = subprocess.run([sys.executable, "-c", restart, str(state_file)], cwd=EPISODE,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr

    result = json.loads(done.stdout.strip().splitlines()[-1])
    texts = [text for _, text in result["saw"]]
    assert texts[1:] == ["Hi, I'm Asha. One masala chai, no nuts please.",
                         "Hi Asha, one masala chai, no nuts, noted.", "Who am I?"]
    assert result["strict"] is True                                   # saved chats are loaded in safe mode


def test_forget_removes_the_name_from_the_sqlite_file(tmp_path):
    state_file = tmp_path / "counter_bot.sqlite"
    checkpointer = open_checkpointer(save=True, path=state_file)
    bot = build_bot("buffer", llm=fake_llm("Hi Asha!", "Noted, Asha."), checkpointer=checkpointer)
    say(bot, "Hi, I'm Asha.", "asha")
    say(bot, "I'm allergic to nuts.", "asha")

    def name_on_disk():
        return sum(path.read_bytes().count(b"Asha") for path in tmp_path.glob("counter_bot.sqlite*"))

    assert name_on_disk() > 0
    forget(checkpointer, "asha")
    assert saved(bot, "asha") == []
    assert name_on_disk() == 0                                        # VACUUM + WAL checkpoint: really gone
    checkpointer.conn.close()
