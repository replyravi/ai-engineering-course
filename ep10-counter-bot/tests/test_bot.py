"""
counter_bot.py tests: the chat loop and its commands, driven by scripted lines and a fake model.
"""

import counter_bot
from testing.fakes import fake_llm


def run_bot(lines, llm, argv=()):
    """Run counter_bot.main with typed lines. Returns (exit code, every printed line)."""
    typed = iter(lines)
    printed = []
    code = counter_bot.main(list(argv), llm=llm, read=lambda prompt: next(typed), write=printed.append)
    return code, printed


def test_commands_switch_customers_show_history_and_forget():
    llm = fake_llm("Hi Asha, one masala chai, no nuts, noted.", "Sorry, I don't know your name yet.", "You are Asha.")
    lines = ["Hi, I'm Asha. One masala chai, no nuts please.", "/customer vikram", "Who am I?",
             "/customer asha", "Who am I?", "/history", "/forget", "/history", "exit"]

    code, out = run_bot(lines, llm)

    assert code == 0
    assert "Now serving vikram (thread_id 'vikram', 0 saved messages)." in out
    assert "Now serving asha (thread_id 'asha', 2 saved messages)." in out
    assert "Saved for asha (thread_id 'asha'): 4 messages" in out
    assert "Deleted every saved message of asha." in out
    assert "Saved for asha (thread_id 'asha'): 0 messages" in out
    assert out[-1] == "Bye!"
    replies = [line for line in out if line.startswith("Bot: ")]
    assert replies == ["Bot: Hi Asha, one masala chai, no nuts, noted.", "Bot: Sorry, I don't know your name yet.",
                       "Bot: You are Asha."]
    meters = [line for line in out if line.startswith("  meter | model call: sent ")]
    assert [line.split(" sent ")[1].split(" ")[0] for line in meters] == ["2", "2", "4"]


def test_the_summary_note_is_labelled_in_history():
    # One model writes the replies AND the note. On turn 5 the note is written first, then the reply.
    llm = fake_llm("reply 1", "reply 2", "reply 3", "reply 4", "Asha, allergic to nuts.", "reply 5")
    lines = [f"turn {n}" for n in range(1, 6)] + ["/history", "exit"]

    code, out = run_bot(lines, llm, ["--memory", "summary"])

    assert code == 0
    history = out[out.index("Saved for asha (thread_id 'asha'): 6 messages") + 1:]
    assert history[0].split()[1] == "summary"
    assert any(line.startswith("  meter | summary call:") for line in out)


def test_a_failed_model_call_is_explained_and_the_chat_goes_on():
    code, out = run_bot(["Hello?", "exit"], fake_llm())      # no scripted reply: the fake raises an error

    assert code == 0
    assert any(line.startswith("Bot: The model call failed: RuntimeError") for line in out)
    assert out[-1] == "Bye!"


def test_no_key_prints_the_free_key_help(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    printed = []

    code = counter_bot.main([], read=lambda prompt: "exit", write=printed.append)

    assert code == 1
    assert printed[0].startswith("No API key found for LLM_PROVIDER=gemini.")
    assert "https://aistudio.google.com/apikey" in printed[0]


def test_tools_mode_asks_you_to_build_the_index_first(monkeypatch, tmp_path):
    monkeypatch.setattr(counter_bot, "INDEX_DIR", tmp_path)          # an empty folder: no index

    code, out = run_bot(["exit"], fake_llm(), ["--tools"])

    assert code == 1
    assert out == ["The cafe index is not built yet. Build it first with:  python cafe_index.py"]


def test_a_saved_chat_greets_the_customer_after_a_restart(tmp_path):
    state = ["--save", "--state-file", str(tmp_path / "chat.sqlite")]
    run_bot(["Hi, I'm Asha. One masala chai, no nuts please.", "exit"], fake_llm("Hi Asha, noted."), state)

    llm = fake_llm("You are Asha.")
    code, out = run_bot(["Who am I?", "exit"], llm, state)               # a second run of the program

    assert code == 0
    assert "Welcome back, asha: 2 saved messages in this thread." in out
    assert [m.text for m in llm.received[0] if m.type == "human"] == [
        "Hi, I'm Asha. One masala chai, no nuts please.", "Who am I?"]
