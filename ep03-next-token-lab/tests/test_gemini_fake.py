"""
tests/test_gemini_fake.py - the Gemini steps (12 and 13) with fake models: no key, no internet, same result every time.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

import gemini_limit
import gemini_stream
from testing.fakes import FakeChatModel, fake_llm, gemini_style_reply

ANSWER = "A chatbot writes one token at a time, so it can show each piece as soon as it is ready."
BILL = {"input_tokens": 31, "output_tokens": 14, "total_tokens": 45}


class FakeStreamWithBill(FakeChatModel):
    """fake_llm() streams word by word but sends no token counts. This one adds them, and a finish reason,
    to the last chunk, the way a real provider reports them."""

    usage: dict = BILL
    reason: str = "STOP"

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        chunks = list(super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs))
        last = chunks[-1].message
        chunks[-1] = ChatGenerationChunk(message=AIMessageChunk(
            content=last.content, usage_metadata=self.usage,
            response_metadata={"finish_reason": self.reason}, chunk_position="last"))
        yield from chunks


def cut_off_reply(text="A large language model first reads", reason="MAX_TOKENS", thinking=0):
    usage = {"input_tokens": 25, "output_tokens": 20, "total_tokens": 45}
    if thinking:
        usage["output_token_details"] = {"reasoning": thinking}
    return AIMessage(content=text, response_metadata={"finish_reason": reason}, usage_metadata=usage)


# ---------------------------------------------------------------------------
# Step 12 - streaming and the token bill
# ---------------------------------------------------------------------------

def test_stream_arrives_in_pieces_and_joins_up():
    full, pieces, first = gemini_stream.stream_answer(fake_llm(ANSWER), gemini_stream.build_messages(), show=False)
    assert len(pieces) > 10                       # the fake sends word by word
    assert "".join(pieces) == ANSWER
    assert full.text == ANSWER
    assert first >= 0


def test_stream_main_prints_chunks_and_says_when_counts_are_missing(capsys):
    assert gemini_stream.main(llm=fake_llm(ANSWER)) == 0
    out = capsys.readouterr().out
    assert "A| |chatbot| |writes|" in out         # a | after every chunk
    assert "Tokens: not reported by this model" in out
    assert "Finish reason: not reported" in out


def test_stream_main_adds_up_the_token_bill(capsys):
    assert gemini_stream.main(llm=FakeStreamWithBill(messages=iter([ANSWER]))) == 0
    out = capsys.readouterr().out
    assert "Tokens: 31 in, 14 out, 45 total" in out
    assert "Finish reason: STOP" in out


def test_gemini_style_content_blocks_still_print_as_text(capsys):
    gemini_stream.main(llm=fake_llm(gemini_style_reply("Hello from the fake")))
    out = capsys.readouterr().out
    assert "Hello|" in out and "[{" not in out     # .text, never the list of blocks


def test_the_prompt_reaches_the_model():
    llm = fake_llm(ANSWER)
    gemini_stream.stream_answer(llm, gemini_stream.build_messages(), show=False)
    system, human = llm.received[0]
    assert human.content == gemini_stream.QUESTION


# ---------------------------------------------------------------------------
# Step 13 - the output limit and the finish reason
# ---------------------------------------------------------------------------

def test_limit_is_reported_when_it_fires(capsys):
    assert gemini_limit.main(llm=fake_llm(cut_off_reply())) == 0
    out = capsys.readouterr().out
    assert "A large language model first reads" in out
    assert "Finish reason: MAX_TOKENS" in out
    assert "The limit fired" in out


def test_thinking_tokens_are_shown(capsys):
    gemini_limit.main(llm=fake_llm(cut_off_reply(text="", thinking=20)))
    out = capsys.readouterr().out
    assert "(of which 20 thinking)" in out
    assert "(no visible text)" in out


def test_a_finished_answer_is_not_called_cut_off(capsys):
    gemini_limit.main(llm=fake_llm(cut_off_reply(text="Done.", reason="STOP")))
    assert "The model finished before the limit." in capsys.readouterr().out


@pytest.mark.parametrize("key, value, cut", [
    ("finish_reason", "MAX_TOKENS", True),     # Gemini
    ("finish_reason", "length", True),         # OpenAI
    ("stop_reason", "max_tokens", True),       # Claude
    ("done_reason", "length", True),           # Ollama
    ("finish_reason", "STOP", False),
])
def test_every_provider_name_for_the_limit(key, value, cut):
    message = AIMessage(content="...", response_metadata={key: value})
    assert gemini_stream.finish_reason(message) == value
    assert gemini_stream.hit_limit(message) is cut


def test_the_limit_is_passed_to_get_llm(monkeypatch):
    sent = {}

    def pretend_get_llm(**settings):
        sent.update(settings)
        return fake_llm(cut_off_reply())

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(gemini_limit, "get_llm", pretend_get_llm)
    assert gemini_limit.main() == 0
    assert sent == {"max_tokens": 20}


def test_ollama_names_the_limit_num_predict():
    assert gemini_limit.limit_settings("ollama", 20) == {"num_predict": 20}
    assert gemini_limit.limit_settings("gemini", 20) == {"max_tokens": 20}


@pytest.mark.parametrize("script", [gemini_stream, gemini_limit])
def test_no_key_gives_the_friendly_help(monkeypatch, capsys, script):
    for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert script.main() == 1
    out = capsys.readouterr().out
    assert "No API key found" in out
    assert "https://aistudio.google.com/apikey" in out
