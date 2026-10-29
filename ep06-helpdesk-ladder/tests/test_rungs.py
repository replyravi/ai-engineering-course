"""
tests/test_rungs.py - rungs 1 to 4 (Steps 2-9), tested with FAKE models.

No key, no internet, same result every time. We script what the "AI" answers and
check OUR code: the messages and prompts we send, how we read the answers, the
paragraph ranking, the typed HelpAnswer, and the friendly help when something is missing.
"""

import httpx
import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import rung1_messages
import rung2_template
import rung3_context
import rung4_structured
from common import HANDBOOK_FILE, HOUSE_RULES, REFUSAL, is_refusal
from helpdesk_fixtures import KeywordEmbeddings, mini_handbook, no_keys, real_embeddings  # noqa: F401
from rung3_context import QUESTION, build_prompt, cited_tags, load_paragraphs, rank
from rung4_structured import HelpAnswer, check_source
from testing.fakes import fake_embeddings, fake_llm, gemini_style_reply, structured_reply


# ---------------------------------------------------------------------------
# Rung 1: messages, .text, usage, model id, the one-line switch (Steps 2 and 3)
# ---------------------------------------------------------------------------

def test_messages_are_system_then_human():
    messages = rung1_messages.build_messages("Hello?")
    assert [m.type for m in messages] == ["system", "human"]
    assert messages[0].text == HOUSE_RULES


def test_rung1_prints_text_usage_and_model_id(capsys):
    reply = AIMessage(
        content=[{"type": "text", "text": "Namaste! A plan brings lunch every day.", "extras": {"signature": "x"}}],
        usage_metadata={"input_tokens": 52, "output_tokens": 11, "total_tokens": 63},
        response_metadata={"model_name": "gemini-3.5-flash-lite"},
    )
    assert rung1_messages.main([], llm=fake_llm(reply)) == 0
    output = capsys.readouterr().out
    assert "Sending 2 messages:" in output
    assert "Namaste! A plan brings lunch every day." in output
    assert "[{" not in output                                   # we printed .text, not .content
    assert "Model id reported by the provider: gemini-3.5-flash-lite" in output
    assert "'input_tokens': 52" in output and "'total_tokens': 63" in output


def test_rung1_without_a_key_shows_where_to_get_one(no_keys, capsys):
    assert rung1_messages.main([]) == 1
    output = capsys.readouterr().out
    assert "https://aistudio.google.com/apikey" in output
    assert "Traceback" not in output


class OllamaNotRunning:
    """Stands in for ChatOllama when no Ollama app is running on this computer."""

    def invoke(self, messages):
        raise httpx.ConnectError("[Errno 61] Connection refused")


def test_switch_to_ollama_when_it_is_not_running(no_keys, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert rung1_messages.main([], llm=OllamaNotRunning()) == 1
    output = capsys.readouterr().out
    assert "Provider: ollama   Model: qwen3.5:4b" in output
    assert "ollama pull qwen3.5:4b" in output


def test_init_chat_model_with_prefix_builds_the_same_class_as_get_llm(no_keys, monkeypatch):
    from langchain.chat_models import init_chat_model

    from llm import get_llm

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")      # only builds, never sends
    direct = init_chat_model("google_genai:gemini-3.5-flash-lite")
    assert type(direct).__name__ == type(get_llm()).__name__ == "ChatGoogleGenerativeAI"


# ---------------------------------------------------------------------------
# Rung 2: template | model | parser, invoke, stream and batch (Steps 4 and 5)
# ---------------------------------------------------------------------------

def test_template_has_two_named_slots():
    assert set(rung2_template.PROMPT.input_variables) == {"topic", "audience"}


def test_filled_prompt_contains_the_house_rules_and_the_topic():
    messages = rung2_template.PROMPT.invoke({"topic": "a delivery slot", "audience": "a student"}).to_messages()
    assert messages[0].text == HOUSE_RULES
    assert messages[-1].type == "human"                               # Gemini needs a human turn last
    assert messages[-1].text == "Explain a delivery slot to a student in two short sentences."


def test_chain_returns_plain_text_even_for_gemini_style_replies():
    chain = rung2_template.build_chain(fake_llm(gemini_style_reply("Namaste! A slot is a time window.")))
    assert chain.invoke({"topic": "a delivery slot", "audience": "a student"}) == "Namaste! A slot is a time window."


def test_stream_sends_the_answer_in_pieces(capsys):
    chain = rung2_template.build_chain(fake_llm("Namaste! A slot is a time window."))
    answer = rung2_template.run_stream(chain, "a delivery slot")
    assert answer == "Namaste! A slot is a time window."
    assert "pieces)" in capsys.readouterr().out
    pieces = list(rung2_template.build_chain(fake_llm("one two three")).stream({"topic": "x", "audience": "y"}))
    assert len(pieces) > 1 and "".join(pieces) == "one two three"


def test_batch_answers_every_topic_in_order():
    llm = fake_llm("answer one", "answer two", "answer three")
    answers = rung2_template.run_batch(rung2_template.build_chain(llm), rung2_template.TOPICS)
    assert answers == ["answer one", "answer two", "answer three"]
    for topic, prompt in zip(rung2_template.TOPICS, llm.received):
        assert topic in prompt[-1].text                               # reply N belongs to topic N


def test_show_prompt_needs_no_key(no_keys, capsys):
    assert rung2_template.main(["--show-prompt"]) == 0
    output = capsys.readouterr().out
    assert "Template slots: audience, topic" in output
    assert "Explain a meal-box subscription to a first-time customer" in output


# ---------------------------------------------------------------------------
# Rung 3: paragraphs, ranking, the grounded prompt (Steps 6, 7 and 8)
# ---------------------------------------------------------------------------

def test_the_handbook_splits_into_numbered_paragraphs():
    paragraphs = load_paragraphs(HANDBOOK_FILE)
    assert len(paragraphs) >= 3
    assert [p["tag"] for p in paragraphs[:3]] == ["handbook.md#1", "handbook.md#2", "handbook.md#3"]
    assert not any(p["text"].startswith("#") or "<!--" in p["text"] for p in paragraphs)


def test_headings_and_comments_are_skipped(mini_handbook):
    paragraphs = load_paragraphs(mini_handbook)
    assert paragraphs == [
        {"tag": "handbook.md#1", "text": "Cut-off: skip lunch before 9:30 am."},
        {"tag": "handbook.md#2", "text": "Pausing: pause up to 10 days a month."},   # line break joined
    ]


def test_ranking_puts_the_cut_off_paragraph_first():
    hits = rank(QUESTION, load_paragraphs(HANDBOOK_FILE), KeywordEmbeddings())
    assert len(hits) == 3
    assert hits[0][1]["text"].startswith("Cut-off times")
    assert hits[0][0] >= hits[1][0] >= hits[2][0]                     # best first


def test_ranking_runs_with_the_template_fake_embeddings():
    paragraphs = load_paragraphs(HANDBOOK_FILE)
    first = rank(QUESTION, paragraphs, fake_embeddings())
    second = rank(QUESTION, paragraphs, fake_embeddings())
    assert [p["tag"] for _, p in first] == [p["tag"] for _, p in second]   # same text, same vectors
    assert all(-1.0 <= score <= 1.0 for score, _ in first)


def test_real_local_model_finds_the_cut_off_rule(real_embeddings):
    hits = rank(QUESTION, load_paragraphs(HANDBOOK_FILE), real_embeddings)
    assert hits[0][1]["text"].startswith("Cut-off times")


def test_the_refusal_rule_can_be_switched_off():
    with_rule = build_prompt(True).messages[0].prompt.template
    without = build_prompt(False).messages[0].prompt.template
    assert REFUSAL in with_rule and REFUSAL not in without
    assert build_prompt(True).messages[-1].prompt.template == "{question}"


def test_context_answer_sends_the_three_tagged_paragraphs():
    llm = fake_llm("Namaste! Skip it before 9:30 am [handbook.md#4].")
    answer, hits = rung3_context.answer_with_context(QUESTION, llm, KeywordEmbeddings())
    assert answer == "Namaste! Skip it before 9:30 am [handbook.md#4]."
    system, human = llm.received[0]
    for _, paragraph in hits:
        assert f"[{paragraph['tag']}]" in system.text
    assert human.text == QUESTION


def test_no_context_sends_no_handbook_text():
    llm = fake_llm("Namaste! Usually by 10 am.")
    rung3_context.answer_without_context(QUESTION, llm)
    sent = " ".join(m.text for m in llm.received[0])
    assert "Cut-off times" not in sent and "handbook.md#" not in sent


def test_cited_tags_are_found():
    assert cited_tags("Before 9:30 am [handbook.md#4]. Paused days move [handbook.md#5].") == \
        ["handbook.md#4", "handbook.md#5"]


def test_rung3_shows_the_search_even_without_a_key(no_keys, capsys):
    assert rung3_context.main([], embeddings=KeywordEmbeddings()) == 1
    output = capsys.readouterr().out
    assert "Top 3 handbook paragraphs" in output and "[handbook.md#4]" in output
    assert "https://aistudio.google.com/apikey" in output


def test_a_failed_model_download_is_not_called_a_model_call(capsys):
    class NoInternetEmbeddings(KeywordEmbeddings):
        def embed_documents(self, texts):
            raise OSError("We couldn't connect to 'https://huggingface.co' to load the files, "
                          "and couldn't find them in the cached files.")

    assert rung3_context.main([], embeddings=NoInternetEmbeddings()) == 1
    output = capsys.readouterr().out
    assert "Could not download the free local embeddings model" in output
    assert "The model call failed" not in output


def test_is_refusal_handles_curly_apostrophes():
    assert is_refusal("Namaste! I don’t know based on the handbook.")
    assert not is_refusal("Namaste! Skip it before 9:30 am.")


# ---------------------------------------------------------------------------
# Rung 4: typed answers (Step 9)
# ---------------------------------------------------------------------------

def test_structured_output_returns_a_help_answer():
    scripted = HelpAnswer(answer="Namaste! Skip it before 9:30 am.", source="handbook.md#4", confidence="high")
    result, hits = rung4_structured.ask_typed(QUESTION, fake_llm(structured_reply(scripted)), KeywordEmbeddings())
    assert isinstance(result, HelpAnswer)
    assert result.source == "handbook.md#4" and result.confidence == "high"
    assert check_source(result, hits).endswith("OK")


def test_source_check_catches_a_paragraph_we_did_not_send():
    hits = [(0.9, {"tag": "handbook.md#4", "text": "..."})]
    wrong = HelpAnswer(answer="x", source="handbook.md#9", confidence="high")
    assert "WARNING" in check_source(wrong, hits)
    nothing = HelpAnswer(answer=REFUSAL, source="none", confidence="low")
    assert check_source(nothing, hits).startswith("Source check: none")


def test_a_badly_filled_form_is_rejected():
    with pytest.raises(ValidationError) as error:
        HelpAnswer.model_validate({"answer": "Before 9:30 am.", "confidence": "very sure"})
    assert "source" in str(error.value) and "'high', 'medium' or 'low'" in str(error.value)


def test_rung4_prints_the_fields(capsys):
    scripted = HelpAnswer(answer="Namaste! Skip it before 9:30 am.", source="handbook.md#4", confidence="high")
    assert rung4_structured.main([], llm=fake_llm(structured_reply(scripted)), embeddings=KeywordEmbeddings()) == 0
    output = capsys.readouterr().out
    assert "HelpAnswer" in output
    assert "  source     : handbook.md#4" in output
    assert "  confidence : high" in output


def test_show_schema_needs_no_key(no_keys, capsys):
    assert rung4_structured.main(["--show-schema"]) == 0
    output = capsys.readouterr().out
    assert '"enum": [' in output and "2 validation errors for HelpAnswer" in output


# ---------------------------------------------------------------------------
# Step 14: the LangSmith status line never shows the key
# ---------------------------------------------------------------------------

def test_tracing_status_line(monkeypatch):
    from common import tracing_status

    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing_status() == "LangSmith tracing: off"
    monkeypatch.setenv("LANGSMITH_TRACING", "True")                  # LangSmith ignores a capital T
    assert tracing_status().startswith("LangSmith tracing: off (write LANGSMITH_TRACING=true in small letters")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert "LANGSMITH_API_KEY is missing" in tracing_status()
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy-key-for-tests")
    monkeypatch.setenv("LANGSMITH_PROJECT", "ep06-helpdesk-ladder")
    line = tracing_status()
    assert "ON (project: ep06-helpdesk-ladder)" in line and "dummy-key" not in line
