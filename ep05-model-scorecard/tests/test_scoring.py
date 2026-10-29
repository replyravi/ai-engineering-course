"""
tests/test_scoring.py - the eval set, the markers, the runner and the scorecard, all offline.

No key and no internet: the "model" is a fake with scripted replies (testing/fakes.py),
and the scorecard tests use the invented practice data in practice/.

Run:  python -m pytest
"""

import json
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk

import run_eval
import scorecard
from eval_set import EVAL_FILE, EvalItem, EvalSetError, check_eval_set, load_eval_set
from testing.fakes import fake_llm, gemini_style_reply

EPISODE = Path(__file__).resolve().parent.parent
SAMPLE_SET = Path(__file__).resolve().parent / "sample_eval_set.jsonl"   # a copy of the sample, so your edits never break tests
KEY_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL"]


def no_keys(monkeypatch):
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def make_item(**fields) -> EvalItem:
    base = {"id": "t-01", "tag": "test", "check": "exact", "question": "What is 2 + 2?", "answer": "4"}
    return EvalItem.model_validate({**base, **fields})


# --- step 2: the eval set ---------------------------------------------------------

def test_your_eval_set_loads_cleanly():
    items, problems = check_eval_set(EVAL_FILE)
    assert problems == [], "fix eval_set.jsonl: run  python eval_set.py"
    assert items


def test_the_sample_set_has_20_items_in_5_tags():
    items = load_eval_set(SAMPLE_SET)
    assert len(items) == 20
    tags = [item.tag for item in items]
    assert {tag: tags.count(tag) for tag in set(tags)} == {
        "arithmetic": 4, "india_gk": 4, "json_format": 4, "hindi_to_english": 4, "refusal": 4}


def test_first_reads_only_the_first_lines():
    assert [item.id for item in load_eval_set(SAMPLE_SET, first=5)] == ["arith-01", "arith-02", "arith-03", "arith-04", "gk-01"]


def test_the_broken_example_shows_every_problem_with_its_line():
    items, problems = check_eval_set(EPISODE / "data" / "broken_eval.jsonl")
    assert [item.id for item in items] == ["arith-01"]
    assert problems[0] == 'line 2: check "exact" needs an "answer" field'
    assert problems[1].startswith("line 3: not valid JSON")
    assert problems[2] == "line 4: the id arith-01 is used twice (first on line 1)"
    assert problems[3].startswith("line 5: check:")
    with pytest.raises(EvalSetError):
        load_eval_set(EPISODE / "data" / "broken_eval.jsonl")


# --- step 3: marking answers ---------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    (" 3,330. ", "3330"), ("**Odisha**", "odisha"), ("₹239.20", "239.20"), ("Rabindranath   Tagore", "rabindranath tagore")])
def test_normalise_ignores_small_differences(reply, expected):
    assert run_eval.normalise(reply) == expected


def test_exact_match_accepts_any_listed_answer_and_nothing_else():
    capital = make_item(answer=["Bengaluru", "Bangalore"])
    assert run_eval.score_exact(capital, "Bangalore") == (True, "matches")
    passed, reason = run_eval.score_exact(capital, "The capital is Bengaluru.")
    assert passed is False and reason == "expected Bengaluru or Bangalore"   # strict on purpose


@pytest.mark.parametrize("reply, passed, reason_part", [
    ('{"name": "Priya", "city": "Indore"}', True, "expected values"),
    ('{"name": "priya", "city": "Indore", "shop": "bakery"}', True, "expected values"),   # case and extra keys are fine
    ('```json\n{"name": "Priya", "city": "Indore"}\n```', False, "code fences"),
    ('Here you go: {"name": "Priya"}', False, "not valid JSON"),
    ('["Priya", "Indore"]', False, "not an object"),
    ('{"name": "Priya"}', False, 'missing key "city"'),
    ('{"name": "Priya", "city": "Bhopal"}', False, '"city" is "Bhopal"'),
])
def test_json_check(reply, passed, reason_part):
    item = make_item(check="json", answer=None, expect={"name": "Priya", "city": "Indore"})
    got_passed, reason = run_eval.score_json(item, reply)
    assert got_passed is passed and reason_part in reason


def test_json_values_keep_their_type():
    order = make_item(check="json", answer=None, expect={"quantity": 3, "unit": ["kg", "kilos"], "is_question": True})
    assert run_eval.score_json(order, '{"quantity": 3.0, "unit": "Kilos", "is_question": true}')[0] is True
    assert run_eval.score_json(order, '{"quantity": "3", "unit": "kg", "is_question": true}')[0] is False
    assert run_eval.score_json(order, '{"quantity": 3, "unit": "kg", "is_question": "true"}')[0] is False


def test_open_answers_wait_for_the_judge():
    item = make_item(check="rubric", answer=None, rubric="PASS if polite.")
    assert run_eval.score(item, "anything") == (None, "waiting for the judge (step 6)")


# --- step 4: timing and tokens -------------------------------------------------

class ChunkModel:
    """A stand-in for a streaming model: it hands out the chunks we give it."""

    def __init__(self, chunks):
        self.chunks = chunks

    def stream(self, messages):
        yield from self.chunks


def test_ask_times_the_first_token_and_adds_up_the_token_counts():
    chunks = [  # Gemini style: content blocks, and each chunk carries its share of the tokens
        AIMessageChunk(content=[{"type": "text", "text": "33", "index": 0}],
                       usage_metadata={"input_tokens": 41, "output_tokens": 20, "total_tokens": 61,
                                       "output_token_details": {"reasoning": 18}}),
        AIMessageChunk(content=[{"type": "text", "text": "30", "index": 0}],
                       usage_metadata={"input_tokens": 0, "output_tokens": 2, "total_tokens": 2}),
    ]
    times = iter([10.0, 10.25, 10.9])        # start, first text, end (in seconds)
    row = run_eval.ask(ChunkModel(chunks), "question", clock=lambda: next(times))
    assert row == {"answer": "3330", "latency_ms": 900, "ttft_ms": 250,
                   "input_tokens": 41, "output_tokens": 22, "thinking_tokens": 18}


def test_ask_with_a_model_that_reports_no_tokens():
    row = run_eval.ask(fake_llm("4"), "What is 2 + 2?")
    assert row["answer"] == "4" and row["input_tokens"] is None and row["thinking_tokens"] is None


# --- steps 3 to 5: the whole run ---------------------------------------------------

def test_run_eval_marks_every_question_with_a_fake_model():
    items = load_eval_set(SAMPLE_SET, first=5)
    llm = fake_llm("3330", gemini_style_reply("10:05"), "239.2", "The answer is 18.", "Bengaluru")
    result = run_eval.run_eval(items, llm, "gemini", "fake-model", out=lambda *a: None)
    assert [row["passed"] for row in result["rows"]] == [True, True, True, False, True]
    assert result["meta"]["model"] == "fake-model" and result["meta"]["items"] == 5
    # every question went out with the same test instructions
    assert all(sent[0].content == run_eval.SYSTEM for sent in llm.received)
    tags = run_eval.by_tag(result["rows"])
    assert tags["arithmetic"] == {"passed": 3, "scored": 4, "waiting": 0}


class BrokenModel:
    def __init__(self, message):
        self.message, self.calls = message, 0

    def stream(self, messages):
        self.calls += 1
        raise RuntimeError(self.message)


def test_the_run_stops_after_three_errors_in_a_row():
    model = BrokenModel("connection refused")
    result = run_eval.run_eval(load_eval_set(SAMPLE_SET), model, "ollama", "qwen3.5:4b", out=lambda *a: None)
    assert model.calls == 3 and len(result["rows"]) == 3
    assert all(row["error"] == "connection refused" for row in result["rows"])


def test_the_free_tier_limit_stops_the_run_at_once():
    model = BrokenModel("429 RESOURCE_EXHAUSTED: quota exceeded")
    lines = []
    result = run_eval.run_eval(load_eval_set(SAMPLE_SET), model, "gemini", "m", out=lines.append)
    assert len(result["rows"]) == 1
    assert any("free-tier limit" in line for line in lines)


def test_a_run_where_every_call_fails_saves_nothing(monkeypatch, tmp_path, capsys):
    no_keys(monkeypatch)
    code = run_eval.main(["--first", "5", "--set", str(SAMPLE_SET)], llm=BrokenModel("connection refused"),
                         results_dir=tmp_path)
    assert code == 1 and list(tmp_path.iterdir()) == []      # an earlier good file would be left alone
    assert "Nothing saved: every call failed" in capsys.readouterr().out


def test_result_file_names_are_safe():
    assert run_eval.file_name_for("gemini-3.8-flash") == "gemini-3.8-flash.json"
    assert run_eval.file_name_for("qwen3.5:4b") == "qwen3.5-4b.json"


def test_main_without_a_key_prints_the_friendly_help(monkeypatch, capsys):
    no_keys(monkeypatch)
    assert run_eval.main(["--first", "1"]) == 1
    assert "Get a FREE Gemini key" in capsys.readouterr().out


def test_main_with_a_fake_model_saves_a_result_file(monkeypatch, tmp_path, capsys):
    no_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash-lite")
    code = run_eval.main(["--first", "2", "--set", str(SAMPLE_SET), "--model", "gemini-3.8-flash"],
                         llm=fake_llm("3330", "10:05"), results_dir=tmp_path)
    assert code == 0
    saved = json.loads((tmp_path / "gemini-3.8-flash.json").read_text(encoding="utf-8"))
    assert saved["meta"]["model"] == "gemini-3.8-flash"        # --model works like LLM_MODEL in .env
    assert [row["passed"] for row in saved["rows"]] == [True, True]
    assert "Saved" in capsys.readouterr().out


# --- steps 8 and 9: the scorecard (practice data) -----------------------------------

def practice_setup(must_overrides=None):
    results = scorecard.load_all_results(EPISODE / "practice" / "results")
    cards, problems = scorecard.load_cards(EPISODE / "practice" / "model_cards.json")
    assert problems == []
    prices = scorecard.load_toml(EPISODE / "practice" / "prices.toml")["models"]
    weights_file = scorecard.load_toml(EPISODE / "weights.toml")
    must = {**weights_file["must_have"], **(must_overrides or {})}
    return results, cards, prices, weights_file, must


def test_weights_become_percentages():
    weights_file = scorecard.load_toml(EPISODE / "weights.toml")
    assert scorecard.profile_weights(weights_file, "quality_first") == pytest.approx(
        {"quality": 0.6, "speed": 0.2, "cost": 0.2})
    with pytest.raises(ValueError, match="no profile named"):
        scorecard.profile_weights(weights_file, "cheapest")


def test_cost_per_1000_questions():
    price = {"input_per_million": 0.30, "output_per_million": 2.50}
    # (100 x 0.30 + 10 x 2.50) dollars per million questions = 0.055 dollars per 1,000 questions
    assert scorecard.cost_per_1000(100, 10, price) == pytest.approx(0.055)
    assert scorecard.cost_per_1000(100, 10, None) is None


def test_criterion_scores_use_your_limits():
    must = {"min_quality": 0.5, "max_median_ms": 10000, "max_cost_per_1000": 5.0}
    perfect = {"quality": 1.0, "median_ms": 0, "cost_per_1000": 0.0}
    just_ok = {"quality": 0.5, "median_ms": 10000, "cost_per_1000": 5.0}
    assert scorecard.criterion_scores(perfect, must) == {"quality": 1.0, "speed": 1.0, "cost": 1.0}
    assert scorecard.criterion_scores(just_ok, must) == {"quality": 0.0, "speed": 0.0, "cost": 0.0}


def test_practice_quality_first_puts_model_b_on_top():
    results, cards, prices, weights_file, must = practice_setup()
    kept, dropped = scorecard.build_candidates(results, cards, prices, must)
    assert dropped == []
    ranking = scorecard.rank(kept, scorecard.profile_weights(weights_file, "quality_first"))
    assert [model for model, _ in ranking] == ["practice-model-b", "practice-model-a"]
    assert [round(total, 2) for _, total in ranking] == [0.78, 0.63]


def test_cost_first_flips_the_ranking_and_says_why():
    results, cards, prices, weights_file, must = practice_setup()
    kept, _ = scorecard.build_candidates(results, cards, prices, must)
    quality, cost = (scorecard.profile_weights(weights_file, p) for p in ("quality_first", "cost_first"))
    assert scorecard.rank(kept, cost)[0][0] == "practice-model-a"
    sentence = scorecard.explain_change(kept, "quality_first", quality, "cost_first", cost)
    assert sentence.startswith("The ranking flips.")
    assert "cost counts for 60% instead of 20%" in sentence
    assert "pass rate 90% against 70%" in sentence


def test_when_the_leader_stays_the_sentence_says_it_holds():
    results, cards, prices, weights_file, must = practice_setup()
    kept, _ = scorecard.build_candidates(results, cards, prices, must)
    speed, cost = (scorecard.profile_weights(weights_file, p) for p in ("speed_first", "cost_first"))
    sentence = scorecard.explain_change(kept, "speed_first", speed, "cost_first", cost)
    assert sentence.startswith("The ranking holds: practice-model-a leads with both profiles.")
    assert "practice-model-b is better on quality" in sentence


def test_must_haves_drop_a_model_before_scoring():
    results, cards, prices, _, must = practice_setup({"private_data": True})
    kept, dropped = scorecard.build_candidates(results, cards, prices, must)
    assert [c["model"] for c in kept] == ["practice-model-a"]
    assert "may use your prompts" in dropped[0]["problems"][0]

    results, cards, prices, _, must = practice_setup({"min_quality": 0.8})
    kept, dropped = scorecard.build_candidates(results, cards, prices, must)
    assert [c["model"] for c in dropped] == ["practice-model-a"]
    assert "below your minimum of 80%" in dropped[0]["problems"][0]


def test_a_model_without_a_card_or_price_is_dropped():
    results, cards, prices, _, must = practice_setup()
    del cards["practice-model-b"]
    del prices["practice-model-a"]
    kept, dropped = scorecard.build_candidates(results, cards, prices, must)
    assert kept == []
    reasons = {c["model"]: " ".join(c["problems"]) for c in dropped}
    assert "no card" in reasons["practice-model-b"] and "no price" in reasons["practice-model-a"]


def test_notes_warn_about_unfair_or_stale_comparisons():
    results, cards, _, _, _ = practice_setup()
    results[0]["rows"] = results[0]["rows"][:5]                       # a --first 5 run
    results[1]["rows"][-1]["passed"] = None                          # one open answer not judged
    results[1]["meta"]["judged_by"] = "practice:practice-model-b"    # the model judged itself
    prices = {"practice-model-b": {"input_per_million": 1, "output_per_million": 1, "source": "x",
                                   "checked": date(2026, 10, 23), "valid_until": date(2026, 12, 31)}}
    notes = " | ".join(scorecard.notes_for(results, cards, prices, today=date(2027, 1, 5)))
    assert "different numbers of questions (5, 20)" in notes
    assert "1 open answers are waiting for the judge" in notes
    assert "judged its own open answers" in notes
    assert "the price expired on 2026-12-31" in notes


def test_the_markdown_report_has_the_ranking_and_licence_columns():
    results, cards, prices, weights_file, must = practice_setup()
    kept, dropped = scorecard.build_candidates(results, cards, prices, must)
    weights = scorecard.profile_weights(weights_file, "quality_first")
    md = scorecard.markdown_report(kept, dropped, scorecard.rank(kept, weights), "quality_first", must, [],
                                   results, date(2026, 10, 23), practice=True)
    assert "| 1 | practice-model-b | 18/20 (90%) |" in md
    assert "| Licence | Data use (free tier) | Score |" in md
    assert "| json_format | 4/4 | 2/4 |" in md
    assert "Re-test by 2027-01-21" in md


def test_the_html_report_escapes_every_answer():
    result = {"meta": {"model": "m"}, "rows": [{"id": "x-1", "question": "Say <b>hi</b>", "passed": True,
                                                "answer": "<script>alert('boo')</script>"}]}
    page = scorecard.html_answers([result], "Answers")
    assert "&lt;script&gt;alert(&#x27;boo&#x27;)&lt;/script&gt;" in page
    assert "<script" not in page and "<b>hi" not in page


def test_scorecard_main_writes_both_reports(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(scorecard, "REPORTS_DIR", tmp_path)
    assert scorecard.main(["--practice"], today=date(2026, 10, 23)) == 0
    assert (tmp_path / "practice_scorecard.md").is_file() and (tmp_path / "practice_answers.html").is_file()
    assert "PRACTICE DATA" in capsys.readouterr().out


def test_scorecard_without_results_says_what_to_do(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(scorecard, "RESULTS_DIR", tmp_path)
    assert scorecard.main([]) == 1
    assert "python run_eval.py" in capsys.readouterr().out


def test_the_real_model_cards_and_prices_are_complete_and_dated():
    cards, problems = scorecard.load_cards(EPISODE / "model_cards.json")
    prices = scorecard.load_toml(EPISODE / "prices.toml")["models"]
    assert problems == []
    for model_id, card in cards.items():
        assert card["source"].startswith("https://")
        date.fromisoformat(card["checked"])
        assert model_id in prices, f"add a price for {model_id} to prices.toml"
        assert prices[model_id]["source"].startswith("https://")
