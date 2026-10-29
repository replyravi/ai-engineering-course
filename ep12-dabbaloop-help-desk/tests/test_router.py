"""
tests/test_router.py - both routers and the evaluation maths.

The LLM router gets SCRIPTED routes from the fake model (we test our prompt, the form and the
fallback, not the AI). The semantic router uses the real local embeddings model.
"""

import pytest
from desk_fixtures import real_embeddings, semantic  # noqa: F401
from pydantic import ValidationError

from eval_router import accuracy, confusion_matrix, format_confusion, load_questions, run_router
from router import FALLBACK_ROUTE, ROUTER_RULES, ROUTES, Route, build_llm_router, route_tag
from embed_router import ROUTE_EXAMPLES, SemanticRouter, build_semantic_router
from testing.fakes import fake_llm, structured_reply

# ---------------------------------------------------------------------------
# The Route form and the LLM router
# ---------------------------------------------------------------------------


def test_the_route_form_allows_exactly_three_labels():
    schema = Route.model_json_schema()
    assert schema["properties"]["destination"]["enum"] == ["policy", "orders", "chat"]
    assert set(schema["required"]) == {"destination", "reason"}
    with pytest.raises(ValidationError):
        Route(destination="maths", reason="not a label we have")


def test_llm_router_returns_the_route_and_its_reason():
    llm = fake_llm(structured_reply(Route(destination="orders", reason="asks for a count of thalis sold")))

    route = build_llm_router(llm).invoke({"question": "How many Jain thalis were sold last week?"})

    assert route == Route(destination="orders", reason="asks for a count of thalis sold")
    assert route_tag(route) == "route=orders  (asks for a count of thalis sold)"
    system, human = llm.received[0]
    assert system.content == ROUTER_RULES
    assert human.content == "How many Jain thalis were sold last week?"     # the last turn is the human


def test_the_router_prompt_describes_every_route_and_the_default():
    for route in ROUTES:
        assert f"- {route}:" in ROUTER_RULES
    assert "If you are not sure, pick chat." in ROUTER_RULES


def test_a_failing_router_falls_back_to_chat():
    llm = fake_llm()            # no scripted reply: the router call raises an error

    route = build_llm_router(llm).invoke({"question": "Anything"})

    assert route.destination == FALLBACK_ROUTE == "chat"
    assert route.reason == "fallback: the router call failed"


# ---------------------------------------------------------------------------
# The semantic router (real local embeddings)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("What is the refund window for a missed lunch?", "policy"),
    ("How many Jain thalis were sold last week?", "orders"),
    ("Tell me a joke about dosas", "chat"),
])
def test_semantic_router_routes_the_intro_questions(semantic, question, expected):
    assert semantic.route(question).destination == expected


def test_semantic_router_falls_back_when_nothing_is_close(real_embeddings):
    strict = SemanticRouter(real_embeddings, min_similarity=0.99)
    route = strict.route("How many Jain thalis were sold last week?")
    assert route.destination == "chat"
    assert route.reason.startswith("fallback: closest example only")


def test_semantic_router_plugs_in_like_the_llm_router(semantic):
    route = build_semantic_router(router=semantic).invoke({"question": "Hi there!"})
    assert route == Route(destination="chat", reason='1.00 like "Hi there!"')


def test_the_examples_are_not_the_test_questions():
    examples = {text.lower() for texts in ROUTE_EXAMPLES.values() for text in texts}
    questions = {item["question"].lower() for item in load_questions()}
    assert examples.isdisjoint(questions)


# ---------------------------------------------------------------------------
# Evaluation maths (Step 16)
# ---------------------------------------------------------------------------

def test_twenty_labelled_questions():
    questions = load_questions()
    assert len(questions) == 20
    assert {q["route"] for q in questions} == set(ROUTES)


def test_confusion_matrix_and_accuracy():
    labels = ["policy", "orders", "chat"]
    truth = ["policy", "policy", "orders", "chat"]
    picked = ["policy", "orders", "orders", "chat"]

    counts = confusion_matrix(truth, picked, labels)

    assert counts["policy"] == {"policy": 1, "orders": 1, "chat": 0}
    assert counts["orders"]["orders"] == 1 and counts["chat"]["chat"] == 1
    assert accuracy(truth, picked) == 0.75
    assert format_confusion(counts, labels).splitlines()[1].split() == ["policy", "1", "1", "0"]


def test_run_router_counts_one_model_call_per_question():
    questions = [{"question": "q1", "route": "policy"}, {"question": "q2", "route": "chat"}]
    llm = fake_llm(structured_reply(Route(destination="policy", reason="rule")),
                   structured_reply(Route(destination="orders", reason="looks like a number")))

    results, calls = run_router(build_llm_router(llm), questions)

    assert [r["route"].destination for r in results] == ["policy", "orders"]
    assert calls == 2


def test_semantic_router_scores_18_of_20(semantic):
    results, calls = run_router(build_semantic_router(router=semantic), load_questions())
    truth = [r["truth"] for r in results]
    picked = [r["route"].destination for r in results]
    assert accuracy(truth, picked) == 0.9
    assert calls == 0
