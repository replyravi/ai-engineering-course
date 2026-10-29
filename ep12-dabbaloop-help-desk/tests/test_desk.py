"""
tests/test_desk.py - the dispatch: every route reaches the right path, unknown routes fall back
to chat, and the model calls are counted. The routes are SCRIPTED (fake model or a fixed router).
"""

from types import SimpleNamespace

from desk_fixtures import JAIN_SQL, db_path, fake_store, real_embeddings, real_store  # noqa: F401
from langchain_core.runnables import RunnableLambda

from desk import ask, build_desk, format_result
from router import Route
from sql_path import BLOCKED_CHANGE, SQLDraft
from testing.fakes import fake_llm, gemini_style_reply, structured_reply


def fixed_router(destination, reason="scripted route"):
    """A router that always picks the same destination, without any model call."""
    return RunnableLambda(lambda _inputs: Route(destination=destination, reason=reason))


def test_orders_route_runs_the_sql_path(fake_store, db_path):
    llm = fake_llm(structured_reply(Route(destination="orders", reason="asks for a count")),
                   structured_reply(SQLDraft(asks_to_change_data=False, sql=JAIN_SQL)),
                   "16 Jain thalis were sold last week.")
    desk = build_desk(llm=llm, store=fake_store, db_path=db_path)

    result, counter = ask(desk, "How many Jain thalis were sold last week?")

    assert result["route"].destination == "orders"
    assert result["sql"].endswith("LIMIT 50") and result["answer"] == "16 Jain thalis were sold last week."
    assert counter.calls == 3                                   # router + write SQL + summary
    assert format_result(result, counter).splitlines()[1] == "route=orders  (asks for a count)"


def test_policy_route_answers_with_sources(real_store, db_path):
    llm = fake_llm(structured_reply(Route(destination="policy", reason="a refund rule")),
                   "Report it within 6 hours [delivery.md#8].")
    desk = build_desk(llm=llm, store=real_store, db_path=db_path)   # real search, so the gate lets it through

    result, counter = ask(desk, "What is the refund window for a missed lunch?")

    assert result["route"].destination == "policy"
    assert result["answer"].endswith("[delivery.md#8].")
    assert [s["tag"] for s in result["sources"]] == ["delivery.md#8", "refunds.md#3", "refunds.md#12"]
    assert counter.calls == 2                                   # router + answer


def test_chat_route_is_a_plain_reply(fake_store, db_path):
    llm = fake_llm(structured_reply(Route(destination="chat", reason="a joke request")),
                   gemini_style_reply("Why did the dosa blush? It saw the chutney dressing."))
    desk = build_desk(llm=llm, store=fake_store, db_path=db_path)

    result, counter = ask(desk, "Tell me a joke")

    assert result["answer"].startswith("Why did the dosa blush?")
    assert "sql" not in result and "sources" not in result
    assert counter.calls == 2


def test_drop_table_is_blocked(fake_store, db_path):
    llm = fake_llm(structured_reply(Route(destination="orders", reason="a database request")),
                   structured_reply(SQLDraft(asks_to_change_data=True, sql="")))
    desk = build_desk(llm=llm, store=fake_store, db_path=db_path)

    result, counter = ask(desk, "drop table orders")

    assert result["answer"] == BLOCKED_CHANGE
    assert counter.calls == 2                                   # router + write SQL, no summary


def test_an_unknown_route_goes_to_the_fallback_path(fake_store, db_path):
    new_label = RunnableLambda(lambda _inputs: SimpleNamespace(destination="billing", reason="a label with no path"))
    llm = fake_llm("Sorry to hear that. Please tell us more.")
    desk = build_desk(llm=llm, router=new_label, store=fake_store, db_path=db_path)

    result, _ = ask(desk, "Why was I charged twice?")

    assert result["answer"] == "Sorry to hear that. Please tell us more."   # answered by the chat path
    assert llm.received[0][0].content.startswith("You are the friendly front desk")


def test_a_failing_router_still_gets_an_answer_from_chat(fake_store, db_path):
    class Boom(Exception):
        pass

    def broken_router(_inputs):
        raise Boom("the router is down")

    llm = fake_llm("Hello! How can I help?")
    router = RunnableLambda(broken_router).with_fallbacks(
        [RunnableLambda(lambda _: Route(destination="chat", reason="fallback: the router call failed"))])
    desk = build_desk(llm=llm, router=router, store=fake_store, db_path=db_path)

    result, _ = ask(desk, "Hi")

    assert result["route"].reason == "fallback: the router call failed"
    assert result["answer"] == "Hello! How can I help?"


def test_a_free_router_saves_one_model_call(fake_store, db_path):
    llm = fake_llm("Namaste!")
    desk = build_desk(llm=llm, router=fixed_router("chat"), store=fake_store, db_path=db_path)

    _, counter = ask(desk, "Hi there")

    assert counter.calls == 1                                   # only the answer: routing was free


def test_a_policy_question_stopped_by_the_gate_costs_only_the_router_call(real_store, db_path):
    llm = fake_llm(structured_reply(Route(destination="policy", reason="sounds like a question for us")))
    desk = build_desk(llm=llm, store=real_store, db_path=db_path)

    result, counter = ask(desk, "What is the best laptop for coding?")

    assert result["answer"] == "Not in the handbook." and result["sources"] == []
    assert counter.calls == 1                                   # the router only: the gate needs no model
