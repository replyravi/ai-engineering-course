"""
tests/test_classic.py - the classic classes still run (with a scripted fake model) and warn.
We test them only so the comparison in classic_demo.py stays true.
"""

from desk_fixtures import db_path  # noqa: F401

from classic_demo import SCRIPTED, classic_router, classic_sql
from testing.fakes import fake_llm


def test_classic_router_warns_and_hides_its_route():
    llm = fake_llm(*SCRIPTED[:2])

    chain, caught = classic_router(llm)
    result = chain.invoke({"input": "How many Jain thalis were sold last week?"})

    text = "\n".join(caught)
    assert "`MultiPromptChain` was deprecated in LangChain 0.2.12" in text
    assert "`LLMRouterChain` was deprecated in LangChain 0.2.12" in text
    assert set(result) == {"input", "text"}                   # no route in the result


def test_classic_sql_chain_returns_text_and_leaks_sample_rows(db_path):
    llm = fake_llm(SCRIPTED[2])

    sql_text, prompt_text, _ = classic_sql(llm, db_path)

    assert sql_text == SCRIPTED[2]                              # just a string: nothing ran
    assert prompt_text.count("@example.com") == 3              # 3 sample customers, with emails
