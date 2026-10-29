"""
classic_demo.py - Step 14: run the CLASSIC classes once, so you recognise them in older tutorials.

    python classic_demo.py            with your real model (needs your key)
    python classic_demo.py --fake     with a SCRIPTED fake model: no key, and the replies are not AI text

Part 1  MultiPromptChain + LLMRouterChain (langchain_classic): the router answers in JSON text,
        a parser reads it, and the chosen route stays hidden inside the chain.
Part 2  create_sql_query_chain (langchain_classic): returns SQL TEXT that nobody runs. Its prompt
        also carries 3 sample rows of every table, so customer names and emails go to the model.

Then the map from each classic class to the 2026 code in this folder.
Do not build new projects on these classes: they are deprecated, or only kept for old code.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re
import warnings

from langchain_core.callbacks import BaseCallbackHandler

from llm import MissingAPIKeyError, get_llm
from seed_db import DB_PATH
from testing.fakes import fake_llm

QUESTION = "How many Jain thalis were sold last week?"

PROMPT_INFOS = [
    {"name": "policy", "description": "Dabbaloop rules, prices and delivery questions",
     "prompt_template": "You answer Dabbaloop policy questions in two sentences.\n\nQuestion: {input}"},
    {"name": "orders", "description": "numbers about Dabbaloop orders and sales",
     "prompt_template": "You answer questions about Dabbaloop order numbers.\n\nQuestion: {input}"},
    {"name": "chat", "description": "greetings and small talk",
     "prompt_template": "You are a friendly front desk. Reply briefly.\n\n{input}"},
]

SCRIPTED = [
    '```json\n{"destination": "orders", "next_inputs": "How many Jain thalis were sold last week?"}\n```',
    "[scripted reply, not AI text] the orders prompt would answer here",
    "SELECT SUM(quantity) FROM orders WHERE item_id = 2",
]

MAPPING = [
    ("MultiPromptChain, LLMRouterChain", "prompt | llm.with_structured_output(Route), dispatch in RunnableLambda",
     "router.py, desk.py"),
    ("(no classic class)", "SemanticRouter: nearest example by local embeddings, no model call",
     "embed_router.py"),
    ("create_sql_query_chain", "SQLDraft (structured output) -> sql_guard.py -> read-only run -> summary",
     "sql_path.py"),
    ("SQLDatabaseChain", "small SQL tools + create_agent (Episode 15), reusing sql_guard.safe_query",
     "langchain_experimental only"),
    ("RetrievalQA", "LCEL: retrieve | format_docs | prompt | llm, with sources and a refusal gate",
     "rag.py"),
]


class PromptCatcher(BaseCallbackHandler):
    """Keeps the text of every prompt sent to the chat model (works for real and fake models)."""

    def __init__(self):
        self.prompts = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.prompts.append("".join(str(m.content) for m in messages[0]))


def short_warning(w):
    """One readable line per warning, without the long file path."""
    return f"{w.category.__name__}: {str(w.message).split(' See ')[0]}"


def classic_router(llm):
    """Build MultiPromptChain the long way, as older docs did. Returns (chain, warnings)."""
    from langchain_core.prompts import PromptTemplate

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from langchain_classic.chains import LLMChain
        from langchain_classic.chains.router import MultiPromptChain
        from langchain_classic.chains.router.llm_router import LLMRouterChain, RouterOutputParser
        from langchain_classic.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

        destinations = {info["name"]: LLMChain(llm=llm, prompt=PromptTemplate.from_template(info["prompt_template"]))
                        for info in PROMPT_INFOS}
        menu = "\n".join(f"{info['name']}: {info['description']}" for info in PROMPT_INFOS)
        router_prompt = PromptTemplate(template=MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=menu),
                                       input_variables=["input"], output_parser=RouterOutputParser())
        router_chain = LLMRouterChain(llm_chain=LLMChain(llm=llm, prompt=router_prompt))
        chain = MultiPromptChain(router_chain=router_chain, destination_chains=destinations,
                                 default_chain=destinations["chat"])
    unique = list(dict.fromkeys(short_warning(w) for w in caught if issubclass(w.category, DeprecationWarning)))
    return chain, unique


def classic_sql(llm, db_path=DB_PATH):
    """create_sql_query_chain on our database (opened read-only). Returns (sql_text, prompt_text, warnings)."""
    from sqlalchemy import create_engine

    from sql_guard import open_read_only

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from langchain_community.utilities import SQLDatabase      # archived package, still works
        from langchain_classic.chains import create_sql_query_chain  # noqa: I001 (the classic helper)

        db = SQLDatabase(create_engine("sqlite://", creator=lambda: open_read_only(db_path)))
        catcher = PromptCatcher()
        sql_text = create_sql_query_chain(llm, db).invoke({"question": QUESTION}, config={"callbacks": [catcher]})
    prompt_text = catcher.prompts[-1] if catcher.prompts else ""
    unique = list(dict.fromkeys(short_warning(w) for w in caught if issubclass(w.category, DeprecationWarning)))
    return sql_text, prompt_text, unique


def main():
    parser = argparse.ArgumentParser(description="Run the classic router and SQL chain once.")
    parser.add_argument("--fake", action="store_true", help="use a scripted fake model (no key needed)")
    args = parser.parse_args()

    if args.fake:
        llm = fake_llm(*SCRIPTED)
        print("Model: a SCRIPTED fake (testing/fakes.py). The replies below were written by hand, not by an AI.\n")
    else:
        try:
            llm = get_llm()
        except MissingAPIKeyError as error:
            print(str(error).strip())
            print("\nNo key yet? See the warnings anyway with:  python classic_demo.py --fake")
            raise SystemExit(1)

    print("=== Part 1: MultiPromptChain (classic router) ===")
    chain, caught = classic_router(llm)
    for line in caught:
        print(f"  {line}")
    result = chain.invoke({"input": QUESTION})
    print(f"Result keys: {list(result)}")
    print(f"Result: {result}")
    print("Which route did it pick? The result does not say (only verbose=True prints it).\n")

    print("=== Part 2: create_sql_query_chain (classic text-to-SQL) ===")
    sql_text, prompt_text, caught = classic_sql(llm)
    for line in caught:
        print(f"  {line}")
    print(f"Returned: {sql_text!r}")
    print("It is only text: nothing ran, no rows, no answer, no guard.")
    if prompt_text:
        emails = sorted(set(re.findall(r"[\w.]+@example\.com", prompt_text)))
        print(f"Customer emails inside the prompt it sent: {len(emails)} ({', '.join(emails)})")
    print()

    print("=== Classic -> 2026 ===")
    for classic, modern, where in MAPPING:
        print(f"  {classic:<33} -> {modern}  [{where}]")


if __name__ == "__main__":
    main()
