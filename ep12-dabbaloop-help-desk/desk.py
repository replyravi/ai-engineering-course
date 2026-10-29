"""
desk.py - Step 13: the Dabbaloop help desk. One question in, the right path out, route printed every time.

    python desk.py "How many Jain thalis were sold last week?"
    python desk.py --router semantic "Is the kitchen nut-free?"      the free embeddings router (Step 12)
    python desk.py --demo                                            the three intro questions + "drop table orders"

    question --> router --> route=policy  -> rag.py       handbook answer with [file#chunk] sources
                        --> route=orders  -> sql_path.py  SQL, guard, read-only run, one sentence
                        --> route=chat    -> a short friendly reply
                        --> anything else -> chat (the explicit fallback)

Sequential chains (Episode 11) run every step for every input. A router runs ONE branch per input,
chosen at run time. The dispatch is a plain Python function inside RunnableLambda: it looks up the
route and RETURNS the chain to run, and LangChain then runs that chain on the same input.

Every run prints how many times the model was called: the router costs one extra call per question.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import os
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from ingest import IndexNotBuiltError, load_index
from llm import MissingAPIKeyError, get_llm
from rag import build_policy_chain
from router import FALLBACK_ROUTE, CallCounter, build_llm_router
from embed_router import SemanticRouter, build_semantic_router
from seed_db import DB_PATH
from sql_path import build_orders_chain

DEMO_QUESTIONS = [
    "What is the refund window for a missed lunch?",
    "How many Jain thalis were sold last week?",
    "Tell me a joke",
    "drop table orders",
]

CHAT_RULES = """You are the friendly front desk of Dabbaloop, a meal-box delivery service in Pune.
This is small talk: reply in one or two short, warm sentences.
In this mode you cannot look up rules or orders. If the person asks about them, invite them to ask
about a Dabbaloop rule or an order number instead."""

CHAT_PROMPT = ChatPromptTemplate.from_messages([("system", CHAT_RULES), ("human", "{question}")])


def build_chat_chain(llm):
    """{"question"} -> {"answer"}"""
    return CHAT_PROMPT | llm | StrOutputParser() | RunnableLambda(lambda text: {"answer": text.strip()})


def build_desk(llm=None, router="llm", store=None, db_path=DB_PATH, semantic=None):
    """{"question": str} -> {"question", "route", "answer", ...the extra keys of the path that answered}

    router: "llm" (structured output, one extra model call), "semantic" (local embeddings, free),
            or your own Runnable that turns {"question"} into a Route (handy in tests and the challenge).
    """
    if llm is None:
        llm = get_llm()
    if store is None:
        store = load_index()

    paths = {
        "policy": itemgetter("question") | build_policy_chain(store, llm),   # -> answer, sources
        "orders": build_orders_chain(llm, db_path),                          # -> answer, sql, rows, status
        "chat": build_chat_chain(llm),                                       # -> answer
    }
    if isinstance(router, Runnable):
        pick_route = router
    elif router == "llm":
        pick_route = build_llm_router(llm)
    else:
        pick_route = build_semantic_router(router=semantic or SemanticRouter(store.embeddings))

    def dispatch(state):
        """Return the chain for this route. A route we do not know goes to the fallback path."""
        return paths.get(state["route"].destination, paths[FALLBACK_ROUTE])

    def tidy(state):
        return {"question": state["question"], "route": state["route"], **state["result"]}

    return (
        RunnablePassthrough.assign(route=pick_route)                  # + route (destination, reason)
        | RunnablePassthrough.assign(result=RunnableLambda(dispatch))  # + the answer of ONE path
        | RunnableLambda(tidy)
    )


def ask(desk, question):
    """Run one question and count the model calls it needed."""
    counter = CallCounter()
    result = desk.invoke({"question": question}, config={"callbacks": [counter]})
    return result, counter


def format_result(result, counter):
    """The lines we print for one answer."""
    route = result["route"]
    lines = [f"Q: {result['question']}", f"route={route.destination}  ({route.reason})"]
    if route.destination == "orders":
        lines.append(f"SQL used: {result.get('sql') or '(none)'}")
        if result.get("rows"):                            # the real numbers, so you can check the sentence
            lines += ["   " + row for row in result["rows"].splitlines()]
    lines.append(f"A: {result['answer']}")
    for source in result.get("sources", []):
        lines.append(f"   source {source['tag']:<18} {source['where']}  (distance {source['distance']:.3f})")
    tokens = f", {counter.tokens} tokens" if counter.tokens else ""
    lines.append(f"model calls: {counter.calls}{tokens}")
    return "\n".join(lines)


def friendly_error(error):
    """A short explanation of a failed model call, never a long traceback."""
    text = f"{type(error).__name__}: {error}"
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "The free-tier limit was reached. Wait a minute and try again: https://aistudio.google.com/rate-limit"
    if os.getenv("LLM_PROVIDER", "gemini").strip().lower() == "ollama" and "Connect" in text:
        return "Could not reach Ollama. Start the Ollama app and run  ollama pull qwen3.5:4b"
    if "Connect" in type(error).__name__:
        return f"Could not reach the model's server ({text[:120]}). Check your internet connection."
    return f"The model call failed: {text[:300]}"


def main():
    parser = argparse.ArgumentParser(description="The Dabbaloop help desk.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--router", choices=["llm", "semantic"], default="llm")
    parser.add_argument("--demo", action="store_true", help="run the four questions of the intro")
    args = parser.parse_intermixed_args()

    if not DB_PATH.exists():
        print(f"{DB_PATH.name} not found. Create it first with:  python seed_db.py")
        raise SystemExit(1)
    try:
        desk = build_desk(router=args.router)
    except (MissingAPIKeyError, IndexNotBuiltError) as error:
        print(str(error).strip())
        raise SystemExit(1)

    questions = DEMO_QUESTIONS if args.demo or not args.question else [" ".join(args.question)]
    for number, question in enumerate(questions):
        if number:
            print()
        try:
            print(format_result(*ask(desk, question)))
        except Exception as error:   # a network problem or the free-tier limit: explain, keep going
            print(f"Q: {question}\n{friendly_error(error)}")


if __name__ == "__main__":
    main()
