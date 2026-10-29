"""
chunklab/ask.py - Step 15: answer with Gemini, citing the section breadcrumb of every fact.

    python -m chunklab ask "What is the refund window for a missed lunch?"
    python -m chunklab ask --show-prompt "What is the refund window for a missed lunch?"   (no key needed)

It searches the 'headers' collection (Step 8), pastes the 3 closest chunks into a grounded prompt,
each one labelled with a tag like [refunds.md > Missed or late meal], and asks the model to cite them.
The tag comes straight from the chunk's metadata, so it always points at a real section.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from chunklab.index import CollectionsMissingError, load_store
from chunklab.strategies import breadcrumb, citation
from llm import MissingAPIKeyError, get_llm

TOP_K = 3
STRATEGY = "headers"            # the collection whose chunks carry a breadcrumb
REFUSAL = "I don't know based on the handbook."

RULES = """You answer questions from customers of Dabbaloop, a meal-box delivery service in Pune.
Use ONLY the handbook extracts below.

Rules:
1. End every sentence with the tag of the extract it came from, copied exactly, for example [refunds.md > Missed or late meal].
2. If the extracts do not answer the question, reply with exactly this sentence and nothing else:
   I don't know based on the handbook.
3. Do not guess any number, time or price.
4. Use at most three sentences.

Handbook extracts:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", RULES),
    ("human", "{question}"),          # the last message is the human turn (Gemini needs this)
])


def format_context(docs):
    """Each extract starts with its tag and its full breadcrumb, then the chunk text."""
    return "\n\n".join(f"[{citation(doc)}] ({breadcrumb(doc)})\n{doc.page_content}" for doc in docs)


def cited_tags(answer):
    """The [file > section] tags that the answer actually uses."""
    return re.findall(r"\[([^\[\]]+? > [^\[\]]+?)\]", answer)


def build_chain(store=None, llm=None, k=TOP_K):
    """question -> {"answer": text, "sources": [{"tag", "breadcrumb", "distance"}, ...]}"""
    if llm is None:
        llm = get_llm()                 # first, so a missing key stops us before anything else loads
    if store is None:
        store = load_store(STRATEGY)

    def retrieve(question):
        docs = []
        for doc, distance in store.similarity_search_with_score(question, k=k):
            doc.metadata["distance"] = round(float(distance), 3)
            docs.append(doc)
        return docs

    def sources(inputs):
        return [{"tag": citation(doc), "breadcrumb": breadcrumb(doc), "distance": doc.metadata["distance"]}
                for doc in inputs["docs"]]

    answer = (
        {"context": lambda inputs: format_context(inputs["docs"]), "question": lambda inputs: inputs["question"]}
        | PROMPT
        | llm
        | StrOutputParser()          # plain text, also for Gemini's list-of-blocks replies
    )
    return {"docs": RunnableLambda(retrieve), "question": RunnablePassthrough()} | RunnableParallel(
        answer=answer, sources=sources,
    )


def format_reply(result):
    """The answer, then every source with 'cited' next to the ones the answer used."""
    used = set(cited_tags(result["answer"]))
    lines = [f"Answer: {result['answer'].strip()}", "Sources:"]
    for source in result["sources"]:
        mark = "cited" if source["tag"] in used else "     "
        lines.append(f"  {mark}  [{source['tag']}]  distance {source['distance']:.3f}  ({source['breadcrumb']})")
    return "\n".join(lines)


def show_prompt(question, store=None, k=TOP_K):
    """Print the exact messages the model would get. No model, no key."""
    if store is None:
        store = load_store(STRATEGY)
    docs = store.similarity_search(question, k=k)
    for message in PROMPT.format_messages(context=format_context(docs), question=question):
        print(f"----- {message.type} message -----")
        print(message.content)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chunklab ask", description="Answer with citations.")
    parser.add_argument("question", nargs="?", default="What is the refund window for a missed lunch?")
    parser.add_argument("--show-prompt", action="store_true", help="only print the filled prompt (no model call)")
    args = parser.parse_args(argv)

    try:
        if args.show_prompt:
            show_prompt(args.question)
            return 0
        chain = build_chain()
        print(f"Question: {args.question}")
        print(format_reply(chain.invoke(args.question)))
    except (MissingAPIKeyError, CollectionsMissingError) as error:
        print(str(error).strip())
        return 1
    except Exception as error:          # for example no internet, or the free-tier limit
        print(f"The model call failed: {error}")
        print("(429 RESOURCE_EXHAUSTED = free-tier limit reached: wait a minute and try again)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
