"""
rag_chain.py - the QUESTION pipeline: retrieve -> augment -> generate.

    python rag_chain.py --show-prompt "What is the refund window for a missed lunch?"   (Step 8, no key)
    python rag_chain.py --simple "What is the refund window for a missed lunch?"        (Step 9)
    python rag_chain.py "What is the refund window for a missed lunch?"                 (Steps 10-11)

Classic name: older tutorials did all of this with RetrievalQA.from_chain_type(chain_type="stuff").
RetrievalQA is deprecated and only lives on in the langchain-classic package. Here we build the
same idea with LCEL pipes, so every step is visible and the answer comes back WITH its sources.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re
from pprint import pprint

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

from ingest import IndexNotBuiltError, load_index, source_tag, source_where
from llm import MissingAPIKeyError, get_llm

TOP_K = 3                   # how many chunks go into the prompt
MAX_DISTANCE = 0.80         # best chunk further away than this = weak retrieval, refuse without asking the model
REFUSAL = "I don't know based on the handbook."

# ---------------------------------------------------------------------------
# Step 8 - Augment: the grounded prompt
# ---------------------------------------------------------------------------

SYSTEM_RULES = """You are the help-desk assistant of Dabbaloop, a meal-box delivery service in Pune.
Answer the customer's question using ONLY the handbook extracts below.

Rules:
1. End every sentence with the tag of the extract it came from, for example [refunds.md#3].
2. If the extracts do not contain the answer, reply with exactly this sentence and nothing else:
   I don't know based on the handbook.
3. Never use outside knowledge and never guess a number, a time or a price.
4. Keep the answer short: at most three sentences.

Handbook extracts:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    ("human", "{question}"),      # the last message is the human turn (Gemini needs this)
])


def format_docs(docs):
    """Paste the retrieved chunks into one block of text, each with its citation tag."""
    blocks = [f"[{source_tag(doc)}] ({source_where(doc)})\n{doc.page_content}" for doc in docs]
    return "\n\n".join(blocks)


def describe_sources(docs):
    """The sources that travel with the answer: tag, where in the file, and distance."""
    return [
        {"tag": source_tag(doc), "where": source_where(doc), "distance": doc.metadata.get("distance")}
        for doc in docs
    ]


def cited_tags(answer):
    """The [file#chunk] tags that the answer actually uses, e.g. ['refunds.md#3']."""
    return re.findall(r"\[([\w.\-]+#\d+)\]", answer)


def estimate_tokens(text):
    """A rough token count: about 4 characters per token for English text."""
    return round(len(text) / 4)


# ---------------------------------------------------------------------------
# Step 9 - The simple LCEL RAG chain (what RetrievalQA used to do)
# ---------------------------------------------------------------------------

def build_simple_chain(store=None, llm=None, k=TOP_K):
    """question -> retrieve k chunks -> fill the prompt -> model -> plain text answer."""
    if store is None:
        store = load_index()
    if llm is None:
        llm = get_llm()
    retriever = store.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Steps 10-11 - Answer + sources together, and refuse when retrieval is weak
# ---------------------------------------------------------------------------

def build_chain(store=None, llm=None, k=TOP_K, max_distance=MAX_DISTANCE):
    """question -> {"answer": str, "sources": [...]}

    1. retrieve k chunks WITH their distances
    2. weak retrieval (best distance > max_distance)? -> refuse, and skip the model call
    3. otherwise a parallel step builds the answer and the source list at the same time
    4. if the model itself says it doesn't know, we return no sources
    """
    if store is None:
        store = load_index()
    if llm is None:
        llm = get_llm()

    def retrieve(question):
        docs = []
        for doc, distance in store.similarity_search_with_score(question, k=k):
            doc.metadata["distance"] = round(float(distance), 3)
            docs.append(doc)
        return docs

    def retrieval_is_weak(inputs):
        docs = inputs["docs"]
        return not docs or docs[0].metadata["distance"] > max_distance

    def refuse(_inputs):
        return {"answer": REFUSAL, "sources": []}

    answer_and_sources = RunnableParallel(
        answer=(
            {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
            | PROMPT
            | llm
            | StrOutputParser()
        ),
        sources=lambda x: describe_sources(x["docs"]),
    )

    def drop_sources_if_refused(result):
        if result["answer"].strip().startswith("I don't know"):
            return {"answer": result["answer"].strip(), "sources": []}
        return result

    return (
        {"docs": RunnableLambda(retrieve), "question": RunnablePassthrough()}
        | RunnableBranch((retrieval_is_weak, RunnableLambda(refuse)), answer_and_sources)
        | RunnableLambda(drop_sources_if_refused)
    )


# ---------------------------------------------------------------------------
# Step 8 helper - show the filled prompt without calling any model
# ---------------------------------------------------------------------------

def show_prompt(question, store=None, k=TOP_K):
    """Print the exact messages the model would receive, and their rough size."""
    if store is None:
        store = load_index()
    docs = store.similarity_search(question, k=k)
    messages = PROMPT.format_messages(context=format_docs(docs), question=question)
    for message in messages:
        print(f"----- {message.type} message -----")
        print(message.content)
    prompt_text = "".join(message.content for message in messages)
    prompt_tokens = estimate_tokens(prompt_text)
    reply_tokens = 100  # we ask for at most three short sentences
    print("-----")
    print(f"Prompt: {len(prompt_text):,} characters, about {prompt_tokens} tokens")
    print(f"Budget: about {prompt_tokens} prompt tokens + up to ~{reply_tokens} reply tokens "
          f"= about {prompt_tokens + reply_tokens} tokens of the model's context window")


def main():
    parser = argparse.ArgumentParser(description="Ask the Dabbaloop handbook one question.")
    parser.add_argument("question", nargs="?", default="What is the refund window for a missed lunch?")
    parser.add_argument("--show-prompt", action="store_true", help="only print the filled prompt (no model call)")
    parser.add_argument("--simple", action="store_true", help="use the simple chain (answer text only)")
    args = parser.parse_args()

    try:
        if args.show_prompt:
            show_prompt(args.question)
        elif args.simple:
            print(build_simple_chain().invoke(args.question))
        else:
            pprint(build_chain().invoke(args.question), sort_dicts=False, width=100)
    except (MissingAPIKeyError, IndexNotBuiltError) as error:
        print(str(error).strip())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
