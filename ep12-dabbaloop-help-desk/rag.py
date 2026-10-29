"""
rag.py - the POLICY path: answer from the Dabbaloop handbook, with sources (Steps 2-5).

    python rag.py index                                Step 2: reuse Episode 07's ingest (no key needed)
    python rag.py ids                                  Step 3: which chunks come back? (no key needed)
    python rag.py ask --show-prompt "question"         Step 4: the filled prompt only (no key needed)
    python rag.py ask "question"                       Step 4: answer + [file#chunk] tags
    python rag.py gate "question"                      Step 5: does the refusal gate stop it? (no key needed)

The search uses embeddings (384 numbers per chunk). The chat model never sees those numbers:
it reads the retrieved chunks as plain text, each with a tag like [delivery.md#8].

Classic name: older tutorials built this with RetrievalQA, or with the same LCEL pipe on top of
three short strings (where every question "retrieved" everything). Here the index has 54 chunks,
so we can prove that each question gets its own 3.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import inspect
from pprint import pprint

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

from ingest import (CHUNK_OVERLAP, CHUNK_SIZE, HANDBOOK_DIR, IndexNotBuiltError, build_index, load_handbook,
                    load_index, source_tag, source_where, split_documents)
from llm import MissingAPIKeyError, get_embeddings, get_llm

TOP_K = 3                     # chunks per question
MAX_DISTANCE = 0.80           # best chunk further than this: refuse WITHOUT asking the model (Episode 07's gate)
REFUSAL = "Not in the handbook."

QUESTION_1 = "What is the refund window for a missed lunch?"
QUESTION_2 = "Is the kitchen nut-free?"

# The "tiny library" of older demos: three short notes. Nothing to split, nothing to select.
TINY_LIBRARY = [
    "Dabbaloop delivers lunch in Pune between 12 pm and 1:45 pm.",
    "A missed meal is credited to your Dabbaloop wallet.",
    "Every steel meal box needs a refundable deposit.",
]

SYSTEM_RULES = """You answer questions for the Dabbaloop help desk using ONLY the handbook extracts below.
Dabbaloop is a meal-box delivery service in Pune.

Rules:
1. Finish every sentence with the tag of the extract you used, like [delivery.md#8].
2. If the extracts do not answer the question, reply with exactly: Not in the handbook.
3. Do not add facts, numbers or prices that are not in the extracts.
4. At most three short sentences.

Handbook extracts:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    ("human", "{question}"),          # the last message is the human turn (Gemini needs this)
])


# ---------------------------------------------------------------------------
# Retrieve and format
# ---------------------------------------------------------------------------

def retrieve(store, question, k=TOP_K):
    """The k closest chunks, closest first. Each chunk remembers its distance in its metadata."""
    docs = []
    for doc, distance in store.similarity_search_with_score(question, k=k):
        doc.metadata["distance"] = round(float(distance), 3)
        docs.append(doc)
    return docs


def format_docs(docs):
    """Turn the retrieved chunks into ONE block of plain text, each chunk under its tag."""
    return "\n\n".join(f"[{source_tag(doc)}] ({source_where(doc)})\n{doc.page_content}" for doc in docs)


def describe_sources(docs):
    """The sources that travel with the answer."""
    return [{"tag": source_tag(d), "where": source_where(d), "distance": d.metadata.get("distance")} for d in docs]


# ---------------------------------------------------------------------------
# The policy chain: retrieve -> gate -> prompt -> model -> answer + sources
# ---------------------------------------------------------------------------

def build_policy_chain(store=None, llm=None, k=TOP_K, max_distance=MAX_DISTANCE):
    """question (str) -> {"answer": str, "sources": [...]}"""
    if store is None:
        store = load_index()
    if llm is None:
        llm = get_llm()

    answer_and_sources = RunnableParallel(
        answer={"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | PROMPT | llm | StrOutputParser(),
        sources=lambda x: describe_sources(x["docs"]),
    )

    def too_far(state):
        return not state["docs"] or state["docs"][0].metadata["distance"] > max_distance

    def drop_sources_if_refused(result):
        answer = result["answer"].strip()
        return {"answer": answer, "sources": [] if answer.startswith(REFUSAL) else result["sources"]}

    return (
        {"docs": RunnableLambda(lambda q: retrieve(store, q, k)), "question": RunnablePassthrough()}
        | RunnableBranch(
            (too_far, RunnableLambda(lambda _: {"answer": REFUSAL, "sources": []})),   # the gate: no model call
            answer_and_sources | RunnableLambda(drop_sources_if_refused),
        )
    )


# ---------------------------------------------------------------------------
# Step helpers (no key needed)
# ---------------------------------------------------------------------------

def index_handbook(embeddings=None, index_dir=None):
    """Step 2: run Episode 07's indexing pipeline and report chunks and vector size."""
    docs = load_handbook()
    chunks = split_documents(docs)
    kwargs = {} if index_dir is None else {"index_dir": index_dir}
    store, report = build_index(chunks, embeddings or get_embeddings(), **kwargs)
    vector = store.get(limit=1, include=["embeddings"])["embeddings"][0]
    files = sorted({d.metadata["source"] for d in docs})
    return {"files": len(files), "documents": len(docs), "chunks": report["total"], "dim": len(vector),
            "added": report["added"], "removed": report["removed"]}


def tiny_library_ids(embeddings, questions):
    """The old demo: 3 notes in Chroma and a retriever with the DEFAULT k. Returns the notes each question got."""
    from langchain_chroma import Chroma

    store = Chroma(collection_name="tiny_library", embedding_function=embeddings)   # in memory only
    store.reset_collection()
    store.add_texts(TINY_LIBRARY, ids=[f"note{i}" for i in range(1, len(TINY_LIBRARY) + 1)])
    retriever = store.as_retriever()               # no search_kwargs: Chroma's default k is used
    found = {q: [doc.id for doc in retriever.invoke(q)] for q in questions}
    store.delete_collection()
    return found


def default_k():
    """The k that Chroma's similarity_search uses when you do not pass one."""
    from langchain_chroma import Chroma

    return inspect.signature(Chroma.similarity_search).parameters["k"].default


def show_ids(questions, store=None):
    """Step 3: print the retrieved ids for each question, first in the tiny library, then in the handbook."""
    store = store or load_index()
    embeddings = store.embeddings
    total = len(store.get(include=[])["ids"])
    labels = [f"Q{n}" for n in range(1, len(questions) + 1)]

    tiny = tiny_library_ids(embeddings, questions)
    print(f"Tiny library: {len(TINY_LIBRARY)} short notes, as_retriever() with no k (Chroma's default k = {default_k()})")
    for label, q in zip(labels, questions):
        print(f"  {label} {q:<48} -> {', '.join(tiny[q])}   ({len(tiny[q])} of {len(TINY_LIBRARY)})")
    print("  Every question got every note: nothing was selected.\n")

    print(f"Real handbook: {total} chunks, k = {TOP_K}")
    found = {}
    for label, q in zip(labels, questions):
        docs = retrieve(store, q)
        found[q] = [source_tag(d) for d in docs]
        tags = ", ".join(f"{source_tag(d)} ({d.metadata['distance']:.3f})" for d in docs)
        print(f"  {label} {q:<48} -> {tags}")
    shared = set(found[questions[0]]).intersection(*map(set, found.values()))
    print(f"  Shared ids: {', '.join(sorted(shared)) or 'none'}. "
          f"Each question got its own {TOP_K} of {total} chunks.")


def show_gate(question, store=None, max_distance=MAX_DISTANCE):
    """Step 5: the first safety net, without any model."""
    store = store or load_index()
    best = retrieve(store, question, k=1)[0]
    distance = best.metadata["distance"]
    print(f"Question: {question}")
    print(f"Best chunk: [{source_tag(best)}] at distance {distance:.3f} (the gate refuses above {max_distance:.2f})")
    if distance > max_distance:
        print(f"Refused WITHOUT calling the model: {REFUSAL}")
    else:
        print("Passes the gate: the model reads the chunks, and rule 2 of the prompt must refuse if they hold no answer.")


def show_prompt(question, store=None):
    """Step 4: the exact messages the model would receive."""
    store = store or load_index()
    for message in PROMPT.format_messages(context=format_docs(retrieve(store, question)), question=question):
        print(f"----- {message.type} message -----")
        print(message.content)


def main():
    parser = argparse.ArgumentParser(description="The policy path: the Dabbaloop handbook with sources.")
    parser.add_argument("command", choices=["index", "ids", "ask", "gate"])
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--show-prompt", action="store_true", help="ask: print the filled prompt only")
    args = parser.parse_intermixed_args()      # options may come before or after the question

    try:
        if args.command == "index":
            report = index_handbook()
            print(f"Loaded {report['files']} files as {report['documents']} documents "
                  f"(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})")
            print(f"{report['chunks']} chunks indexed, dim {report['dim']} "
                  f"(new this run: {report['added']}, removed: {report['removed']})")
            print(f"Handbook folder: {HANDBOOK_DIR.name}/   Index folder: index/")
        elif args.command == "ids":
            show_ids(args.questions or [QUESTION_1, QUESTION_2])
        elif args.command == "gate":
            show_gate(" ".join(args.questions) or "What is the best laptop for coding?")
        else:
            question = " ".join(args.questions) or QUESTION_1
            if args.show_prompt:
                show_prompt(question)
            else:
                pprint(build_policy_chain().invoke(question), sort_dicts=False, width=100)
    except (MissingAPIKeyError, IndexNotBuiltError) as error:
        print(str(error).strip())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
