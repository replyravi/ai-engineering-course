"""
rung3_context.py - Rung 3: give the model CONTEXT from the handbook (Steps 6, 7 and 8).

Retrieve first, then answer:
    1. split data/handbook.md into numbered paragraphs (#1, #2, ...)
    2. embed every paragraph and the question with the free local model (get_embeddings)
    3. rank the paragraphs by cosine similarity (the same code as Episode 04) and keep the top 3
    4. paste those 3 paragraphs into the prompt, and ask the model to answer ONLY from them

The model never searches anything itself. OUR code searches first and hands it the text.
(Episode 07 turns this into full RAG with a vector database.)

Run:  python rung3_context.py --no-context          (Step 6: no handbook, watch it guess)
      python rung3_context.py                       (Step 7: top 3 paragraphs + citations)
      python rung3_context.py "Is there a student discount?"              (Step 8: refusal)
      python rung3_context.py --no-rule "Is there a student discount?"    (Step 8: without the rule)
      python rung3_context.py --show-prompt         (the filled prompt only, no model call)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import re
import time

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import HANDBOOK_FILE, HOUSE_RULES, REFUSAL, friendly_error
from llm import MissingAPIKeyError, get_embeddings, get_llm

QUESTION = "What is the last time I can skip today's lunch?"
TOP_K = 3   # how many paragraphs we paste into the prompt


# ---------------------------------------------------------------------------
# 1. The handbook as numbered paragraphs
# ---------------------------------------------------------------------------

def load_paragraphs(path=HANDBOOK_FILE) -> list[dict]:
    """Every paragraph of the handbook with a tag like 'handbook.md#4'.

    A paragraph is a block of text between blank lines.
    Headings (lines starting with #) and <!-- comments --> are skipped.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)        # drop comments
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text)]
    paragraphs = [block for block in blocks if block and not block.startswith("#")]
    return [{"tag": f"{path.name}#{number}", "text": " ".join(paragraph.split())}
            for number, paragraph in enumerate(paragraphs, start=1)]


# ---------------------------------------------------------------------------
# 2-3. Rank paragraphs by meaning (cosine similarity, as in Episode 04)
# ---------------------------------------------------------------------------

def cosine_similarity(a, b) -> float:
    """How closely two vectors point the same way: 1.0 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank(question: str, paragraphs: list[dict], embeddings, top: int = TOP_K) -> list[tuple[float, dict]]:
    """The `top` paragraphs closest in meaning to the question, best first, as (score, paragraph)."""
    vectors = embeddings.embed_documents([p["text"] for p in paragraphs])
    question_vector = embeddings.embed_query(question)
    scored = [(cosine_similarity(question_vector, v), p) for v, p in zip(vectors, paragraphs)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top]


def show_hits(hits) -> None:
    print(f"Top {len(hits)} handbook paragraphs (cosine similarity, higher = closer in meaning):")
    for score, paragraph in hits:
        print(f"  {score:.3f}  [{paragraph['tag']}]  {paragraph['text'][:70]}...")


# ---------------------------------------------------------------------------
# 4. The grounded prompt
# ---------------------------------------------------------------------------

GROUNDED_RULES = """Answer the customer's question using ONLY the handbook paragraphs below.
1. End every sentence with the tag of the paragraph it came from, like [handbook.md#N] (N is the paragraph number).
2. Never use outside knowledge, and never guess a time, a price or a date."""

# Step 8: the "I don't know" rule.
REFUSAL_RULE = (f"3. If the paragraphs do not contain the answer, say \"{REFUSAL}\" "
                "and offer to connect the customer to a support agent in the in-app chat.")


def build_prompt(with_refusal_rule: bool = True) -> ChatPromptTemplate:
    """House rules + grounding rules (+ the refusal rule) + the retrieved paragraphs."""
    rules = GROUNDED_RULES + ("\n" + REFUSAL_RULE if with_refusal_rule else "")
    return ChatPromptTemplate.from_messages([
        ("system", HOUSE_RULES + "\n\n" + rules + "\n\nHandbook paragraphs:\n{context}"),
        ("human", "{question}"),      # the last message is the human turn (Gemini needs this)
    ])


# Step 6: the same question with NO handbook text at all.
NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HOUSE_RULES),
    ("human", "{question}"),
])


def format_context(hits) -> str:
    """Paste the retrieved paragraphs into one block of text, each with its tag."""
    return "\n\n".join(f"[{p['tag']}] {p['text']}" for _, p in hits)


def cited_tags(answer: str) -> list[str]:
    """The [handbook.md#N] tags that appear in an answer."""
    return re.findall(r"\[([\w.\-]+#\d+)\]", answer)


def answer_with_context(question: str, llm=None, embeddings=None, with_refusal_rule: bool = True,
                        paragraphs=None, hits=None):
    """Retrieve the top 3 paragraphs, then ask the model. Returns (answer, hits)."""
    if hits is None:
        if embeddings is None:
            embeddings = get_embeddings()
        if paragraphs is None:
            paragraphs = load_paragraphs()
        hits = rank(question, paragraphs, embeddings)
    if llm is None:
        llm = get_llm()
    chain = build_prompt(with_refusal_rule) | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(hits), "question": question})
    return answer, hits


def answer_without_context(question: str, llm=None) -> str:
    """Step 6: ask with the house rules only. No handbook text is sent."""
    if llm is None:
        llm = get_llm()
    return (NO_CONTEXT_PROMPT | llm | StrOutputParser()).invoke({"question": question})


def main(argv=None, llm=None, embeddings=None) -> int:
    parser = argparse.ArgumentParser(description="Rung 3: answer from the top 3 handbook paragraphs.")
    parser.add_argument("question", nargs="?", default=QUESTION)
    parser.add_argument("--no-context", action="store_true", help="Step 6: ask without the handbook")
    parser.add_argument("--no-rule", action="store_true", help="Step 8: leave out the 'I don't know' rule")
    parser.add_argument("--show-prompt", action="store_true", help="print the filled prompt only (no model call)")
    args = parser.parse_args(argv)
    print(f"Question: {args.question}\n")

    try:
        if args.no_context:
            answer = answer_without_context(args.question, llm)
            print(answer)
            print("\n(No handbook text was sent with this question. Any time, price or rule in the answer "
                  "above is a guess, not Dabbaloop's rule. The real rule is in data/handbook.md.)")
            return 0

        start = time.perf_counter()
        if embeddings is None:
            embeddings = get_embeddings()                    # free local model, no key
        loaded = time.perf_counter()
        paragraphs = load_paragraphs()
        hits = rank(args.question, paragraphs, embeddings)
        show_hits(hits)
        print(f"(on your computer: model loaded in {loaded - start:.1f} s, "
              f"{len(paragraphs)} paragraphs ranked in {time.perf_counter() - loaded:.2f} s)\n")

        if args.show_prompt:
            filled = build_prompt(not args.no_rule).invoke(
                {"context": format_context(hits), "question": args.question})
            for message in filled.to_messages():
                print(f"----- {message.type} message -----\n{message.text}")
            return 0

        answer, _ = answer_with_context(args.question, llm, with_refusal_rule=not args.no_rule, hits=hits)
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1

    print(f"Answer:\n{answer}\n")
    sent = {p["tag"] for _, p in hits}
    cited = cited_tags(answer)
    if cited:
        ok = all(tag in sent for tag in cited)
        print(f"Cited: {', '.join(cited)}  ({'all from the 3 paragraphs we sent' if ok else 'WARNING: not a paragraph we sent'})")
    else:
        print("Cited: nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
