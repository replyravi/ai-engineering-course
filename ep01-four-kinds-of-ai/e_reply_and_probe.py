"""
e_reply_and_probe.py - GenAI GENERATES new text, and it can also make things up.

Part 1 (step 12): draft short, polite replies to three complaints. Printed only, never sent.
Part 2 (step 13): the hallucination probe. We ask about opening hours twice:
                  first with no facts, then with one fact pasted into the prompt
                  (a tiny taste of RAG, which we build properly in Episode 7).

Run:  python e_reply_and_probe.py      (needs your free Gemini key)
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import load_feedback
from llm import MissingAPIKeyError, get_llm

# Byte Bites is invented, so no AI model can know its hours. Only this line knows.
CANTEEN_FACT = "Byte Bites opens at 7:30 am on weekdays and 9 am on Saturdays. It is closed on Sundays."
QUESTION = "What time does the Byte Bites canteen open on Sunday?"

reply_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the friendly manager of Byte Bites, a college canteen. "
     "Reply to the student's complaint in at most two short sentences. "
     "Say sorry and say what you will check. Do not promise refunds or free food."),
    ("human", "Complaint: {text}"),
])

plain_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])

grounded_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using ONLY the fact below. "
     "If the fact does not answer the question, say \"I don't know\".\n\n"
     "Fact: {fact}"),
    ("human", "{question}"),
])


def draft_replies(complaints, llm) -> list:
    """Write one short reply draft per complaint. Printed only, never sent.

    Why this exists: approaches A-D only sort text. This shows generation:
    the model writes new sentences.

    What it does: reply_prompt | llm | StrOutputParser, once per complaint.
    The system prompt says sorry, say what you will check, and do not
    promise refunds or free food.

    Called by: main (below, the first 3 plain complaints) and tests/test_genai.py.
    """
    chain = reply_prompt | llm | StrOutputParser()
    return [chain.invoke({"text": text}) for text in complaints]


def ask(question: str, llm, fact: str | None = None) -> str:
    """Ask one question. Paste a fact into the prompt only when fact is set.

    Why this exists: Byte Bites is invented, so the model cannot know its
    hours. The first call has no fact and often invents an answer. The
    second call pastes CANTEEN_FACT. That is a tiny taste of RAG.

    What it does: fact is None -> plain_prompt. fact is a string ->
    grounded_prompt, which says to use only that fact.

    Called by: main (below, twice for the Sunday question) and tests/test_genai.py.
    """
    if fact is None:
        chain = plain_prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})
    chain = grounded_prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "fact": fact})


def main(llm=None) -> int:
    """Draft 3 replies, then ask the Sunday question twice.

    Why this exists: this is the script entry for `python e_reply_and_probe.py`.

    What it does: get_llm(), load_feedback(), keep 3 complaints that are not
    marked sarcasm or negation, draft_replies(), then ask() with no fact and
    ask() with CANTEEN_FACT. Prints whether the grounded answer says "closed".
    Returns 1 if the key is missing. Tests pass a fake llm.

    Called by: the bottom of this file, and tests/test_genai.py.
    """
    if llm is None:
        try:
            llm = get_llm()
        except (MissingAPIKeyError, ValueError) as error:
            print(f"\n{str(error).strip()}")   # friendly help (no key, or a typo in LLM_PROVIDER)
            return 1

    # Part 1 - generate replies
    df = load_feedback()
    complaints = df[(df["label"] == "complaint") & (df["note"] == "")]["text"].head(3)
    print("Part 1 - reply drafts (printed only, nothing is sent)\n")
    for text, reply in zip(complaints, draft_replies(complaints, llm)):
        print(f"Complaint: {text}")
        print(f"Draft:     {reply.strip()}\n")

    # Part 2 - hallucination probe
    print("Part 2 - hallucination probe")
    print(f"Question: {QUESTION}\n")
    print("[1] No facts given:")
    print(ask(QUESTION, llm).strip() + "\n")
    print("[2] With one fact pasted into the prompt:")
    grounded = ask(QUESTION, llm, fact=CANTEEN_FACT).strip()
    print(grounded + "\n")
    print("Answer [2] uses the pasted fact (says 'closed'):", "yes" if "closed" in grounded.lower() else "no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
