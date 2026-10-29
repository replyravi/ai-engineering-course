"""
rung2_template.py - Rung 2: a prompt TEMPLATE piped into the model and a parser (Steps 4 and 5).

    chain = PROMPT | llm | StrOutputParser()

    PROMPT            fills the {slots} and makes the messages      (dict    -> messages)
    llm               the chat model from get_llm()                  (messages -> AI message)
    StrOutputParser   keeps only the reply text                      (AI message -> plain str)

The finished chain is a "runnable", so the SAME object can run three ways:
    chain.invoke(one_input)       wait for the whole answer
    chain.stream(one_input)       get the answer in pieces as they are written
    chain.batch([in1, in2, in3])  several inputs in one call

Run:  python rung2_template.py                    (Step 4: invoke)
      python rung2_template.py --show-prompt      (the filled prompt only, no model call, no key)
      python rung2_template.py --stream           (Step 5: stream)
      python rung2_template.py --batch            (Step 5: batch)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import HOUSE_RULES, friendly_error, tracing_status
from llm import MissingAPIKeyError, get_llm

# A template with two named slots: {topic} and {audience}.
PROMPT = ChatPromptTemplate.from_messages([
    ("system", HOUSE_RULES),
    ("human", "Explain {topic} to {audience} in two short sentences."),
])

AUDIENCE = "a first-time customer"
TOPICS = ["a meal-box subscription", "a delivery slot", "a refundable deposit"]


def build_chain(llm=None):
    """prompt -> model -> plain text. Pass llm=fake_llm(...) in tests."""
    if llm is None:
        llm = get_llm()
    return PROMPT | llm | StrOutputParser()


def show_prompt(topic: str, audience: str = AUDIENCE) -> None:
    """Fill the template and print the messages the model would receive (no model call)."""
    print(f"Template slots: {', '.join(sorted(PROMPT.input_variables))}\n")
    for message in PROMPT.invoke({"topic": topic, "audience": audience}).to_messages():
        print(f"----- {message.type} message -----\n{message.text}")


def run_invoke(chain, topic: str, audience: str = AUDIENCE) -> str:
    start = time.perf_counter()
    answer = chain.invoke({"topic": topic, "audience": audience})
    print(answer)
    print(f"\n(invoke: the whole answer arrived after {time.perf_counter() - start:.2f} s)")
    return answer


def run_stream(chain, topic: str, audience: str = AUDIENCE) -> str:
    start = time.perf_counter()
    first = None
    pieces = []
    for piece in chain.stream({"topic": topic, "audience": audience}):
        if first is None:
            first = time.perf_counter() - start
        print(piece, end="", flush=True)       # print each piece the moment it arrives
        pieces.append(piece)
    total = time.perf_counter() - start
    print(f"\n\n(stream: first piece after {first or 0:.2f} s, whole answer after {total:.2f} s, "
          f"{len(pieces)} pieces)")
    return "".join(pieces)


def run_batch(chain, topics: list[str], audience: str = AUDIENCE) -> list[str]:
    start = time.perf_counter()
    answers = chain.batch([{"topic": t, "audience": audience} for t in topics])
    for topic, answer in zip(topics, answers):
        print(f"[{topic}]\n{answer}\n")
    print(f"(batch: {len(answers)} answers after {time.perf_counter() - start:.2f} s)")
    return answers


def main(argv=None, llm=None) -> int:
    parser = argparse.ArgumentParser(description="Rung 2: prompt | model | parser.")
    parser.add_argument("--topic", default=TOPICS[0])
    parser.add_argument("--show-prompt", action="store_true", help="print the filled prompt only (no key needed)")
    parser.add_argument("--stream", action="store_true", help="print the answer piece by piece")
    parser.add_argument("--batch", action="store_true", help="answer all three topics in one call")
    args = parser.parse_args(argv)

    if args.show_prompt:
        show_prompt(args.topic)
        return 0

    print(tracing_status() + "\n")      # Step 14: is LangSmith recording this run?
    try:
        chain = build_chain(llm)
    except (MissingAPIKeyError, ValueError) as error:
        print(str(error).strip())
        return 1

    try:
        if args.batch:
            run_batch(chain, TOPICS)
        elif args.stream:
            run_stream(chain, args.topic)
        else:
            run_invoke(chain, args.topic)
    except Exception as error:
        print(f"\n{friendly_error(error)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
