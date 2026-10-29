"""
app.py - a tiny example project: "Explain any topic like I'm a beginner".

Each episode has its own app.py. This one shows the pattern every episode follows:
  1. build the chain in a function that ACCEPTS an llm (so tests can pass a fake one)
  2. use the real model from llm.py when no llm is given

Run it:   python app.py
Test it:  python -m pytest
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm

# The prompt: a system message (the AI's role) + the user's question.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly teacher. Explain things in 2 simple sentences."),
        ("human", "Explain {topic} to a beginner."),
    ]
)


def build_chain(llm=None):
    """Build the chain: prompt -> model -> plain text.

    llm: the chat model to use. If None, we use the real one from llm.py.
    """
    if llm is None:
        llm = get_llm()
    return prompt | llm | StrOutputParser()


def explain(topic: str, llm=None) -> str:
    """Return a beginner-friendly explanation of `topic`."""
    chain = build_chain(llm)
    return chain.invoke({"topic": topic})


if __name__ == "__main__":
    print(explain("embeddings"))
