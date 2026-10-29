"""
embed_router.py - Step 12: a FREE router (a "semantic router"). No chat model, just the local embeddings model.

    python embed_router.py                             four sample questions (no key needed)
    python embed_router.py "your question" "another one"

How it works:
  1. We write a few example questions for each route (ROUTE_EXAMPLES below).
  2. The local all-MiniLM-L6-v2 model turns every example into 384 numbers, once.
  3. A new question is embedded the same way. The example closest in meaning wins,
     and its route is the answer (cosine similarity: 1.0 = same direction).
  4. If even the closest example is far away (below MIN_SIMILARITY), we fall back to chat.

Compared with the LLM router (router.py): no API call, no cost, a few milliseconds, the same answer
every time. But it only knows what its examples cover, so tricky questions fool it more easily.
Step 16 measures both on the same 20 labelled questions.

The examples here are NOT the questions of data/router_questions.json. Testing a router on its own
examples would be like marking an exam with the answer key the student studied from.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import time

import numpy as np
from langchain_core.runnables import RunnableLambda

from llm import get_embeddings
from router import FALLBACK_ROUTE, Route

MIN_SIMILARITY = 0.30          # closest example below this: not sure -> fallback route

ROUTE_EXAMPLES = {
    "policy": [
        "What is your refund policy for a missed meal?",
        "Which areas of Pune do you deliver to?",
        "How do I pause my meal plan?",
        "What is the deposit for the steel box?",
        "What time is the cut-off for changing lunch?",
        "Is the food safe for someone with a peanut allergy?",
        "How much does the monthly plan cost?",
        "When is customer support open?",
    ],
    "orders": [
        "How many orders did we get yesterday?",
        "How many thalis were sold this month?",
        "Which menu item sold the most last week?",
        "What was the total revenue in September?",
        "How many dinner orders came from Zone B?",
        "Show the number of orders per day for the last 7 days.",
        "How many add-ons were ordered with lunch?",
        "Which zone had the most orders in October?",
    ],
    "chat": [
        "Hi there!",
        "Thank you so much",
        "Tell me a joke",
        "How are you today?",
        "Who are you?",
        "Write a short poem about rain",
        "Good night",
        "What is the capital of France?",
    ],
}

SAMPLE_QUESTIONS = [
    "What is the refund window for a missed lunch?",
    "How many Jain thalis were sold last week?",
    "Tell me a joke about dosas",
    "drop table orders",
]


def unit_vectors(vectors):
    """Scale every vector to length 1, so a dot product is the cosine similarity."""
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


class SemanticRouter:
    """Nearest example wins. Built once (embeds the examples), then routes many questions."""

    def __init__(self, embeddings=None, examples=None, min_similarity=MIN_SIMILARITY):
        self.embeddings = embeddings or get_embeddings()
        self.examples = examples or ROUTE_EXAMPLES
        self.min_similarity = min_similarity
        self.routes = [route for route, texts in self.examples.items() for _ in texts]
        self.texts = [text for texts in self.examples.values() for text in texts]
        self.vectors = unit_vectors(self.embeddings.embed_documents(self.texts))

    def closest(self, question):
        """(similarity, route, example) of the closest example."""
        similarities = self.vectors @ unit_vectors(self.embeddings.embed_query(question))
        best = int(np.argmax(similarities))
        return float(similarities[best]), self.routes[best], self.texts[best]

    def route(self, question):
        """A Route, just like the LLM router returns, with the similarity as the reason."""
        similarity, route, example = self.closest(question)
        if similarity < self.min_similarity:
            return Route(destination=FALLBACK_ROUTE,
                         reason=f"fallback: closest example only {similarity:.2f} (below {self.min_similarity:.2f})")
        return Route(destination=route, reason=f'{similarity:.2f} like "{example}"')


def build_semantic_router(embeddings=None, router=None):
    """{"question": str} -> Route, so it plugs into the desk exactly where the LLM router goes."""
    router = router or SemanticRouter(embeddings)
    return RunnableLambda(lambda inputs: router.route(inputs["question"]))


def main():
    parser = argparse.ArgumentParser(description="Route questions with local embeddings (no key).")
    parser.add_argument("questions", nargs="*", default=SAMPLE_QUESTIONS)
    args = parser.parse_args()

    start = time.perf_counter()
    router = SemanticRouter()
    print(f"Embedded {len(router.texts)} examples ({len(router.examples)} routes) in "
          f"{time.perf_counter() - start:.1f} s. Fallback to {FALLBACK_ROUTE} below {MIN_SIMILARITY:.2f}.\n")
    for question in args.questions:
        start = time.perf_counter()
        route = router.route(question)
        ms = (time.perf_counter() - start) * 1000
        print(f"{question}\n  route={route.destination}  ({route.reason})  {ms:.0f} ms")


if __name__ == "__main__":
    main()
