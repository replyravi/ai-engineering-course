"""
tests/test_eval.py - the retrieval scoreboard (eval_retrieval.py).

The last test runs the real golden set with the real local model and checks
that retrieval still finds every answer at k=3. If you change the handbook
(for example in the challenge), this test tells you if retrieval got worse.
"""

from langchain_core.documents import Document
from rag_fixtures import real_embeddings  # noqa: F401

from eval_retrieval import build_test_index, evaluate, is_hit, load_golden

CASE = {"question": "How long can I pause?", "sources": ["plans.md", "support.md"], "answer_contains": "10 days"}


def test_a_hit_needs_the_right_file_and_the_answer_phrase():
    right = Document(page_content="Pause for up to 10 days a month.", metadata={"source": "plans.md"})
    wrong_file = Document(page_content="Pause for up to 10 days a month.", metadata={"source": "refunds.md"})
    no_answer = Document(page_content="Pause from the app.", metadata={"source": "plans.md"})

    assert is_hit([no_answer, right], CASE)
    assert not is_hit([wrong_file, no_answer], CASE)


def test_golden_file_is_well_formed():
    golden = load_golden()
    assert len(golden["in_scope"]) >= 6
    for case in golden["in_scope"]:
        assert set(case) == {"question", "sources", "answer_contains"}
    assert golden["out_of_scope"]


def test_every_golden_answer_is_found_at_k3(real_embeddings):
    golden = load_golden()
    store, _ = build_test_index(400, 60, real_embeddings)

    rows, hits, _ = evaluate(store, golden, k=3)

    missed = [row["question"] for row in rows if not row["hit"]]
    assert missed == []
    assert hits == len(golden["in_scope"])
