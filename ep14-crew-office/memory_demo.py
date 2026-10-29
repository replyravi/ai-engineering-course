"""
memory_demo.py - Step 8: crew memory, briefly. A shared office notebook the next shift can search.

What Crew(memory=True) does in CrewAI 1.15.28 (read from the installed code on 2026-10-26):
  1. BEFORE each task it searches memory with the task text and pastes up to 5 "Relevant
     memories" into the agent's prompt (a long search text costs a model call to plan the search),
  2. AFTER each task, extra model calls: one pulls short "memories" out of the answer, and one
     more per memory files it (scope, categories, importance),
  3. an embeddings model turns each memory into numbers and stores it (LanceDB, in a "memory"
     folder in CrewAI's data folder on your computer),
  4. every agent gets two extra tools, "Search memory" and "Save to memory".
The model for those calls is your first agent's model. The embeddings model is OpenAI's
text-embedding-3-large (it needs an OpenAI key) UNLESS you choose one: Crew(embedder=...).

Our outreach crew keeps memory OFF, because every lead must be its own case. This demo shows
why, with the memory store itself, a free local embedder, and NO model at all:

    python memory_demo.py

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import privacy  # noqa: F401  (switches off CrewAI telemetry; must come before crewai)

import os
import tempfile
import warnings

from crewai import Crew
from crewai.memory.unified_memory import Memory

from outreach_crew import LEADS, build_crew
from replay import ScriptedLLM, scripted_brief

warnings.filterwarnings("ignore", message=".*callbacks cannot be serialized.*")

LOCAL_EMBEDDER = {"provider": "onnx"}     # all-MiniLM-L6-v2 on your computer (Episode 13), no key, no cost
QUESTION = "parents or customers miss a time that changed"


def show_crew_memory_settings() -> None:
    """Build (do not run) the outreach crew with memory on, and print what CrewAI set up."""
    plain = build_crew(llm=ScriptedLLM(responses=[]), verbose=False)
    old_folder = os.environ.get("CREWAI_STORAGE_DIR")
    with tempfile.TemporaryDirectory() as folder:
        os.environ["CREWAI_STORAGE_DIR"] = folder      # building a Memory creates its store: keep it temporary
        crew = Crew(agents=plain.agents, tasks=plain.tasks, memory=True, embedder=LOCAL_EMBEDDER)
        without = Crew(agents=plain.agents, tasks=plain.tasks, memory=True)
    if old_folder is None:
        del os.environ["CREWAI_STORAGE_DIR"]
    else:
        os.environ["CREWAI_STORAGE_DIR"] = old_folder
    memory = crew._memory                  # CrewAI keeps the Memory object it built here
    print("Crew(memory=True, embedder={'provider': 'onnx'}) builds:")
    print(f"  {type(memory).__name__} with storage {type(memory._storage).__name__}, "
          f"scope {memory.root_scope}, embedder {type(memory.embedder).__name__}")
    print(f"  analysis model: the first agent's model ({type(memory.llm).__name__} here, Gemini in a real run)")
    print(f"  ... and without embedder=...: embedder {without._memory.embedder} -> OpenAI's default, "
          "which needs an OpenAI key\n")


def leak_demo(folder: str) -> None:
    # A scripted model with NO replies: if the memory asked any model anything, this would crash.
    memory = Memory(llm=ScriptedLLM(responses=[]), embedder=LOCAL_EMBEDDER, storage=folder,
                    consolidation_threshold=1.0)     # 1.0 = never merge memories (merging calls a model)

    print("1. Save each lead's facts as memories, one scope per lead (scope, categories and")
    print("   importance given by us, so no model is needed):")
    for lead in LEADS:
        facts = [f["text"] for f in scripted_brief(lead)["facts"]]
        memory.remember_many(facts, scope=f"/lead/{lead.stem}", categories=["lead"], importance=0.5)
        print(f"   /lead/{lead.stem:<24} {len(facts)} memories")
    memory.drain_writes()                            # saving runs in the background: wait for it

    print(f"\n2. The run for kayal-learning-circle searches memory: {QUESTION!r}")
    for match in memory.recall(QUESTION, limit=3, depth="shallow"):     # shallow = one vector search, no model
        owner = match.record.scope.rsplit("/", 1)[-1]
        flag = "" if owner == "kayal-learning-circle" else "   <- ANOTHER lead's fact"
        print(f"   {match.score:.3f}  {owner:<24} {match.record.content[:70]}{flag}")

    print("\n3. The same search inside one lead's scope:")
    for match in memory.scope("/lead/kayal-learning-circle").recall(QUESTION, limit=3, depth="shallow"):
        print(f"   {match.score:.3f}  {match.record.content[:70]}")


def main() -> int:
    show_crew_memory_settings()
    with tempfile.TemporaryDirectory() as folder:        # the demo store is deleted afterwards
        leak_demo(folder)
    print("\nMemory helps when later work SHOULD build on earlier work. For independent cases, like one")
    print("email per lead, it can carry one customer's details into another's email. We keep it off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
