# Your Turn: add true semantic chunking and try to beat the leader

All four of our splitters cut by **rules**: a blank line, a size, a token count, a heading. Real
**semantic chunking** cuts by **meaning**: it embeds every sentence and starts a new chunk where the
meaning of the next sentence jumps away from the one before it (where the embedding distance spikes).
LangChain has one: `SemanticChunker` in the `langchain-experimental` package.

Can a fifth strategy, `semantic`, beat **headers + recursive** on the scoreboard?

| To beat | hit@3 | found first | context per question |
|---|---|---|---|
| headers + recursive (Step 11) | 10/10 | 10/10 | 305 MiniLM tokens |

## The task

1. Install the extra package into the shared `.venv` (it fits the course versions and changes nothing else):

   ```bash
   pip install langchain-experimental==0.4.2
   ```

   Like `langchain-community`, this package is being sunset, so it prints a `DeprecationWarning` when you import it.
2. In `chunklab/strategies.py`, add a `"semantic"` entry to `STRATEGIES` and `LABELS`, and a branch in
   `make_chunks` that uses:

   ```python
   from langchain_experimental.text_splitter import SemanticChunker
   from llm import get_embeddings

   SemanticChunker(get_embeddings(), breakpoint_threshold_type="percentile").split_documents(docs)
   ```

3. Run `python -m chunklab measure`. **Check the `largest` and `over 254` columns first.** A chunk longer
   than 254 MiniLM tokens is cut off when it is embedded (Step 4): the words after that point do not count
   in the search at all. Fix that before you trust any score.
4. Run `python -m chunklab index`, then `python -m chunklab score`. The scoreboard now has five columns.
5. Try the other `breakpoint_threshold_type` values (`"standard_deviation"`, `"interquartile"`, `"gradient"`)
   and `breakpoint_threshold_amount`. Keep notes: chunks, hit@3, found first, context.
6. Run `python -m pytest`. `test_no_chunk_is_longer_than_the_embedder_reads` now checks your strategy too.

Post your best scoreboard line (and which settings gave it) in the video comments.

## Done when

- [ ] `python -m chunklab measure` shows a `semantic` row with `0` in the `over 254` column
- [ ] `python -m chunklab score` shows five columns and names a winner
- [ ] You can say, with numbers, whether meaning-based cuts beat heading-based cuts on this handbook
- [ ] `python -m pytest` passes

<details>
<summary>Hint</summary>

- `SemanticChunker` embeds every sentence of the handbook, so indexing takes longer than the other strategies.
- Some settings find almost no breakpoints and hand back whole files: look at the `chunks` column.
- To keep every chunk under the limit, run the semantic chunks through `section_splitter()` from
  `strategies.py` (the token-measured recursive splitter of Step 6), the same two-stage idea as headers + recursive.
- Our handbook is small and already has good headings. Semantic chunking earns its keep on text **without**
  headings, such as a PDF that came out as one long page. Try it on the PDF's raw text too.

</details>

No solutions are published yet. A worked solution will be added to the course repo later, after the episode is live, so try it yourself first.
