"""
tools/docs.py - Step 13: retrieval as a TOOL. The agent searches the crawled rule book itself.

In Episode 07 our code always searched first, then asked the model. Here the model DECIDES
when to search and what to search for, like any other tool. Every result carries its source
URL, so the answer can say where a rule came from.

    python -m tools.docs "loading charge per bag"      the tool run by hand (no key; run ingest.py first)

Author: Ravi Shanker Singh (TechWithRavi)
"""

import sys

from langchain.tools import tool

from ingest import IndexNotBuiltError, chunk_tag, load_index

TOP_K = 3          # chunks per search


def make_search_tool(store=None, k=TOP_K):
    """The search_docs tool over a Chroma index. With store=None the index is opened on first use."""
    opened = {"store": store}

    def get_store():
        if opened["store"] is None:
            opened["store"] = load_index()          # loads the local embeddings model (a few seconds)
        return opened["store"]

    @tool
    def search_docs(query: str) -> str:
        """Search the Mandi Tool Belt rule book (crawled web pages) for rules about buyer charges,
        commission, market fee, loading, bag and crate sizes, quintals, grades and how to read the
        price board. Returns the 3 closest passages, each with its source URL. Prices are NOT here:
        use the SQL tools for prices and arrivals.
        """
        try:
            results = get_store().similarity_search_with_score(query, k=k)
        except IndexNotBuiltError as error:
            return f"Error: {error}"
        except Exception as error:          # e.g. an index built with another embeddings model
            return f"Error: the search failed ({error}). Rebuild the index with:  python ingest.py"
        blocks = []
        for number, (doc, distance) in enumerate(results, start=1):
            blocks.append(f"[{number}] {chunk_tag(doc)}  source: {doc.metadata['source']}  "
                          f"(distance {distance:.3f})\n{doc.page_content}")
        return "\n\n".join(blocks) if blocks else "No passages found."

    return search_docs


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    query = " ".join(argv) or "loading charge per bag"
    print(f'search_docs("{query}") ->')
    print(make_search_tool().invoke({"query": query}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
