"""
chunklab/show.py - Steps 2-6: see exactly where each splitter cuts.

    python -m chunklab show char                 Step 2  CharacterTextSplitter (and its size warnings)
    python -m chunklab show recursive            Step 3  RecursiveCharacterTextSplitter, overlap in amber
    python -m chunklab show tokens               Step 4  TokenTextSplitter, counted in two kinds of tokens
    python -m chunklab show minilm               Step 4  the recursive splitter, measured in MiniLM tokens
    python -m chunklab show headers              Step 5  MarkdownHeaderTextSplitter and its metadata
    python -m chunklab show headers+recursive    Step 6  headers first, then long sections split again

Add --file plans.md (or any handbook file) to look at another file. The default is refunds.md.
No API key and no internet needed (after the first download of the MiniLM model and the tiktoken file).

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse

from rich.console import Console
from rich.text import Text

from chunklab.strategies import (
    CHUNK_OVERLAP, CHUNK_SIZE, HEADERS, SECTION_MAX_TOKENS, STRATEGIES, TOKEN_ENCODING, breadcrumb, count_tokens,
    character_splitter, load_handbook, make_chunks, number_chunks, recursive_splitter, shared_text, split_by_headers,
    tag, text_limit,
)

console = Console(highlight=False, soft_wrap=True)
CHUNK_COLOURS = ["cyan", "bright_white"]      # neighbouring chunks take turns
OVERLAP = "bold yellow"                       # amber: text that is stored twice
TOO_BIG = "bold magenta"                      # magenta: bigger than we asked for
SAMPLE = "Report a missed lunch or dinner within 6 hours of the end of your slot."   # a line from refunds.md


def say(*parts):
    """Print one line made of (text, style) pairs or plain strings."""
    line = Text()
    for part in parts:
        if isinstance(part, tuple):
            line.append(*part)      # (text, style)
        else:
            line.append(part)       # plain text
    console.print(line)


def per_file(chunks):
    """'delivery.md 10, food_safety.md 10, ...' - how many chunks each file gave."""
    counts = {}
    for chunk in chunks:
        counts[chunk.metadata["source"]] = counts.get(chunk.metadata["source"], 0) + 1
    return ", ".join(f"{name} {n}" for name, n in counts.items())


def overlaps(chunks):
    """For each neighbouring pair inside one file: the text they share (or '')."""
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata["source"] == b.metadata["source"]]
    return [shared_text(a.page_content, b.page_content) for a, b in pairs]


def show_chunks(chunks, file_name, details, too_big=lambda chunk: False):
    """Print every chunk of one file in turns of colour, with the overlap marked « » in amber."""
    mine = [chunk for chunk in chunks if chunk.metadata["source"] == file_name]
    if not mine:
        names = sorted({chunk.metadata["source"] for chunk in chunks})
        raise SystemExit(f"No file called {file_name!r}. Try one of: {', '.join(names)}")
    say("")
    say((f"{file_name}, chunk by chunk", "bold"), f" ({len(mine)} chunks, overlap with the previous chunk in amber between « »):")
    previous = None
    for number, chunk in enumerate(mine):
        colour = CHUNK_COLOURS[number % 2]
        header_style = TOO_BIG if too_big(chunk) else "dim"
        say((f"── {tag(chunk)} · {details(chunk)} ──", header_style))
        shared = shared_text(previous.page_content, chunk.page_content) if previous else ""
        text = Text()
        if shared:
            text.append(f"«{shared}»", style=OVERLAP)
        text.append(chunk.page_content[len(shared):], style=colour)
        console.print(text)
        previous = chunk


# ---------------------------------------------------------------------------
# Step 2 - CharacterTextSplitter
# ---------------------------------------------------------------------------

def show_character(file_name, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    say(("Splitter: ", "bold"), f"CharacterTextSplitter(separator='\\n\\n', chunk_size={size}, chunk_overlap={overlap})")
    say("It cuts ONLY at blank lines, then glues neighbouring pieces together while they fit in "
        f"{size} characters.")
    # any "Created a chunk of size ..." warnings print while it splits
    chunks = number_chunks(character_splitter(size, overlap).split_documents(load_handbook()))
    say(f"Whole handbook: {len(chunks)} chunks ({per_file(chunks)})")

    too_big = [chunk for chunk in chunks if len(chunk.page_content) > size]
    if too_big:
        say((f"{len(too_big)} chunks are longer than {size} characters", TOO_BIG),
            ": that is what each 'Created a chunk of size ...' warning means.")
        for chunk in too_big:
            first_line = chunk.page_content.split("\n")[0][:60]
            say(f"  {tag(chunk):<16} {len(chunk.page_content)} characters, starts with: {first_line}")
        say("A piece with no blank line inside it cannot be cut by this splitter, so it stays whole.")

    shared = overlaps(chunks)
    with_overlap = [s for s in shared if s]
    headings = [s for s in with_overlap if s.startswith("#")]
    say(f"Overlap: {len(with_overlap)} of {len(shared)} neighbouring pairs share text "
        f"({len(headings)} headings, {len(with_overlap) - len(headings)} other short pieces, "
        f"at most {max(map(len, with_overlap), default=0)} characters).")
    say(f"chunk_overlap={overlap} can only repeat WHOLE pieces that fit in {overlap} characters, "
        "and most paragraphs are longer.")

    def details(chunk):
        length = len(chunk.page_content)
        return f"{length} characters" + (" · longer than asked" if length > size else "")

    show_chunks(chunks, file_name, details, too_big=lambda chunk: len(chunk.page_content) > size)


# ---------------------------------------------------------------------------
# Step 3 - RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------

def show_recursive(file_name, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    say(("Splitter: ", "bold"), f"RecursiveCharacterTextSplitter(chunk_size={size}, chunk_overlap={overlap})")
    say('Separators it tries, biggest first: "\\n\\n" (blank line), "\\n" (line break), " " (space), "" (any character).')
    say("A smaller separator is used only for a piece that is still too big. Then pieces are glued up to "
        f"{size} characters, and the end of one chunk is repeated at the start of the next (the overlap).")
    chunks = number_chunks(recursive_splitter(size, overlap).split_documents(load_handbook()))
    sizes = [len(chunk.page_content) for chunk in chunks]
    say(f"Whole handbook: {len(chunks)} chunks ({per_file(chunks)}), {min(sizes)} to {max(sizes)} characters")

    shared = overlaps(chunks)
    with_overlap = [s for s in shared if s]
    headings = [s for s in with_overlap if s.startswith("#")]
    say(f"Overlap: {len(with_overlap)} of {len(shared)} neighbouring pairs share text "
        f"({len(headings)} of them share just a heading line), {sum(map(len, with_overlap)):,} characters stored twice.")

    show_chunks(chunks, file_name, lambda chunk: f"{len(chunk.page_content)} characters")


# ---------------------------------------------------------------------------
# Step 4 - TokenTextSplitter, and measuring with MiniLM's own tokenizer
# ---------------------------------------------------------------------------

def show_tokens(file_name):
    import tiktoken

    encoding = tiktoken.get_encoding(TOKEN_ENCODING)       # the same tokenizer TokenTextSplitter uses

    def tiktoken_count(text):
        return len(encoding.encode(text))

    say(("Splitter: ", "bold"), STRATEGIES["tokens"])
    say("It counts tiktoken tokens (OpenAI's tokenizer). Without encoding_name it would use 'gpt2'.")
    say("")
    say(("One sentence, four rulers:", "bold"), f' "{SAMPLE}"')
    say(f"  characters {len(SAMPLE)} · words {len(SAMPLE.split())} · "
        f"{TOKEN_ENCODING} tokens {tiktoken_count(SAMPLE)} · MiniLM tokens {count_tokens(SAMPLE)}")

    chunks = make_chunks("tokens")
    minilm = [count_tokens(chunk.page_content) for chunk in chunks]
    limit = text_limit()
    say("")
    say(f"Whole handbook: {len(chunks)} chunks ({per_file(chunks)}), each at most 100 {TOKEN_ENCODING} tokens.")
    say(f"The same chunks in MiniLM tokens: {min(minilm)} to {max(minilm)}. MiniLM reads {limit} tokens of text "
        f"(256 minus 2 markers), so ", ("no chunk is cut off", "bold green") if max(minilm) <= limit
        else ("some chunks are cut off", TOO_BIG), ".")

    show_chunks(chunks, file_name, lambda chunk: f"{tiktoken_count(chunk.page_content)} {TOKEN_ENCODING} tokens · "
                                                 f"{count_tokens(chunk.page_content)} MiniLM tokens")

    # What if we did not split at all? One vector per whole file.
    say("")
    say(("Without splitting (one vector per whole file):", "bold"))
    docs = load_handbook()
    for doc in docs:
        size = count_tokens(doc.page_content)
        lost = size - limit
        say(f"  {doc.metadata['source']:<16} {size} MiniLM tokens -> MiniLM reads the first {limit} and ignores ",
            (f"{lost} ({lost / size:.0%})", "bold yellow"))
    prove_truncation(docs[[doc.metadata["source"] for doc in docs].index(file_name)], limit)


def prove_truncation(doc, limit):
    """Embed a whole file and only its first `limit` tokens: if the vectors match, the rest was ignored."""
    import numpy as np

    from chunklab.strategies import tokenizer
    from llm import get_embeddings

    first_part = tokenizer().convert_tokens_to_string(tokenizer().tokenize(doc.page_content)[:limit])
    whole, start = np.array(get_embeddings().embed_documents([doc.page_content, first_part]))
    similarity = float(whole @ start / (np.linalg.norm(whole) * np.linalg.norm(start)))
    say(f"Proof with {doc.metadata['source']}: the vector of the whole file and the vector of only its first "
        f"{limit} tokens have cosine similarity ", (f"{similarity:.4f}", "bold yellow"),
        ": the rest of the file changed nothing.")
    say("(The 'Token indices sequence length' warning above came from COUNTING whole files with the tokenizer, "
        "whose own file says 512. When MiniLM embeds a long text, it cuts it at 256 without a word.)")


def show_minilm(file_name):
    say(("Splitter: ", "bold"), f"RecursiveCharacterTextSplitter.from_huggingface_tokenizer(MiniLM tokenizer, "
                                f"chunk_size={SECTION_MAX_TOKENS}, chunk_overlap=30)")
    say("The same recursive cutting as Step 3, but chunk_size is now counted in MiniLM tokens, "
        "the units the embedder reads.")
    from chunklab.strategies import number_chunks, section_splitter

    chunks = number_chunks(section_splitter().split_documents(load_handbook()))
    sizes = [count_tokens(chunk.page_content) for chunk in chunks]
    say(f"Whole handbook: {len(chunks)} chunks ({per_file(chunks)}), {min(sizes)} to {max(sizes)} MiniLM tokens: "
        f"every chunk at or under {SECTION_MAX_TOKENS}, and MiniLM reads {text_limit()}.")
    show_chunks(chunks, file_name, lambda chunk: f"{count_tokens(chunk.page_content)} MiniLM tokens")


# ---------------------------------------------------------------------------
# Steps 5 and 6 - MarkdownHeaderTextSplitter, then a recursive split of long sections
# ---------------------------------------------------------------------------

def show_headers(file_name):
    say(("Splitter: ", "bold"), f"MarkdownHeaderTextSplitter(headers_to_split_on={HEADERS}, strip_headers=False)")
    say("It cuts at every # and ## line and writes the heading into metadata, under the key you chose.")
    sections = number_chunks(split_by_headers(load_handbook()))
    keys = sorted({key for section in sections for key in section.metadata if key != "chunk"})
    say(f"Whole handbook: {len(sections)} sections ({per_file(sections)}). Metadata keys: {keys}")

    def details(section):
        return f"{count_tokens(section.page_content)} MiniLM tokens · {breadcrumb(section)}"

    mine = [section for section in sections if section.metadata["source"] == file_name]
    say("")
    say((f"{file_name}, section by section", "bold"), f" ({len(mine)} sections):")
    for number, section in enumerate(mine):
        say((f"── {tag(section)} · {details(section)} ──", "dim"))
        say(("metadata: ", "dim"), (str({k: v for k, v in section.metadata.items() if k != "chunk"}), "magenta"))
        console.print(Text(section.page_content, style=CHUNK_COLOURS[number % 2]))

    long_ones = [section for section in sections if count_tokens(section.page_content) > SECTION_MAX_TOKENS]
    say("")
    say(f"Sections longer than {SECTION_MAX_TOKENS} MiniLM tokens (Step 6 splits them again): {len(long_ones)}")
    for section in long_ones:
        say(f"  {section.metadata['source']} > {section.metadata.get('h2', '')}: "
            f"{count_tokens(section.page_content)} MiniLM tokens")


def show_headers_recursive(file_name, file_given):
    from chunklab.strategies import split_headers_then_recursive

    sections = split_by_headers(load_handbook())
    chunks = number_chunks(split_headers_then_recursive(load_handbook()))
    say(("Stage 1: ", "bold"), f"MarkdownHeaderTextSplitter -> {len(sections)} sections")
    say(("Stage 2: ", "bold"), f"RecursiveCharacterTextSplitter.from_huggingface_tokenizer(MiniLM, "
                               f"chunk_size={SECTION_MAX_TOKENS}, chunk_overlap=30) -> {len(chunks)} chunks")
    sizes = [count_tokens(chunk.page_content) for chunk in chunks]
    say(f"Whole handbook: {len(chunks)} chunks ({per_file(chunks)}), {min(sizes)} to {max(sizes)} MiniLM tokens")

    split_sections = {}
    for chunk in chunks:
        if "part" in chunk.metadata:
            split_sections.setdefault((chunk.metadata["source"], breadcrumb(chunk)), []).append(chunk)
    say(f"Sections that were too long and got split: {len(split_sections)}")
    for (source, crumb), parts in split_sections.items():
        say("")
        say((f"{source} > {crumb}", "bold"), f" -> {len(parts)} parts")
        previous = None
        for number, part in enumerate(parts):
            meta = {k: v for k, v in part.metadata.items() if k != "chunk"}
            say((f"── {tag(part)} · {count_tokens(part.page_content)} MiniLM tokens ──", "dim"))
            say(("metadata: ", "dim"), (str(meta), "magenta"))
            shared = shared_text(previous.page_content, part.page_content) if previous else ""
            text = Text()
            if shared:
                text.append(f"«{shared}»", style=OVERLAP)
            text.append(part.page_content[len(shared):], style=CHUNK_COLOURS[number % 2])
            console.print(text)
            previous = part
    say("")
    say(("Every part keeps its section's metadata", "bold green"), ", so a chunk from the middle of a section "
        "still knows its file and its headings.")
    if file_given:
        show_chunks(chunks, file_name, lambda chunk: f"{count_tokens(chunk.page_content)} MiniLM tokens · "
                                                     f"{breadcrumb(chunk)}")


MODES = {
    "char": show_character, "character": show_character,
    "recursive": show_recursive,
    "tokens": show_tokens,
    "minilm": show_minilm,
    "headers": show_headers,
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chunklab show", description="See where a splitter cuts.")
    parser.add_argument("mode", choices=[*MODES, "headers+recursive"])
    parser.add_argument("--file", default=None, help="handbook file to print chunk by chunk (default refunds.md)")
    parser.add_argument("--size", type=int, default=CHUNK_SIZE, help="chunk_size for char and recursive (default 400)")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="chunk_overlap for char and recursive")
    args = parser.parse_args(argv)

    file_name = args.file or "refunds.md"
    if args.mode == "headers+recursive":
        show_headers_recursive(file_name, file_given=args.file is not None)
    elif args.mode in ("char", "character", "recursive"):
        MODES[args.mode](file_name, args.size, args.overlap)
    else:
        MODES[args.mode](file_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
