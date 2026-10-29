"""
gemini_stream.py - stream a Gemini answer piece by piece, then read the token bill (step 12).

A cloud model writes the same way as our small decoder: one token after another. With .stream()
you see the pieces ("chunks") as they arrive instead of waiting for the whole answer. At the end,
the response says how many tokens went in and came out. Free-tier limits and prices count those.

Run:  python gemini_stream.py      (needs your free Gemini key in the .env file at the repo root)

Note (2026): Gemini 3.5 Flash-Lite and newer ignore temperature, top_p and top_k (Google changelog,
2026-07-21). That is why steps 7 and 8 play with those settings on the local model instead.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage

from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

SYSTEM = "You are a patient teacher. Use simple words."
QUESTION = "In three short sentences, explain to a college fresher why a chatbot shows its answer a few words at a time."

# What each provider calls "stopped because of the output limit" (Gemini, OpenAI and Ollama, Claude)
LIMIT_REASONS = {"MAX_TOKENS", "length", "max_tokens"}


def build_messages(question: str = QUESTION) -> list:
    return [SystemMessage(SYSTEM), HumanMessage(question)]


def stream_answer(llm, messages, show: bool = True):
    """Print every chunk as it arrives, with a | after each one.

    Returns (the whole message, the text of every chunk, seconds until the first chunk).
    """
    full, pieces, first = None, [], None
    start = time.perf_counter()
    for chunk in llm.stream(messages):
        if first is None:
            first = time.perf_counter() - start
        pieces.append(chunk.text)                         # .text, not .content (Gemini 3 sends content blocks)
        if show:
            print(chunk.text, end="|", flush=True)
        full = chunk if full is None else full + chunk    # each chunk brings its share of the token counts: add them up
    return full, pieces, first


def finish_reason(message) -> str:
    """Why the model stopped. Each provider uses its own key and words for it."""
    meta = message.response_metadata or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason") or meta.get("done_reason")
    return str(reason) if reason else "not reported"


def hit_limit(message) -> bool:
    return finish_reason(message) in LIMIT_REASONS


def token_line(message) -> str:
    """The token bill from the response's usage metadata."""
    usage = message.usage_metadata
    if not usage:
        return "Tokens: not reported by this model"
    thinking = (usage.get("output_token_details") or {}).get("reasoning", 0)
    extra = f" (of which {thinking} thinking)" if thinking else ""
    return f"Tokens: {usage['input_tokens']} in, {usage['output_tokens']} out{extra}, {usage['total_tokens']} total"


def main(llm=None) -> int:
    try:
        provider = get_provider()
        print(f"Provider: {provider}   Model: {get_model_name(provider)}")
        if llm is None:
            llm = get_llm()
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")   # friendly help (no key, or a typo in LLM_PROVIDER), not a traceback
        return 1

    print(f"\nYou asked: {QUESTION}\n")
    print("Streaming (each | marks the end of one chunk):")
    start = time.perf_counter()
    full, pieces, first = stream_answer(llm, build_messages())
    total = time.perf_counter() - start
    print(f"\n\nChunks received: {len(pieces)}   first chunk after {first:.1f} s, whole answer after {total:.1f} s")
    if pieces and not pieces[-1]:
        print("(The last chunk had no text: it only marks the end of the stream.)")
    print(token_line(full))
    print(f"Finish reason: {finish_reason(full)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
