"""
skills/chat_model.py - the small local chat model that summarise.py and translate.py share.

transformers v5 removed the old "summarization" and "translation" pipelines (step 3).
The 2026 way: ONE small instruction-tuned chat model through the "text-generation" pipeline.
A system message tells it which job to do, exactly like a system prompt for Gemini.

    reply = ask_local("You summarise notes in one sentence.", "Summarise: ...")
    reply.text, reply.seconds, reply.tokens_in, reply.tokens_out
"""

from dataclasses import dataclass

from common import CHAT_MODEL, Stopwatch, device
from skills.lazy import lazy_load


@dataclass
class Reply:
    text: str          # what the model wrote
    seconds: float     # how long the writing took (not counting the model load)
    tokens_in: int     # tokens the model read: system + user message + chat-format markers
    tokens_out: int    # tokens the model wrote
    hit_limit: bool    # True when it stopped because it reached max_new_tokens (it was not finished)


def load_chat():
    """Build the text-generation pipeline (downloads about 1 GB the very first time)."""
    from transformers import pipeline

    return pipeline(
        "text-generation",
        model=CHAT_MODEL,
        dtype="auto",          # use the dtype stored in the model files (bfloat16 here). Old name: torch_dtype
        device=device(),       # "cpu" by default; "mps" or "cuda" for a GPU
    )


def chat_model():
    """The loaded pipeline. Loaded on first use, then kept (see skills/lazy.py)."""
    return lazy_load("chat", load_chat)


def ask_local(system: str, user: str, max_new_tokens: int = 80) -> Reply:
    """Send one system + user message to the local model and return its reply."""
    chat = chat_model()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    with Stopwatch() as watch:
        # do_sample=False: always pick the most likely next token, so the same input gives the same answer
        output = chat(messages, max_new_tokens=max_new_tokens, do_sample=False)
    text = output[0]["generated_text"][-1]["content"]           # the last message is the model's reply
    text = text.replace("\ufffd", "").strip()   # a cut in the middle of a Hindi letter leaves a broken character

    tokenizer = chat.tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens_in = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    tokens_out = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return Reply(text, watch.seconds, tokens_in, tokens_out, hit_limit=tokens_out >= max_new_tokens - 1)


def count_tokens(text: str) -> int:
    """How many tokens the chat model splits this text into (length limits count tokens, not words)."""
    return len(chat_model().tokenizer(text, add_special_tokens=False)["input_ids"])
