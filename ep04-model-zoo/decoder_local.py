"""
decoder_local.py - a small DECODER with open weights, running on your own laptop (step 5).

Qwen2.5-0.5B-Instruct writes text one token at a time, like the big chat models do.
Its weights (the learned numbers) are a file on your disk, about 1 GB, downloaded once.
Licence: Apache-2.0, so you may use it in your own projects, also commercially.

While it answers, tokens flow through the model but the weights stay exactly the same.
We prove that with a fingerprint of all the weights, taken before and after.

Run:  python decoder_local.py
"""

import hashlib
import json
import time

import torch
from transformers import pipeline

from common import LOCAL_RESULT_FILE, OUTPUTS, OWNER_PROMPT, OWNER_SYSTEM, count_parameters, in_millions

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
MAX_NEW_TOKENS = 80                            # the longest reply we allow, in tokens


def load_decoder():
    """Load the model (the first run downloads it). device='cpu' works on every laptop."""
    return pipeline("text-generation", model=MODEL_ID, device="cpu")


def chat_messages(system: str, prompt: str) -> list[dict]:
    """A chat in the format every chat model understands: a list of role + content."""
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def generate(chat, system: str, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Send one chat to the model and return its reply as plain text."""
    messages = chat_messages(system, prompt)
    # do_sample=False: always pick the most likely next token, so you get the same reply every run
    out = chat(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"]   # the last message in the chat is the model's reply


def count_tokens(chat, text: str) -> int:
    """How many tokens (pieces of text) the model's tokenizer cuts this text into."""
    return len(chat.tokenizer(text)["input_ids"])


def prompt_tokens(chat, system: str, prompt: str) -> int:
    """Tokens the model really reads: our text plus the chat format markers around it."""
    text = chat.tokenizer.apply_chat_template(chat_messages(system, prompt), tokenize=False,
                                              add_generation_prompt=True)
    return count_tokens(chat, text)


def weights_fingerprint(model) -> str:
    """A short fingerprint of every weight. If one number changed, the fingerprint would change."""
    digest = hashlib.sha256()
    for p in model.parameters():
        digest.update(p.detach().contiguous().view(torch.uint8).numpy())   # the raw bytes of the numbers
    return digest.hexdigest()[:12]


def main() -> int:
    torch.manual_seed(SEED)   # only matters if you switch to do_sample=True

    start = time.perf_counter()
    chat = load_decoder()
    load_seconds = time.perf_counter() - start

    params = count_parameters(chat.model)
    print(f"Decoder: {MODEL_ID} (open weights, running on this computer)")
    print(f"Parameters: {params:,} (about {in_millions(params)})")
    print(f"Loaded in {load_seconds:.1f} s")

    before = weights_fingerprint(chat.model)
    start = time.perf_counter()
    reply = generate(chat, OWNER_SYSTEM, OWNER_PROMPT)
    seconds = time.perf_counter() - start
    after = weights_fingerprint(chat.model)

    print(f"\nPrompt: {OWNER_PROMPT}")
    print(f"\nReply ({seconds:.1f} s):\n{reply}")
    print(f"\nTokens: {prompt_tokens(chat, OWNER_SYSTEM, OWNER_PROMPT)} read, {count_tokens(chat, reply)} written")
    print(f"Weights fingerprint before {before}, after {after}: "
          f"{'unchanged' if before == after else 'CHANGED'}")

    OUTPUTS.mkdir(exist_ok=True)
    result = {"model": MODEL_ID, "parameters": params, "seconds": round(seconds, 1), "reply": reply}
    LOCAL_RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved outputs/{LOCAL_RESULT_FILE.name} (decoder_cloud.py compares with it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
