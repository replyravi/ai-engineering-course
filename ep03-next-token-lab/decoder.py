"""
decoder.py - shared helpers: read our prompts and load the small decoder we look inside.

The model is SmolLM2-135M, "HuggingFaceTB/SmolLM2-135M" on Hugging Face. It is a small GPT-style,
decoder-only model (the same design as Llama) with about 135 million learned numbers.
It is a BASE model: it was trained only to continue text, not to chat or follow instructions.
Licence: Apache-2.0. The first load downloads about 269 MB. After that it works offline.

Used by next_token.py, greedy_loop.py, sampling.py, kv_cache_timing.py and three_families.py.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROMPTS_FILE = HERE / "data" / "prompts.txt"
PLOTS = HERE / "plots"          # every picture lands here (git ignores it)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
SEED = 42                       # fixed seed: the "random" choices are the same on every run


def read_prompts(path=PROMPTS_FILE) -> dict:
    """Read data/prompts.txt into {"group name": [text, ...]}."""
    groups = {}
    group = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue                         # skip empty lines and comments
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1]               # a new group starts, like [cold_open]
            groups[group] = []
        else:
            groups[group].append(line)
    return groups


def load_tokenizer(name: str = MODEL_NAME):
    """The tokenizer that belongs to the model: text -> tokens -> IDs."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def load_decoder(name: str = MODEL_NAME):
    """Return (tokenizer, model), ready to write.

    dtype=torch.float32: the file on Hugging Face stores the numbers as bfloat16, and transformers 5
    keeps that type unless you ask. float32 is exact enough that the loop with and without the
    KV cache picks the same tokens (step 9). On the recording laptop's CPU it was also about
    twice as fast as bfloat16.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    transformers.logging.set_verbosity_error()   # hide the load report and progress messages
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()   # writing mode: switches off dropout, so the same input gives the same numbers
    return load_tokenizer(name), model


def next_token_logits(model, ids: list[int]) -> np.ndarray:
    """One forward pass over the whole text. Returns the scores for the NEXT token (one per vocabulary entry)."""
    import torch

    with torch.inference_mode():
        logits = model(torch.tensor([ids])).logits      # (1, positions, vocabulary)
    return logits[0, -1].double().numpy()              # only the last position predicts what comes next


def piece(tokenizer, token_id: int) -> str:
    """One token as readable text, in quotes, so you can see its leading space: ' tastes'."""
    return repr(tokenizer.decode([token_id]))


def save_note(path: Path) -> str:
    """'plots/x.png' instead of a long absolute path, for printing."""
    path = Path(path)
    return str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path)
