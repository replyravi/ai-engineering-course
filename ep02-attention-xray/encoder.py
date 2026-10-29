"""
encoder.py - shared helpers: read our sentences and load the small encoder we look inside.

The model is DistilBERT, "distilbert/distilbert-base-uncased" on Hugging Face. It is a smaller copy
of BERT-base: the same 30,522-entry vocabulary and 768 numbers per token, but 6 layers instead of 12.
The first load downloads about 270 MB. After that it works offline.

Used by tokens.py, embeddings_peek.py, real_attention.py, two_meanings.py and explain.py.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
SENTENCES_FILE = HERE / "data" / "sentences.txt"
PLOTS = HERE / "plots"          # every picture lands here (git ignores it)

MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42                       # fixed seed: the "random" numbers are the same on every run


def read_sentences(path=SENTENCES_FILE) -> dict:
    """Read data/sentences.txt into {"group name": [sentence, ...]}."""
    groups = {}
    group = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue                         # skip empty lines and comments
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1]               # a new group starts, like [feedback]
            groups[group] = []
        else:
            groups[group].append(line)
    return groups


def load_tokenizer(name: str = MODEL_NAME):
    """The tokenizer that belongs to the model: text -> tokens -> IDs."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def load_encoder(name: str = MODEL_NAME):
    """Return (tokenizer, model), ready to read sentences.

    attn_implementation="eager" runs attention the plain way, so the model can hand back its
    attention weights. (The faster default, "sdpa", returns none of them in transformers 5.)
    """
    import transformers
    from transformers import AutoModel

    # Hide the load report: it lists the unused "guess the hidden word" layers, which we do not need.
    transformers.logging.set_verbosity_error()
    model = AutoModel.from_pretrained(name, attn_implementation="eager")
    model.eval()   # reading mode: switches off dropout, so the same sentence gives the same numbers
    return load_tokenizer(name), model


def save_note(path: Path) -> str:
    """'plots/x.png' instead of a long absolute path, for printing."""
    path = Path(path)
    return str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path)
