"""
three_families.py - one job for each Transformer family, in one script (step 10).

    Encoder-only     (BERT family)   reads the whole input at once, left AND right -> a label or a vector
    Encoder-decoder  (T5, BART)      an encoder reads the input, a decoder writes new text while
                                     looking back at it (cross-attention)  -> translation, summaries
    Decoder-only     (GPT family)    only looks back, and keeps writing  -> open-ended text, chat

Three small models from Hugging Face (all Apache-2.0). The first run downloads about 270 MB for
the encoder and 242 MB for T5-small; the decoder is already on your computer from step 2.

Run:  python three_families.py
"""

from decoder import load_decoder, next_token_logits, read_prompts
from greedy_loop import generate

ENCODER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ENCODER_DECODER = "google-t5/t5-small"

FAMILIES = [
    # family, short model name, what its attention can see, cross-attention, the job it does here
    ("Encoder-only", "DistilBERT SST-2", "every token, left and right", "no", "label the feedback lines"),
    ("Encoder-decoder", "T5-small", "encoder: all; decoder: past", "yes", "summarise the notice"),
    ("Decoder-only", "SmolLM2-135M", "only the past", "no", "continue a sentence"),
]


def count(model) -> str:
    return f"{sum(p.numel() for p in model.parameters()):,} learned numbers"


def load_classifier():
    """A small encoder fine-tuned to say POSITIVE or NEGATIVE (the text-classification pipeline)."""
    import transformers
    from transformers import pipeline

    transformers.logging.set_verbosity_error()
    return pipeline("text-classification", model=ENCODER, device="cpu")


def load_t5():
    """T5-small: the summarization pipeline is gone in transformers 5, so we call the model ourselves."""
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    transformers.logging.set_verbosity_error()
    model = AutoModelForSeq2SeqLM.from_pretrained(ENCODER_DECODER, attn_implementation="eager")
    model.eval()
    return AutoTokenizer.from_pretrained(ENCODER_DECODER), model


def classify(classifier, lines: list[str]) -> list[tuple[str, float]]:
    """One label and its score for each line."""
    return [(r["label"], r["score"]) for r in classifier(lines)]


def summarise(tokenizer, model, text: str, max_new_tokens: int = 60) -> str:
    """T5 knows its jobs by a text prefix. 'summarize: ' asks for a summary."""
    import torch

    inputs = tokenizer("summarize: " + text, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)   # greedy, like step 5
    return tokenizer.decode(out[0], skip_special_tokens=True)


def continue_text(tokenizer, model, prompt: str, new_tokens: int = 20) -> str:
    prompt_ids = tokenizer(prompt)["input_ids"]
    new_ids, _ = generate(lambda ids: next_token_logits(model, ids), prompt_ids, new_tokens,
                          eos_id=tokenizer.eos_token_id)
    return tokenizer.decode(new_ids)


def family_table() -> str:
    widths = [17, 18, 30, 17, 0]
    header = ["Family", "Model", "Attention can see", "Cross-attention", "Job here"]
    lines = ["".join(f"{h:<{w}}" for h, w in zip(header, widths)).rstrip()]
    lines.append("-" * 100)
    for row in FAMILIES:
        lines.append("".join(f"{cell:<{w}}" for cell, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def main() -> int:
    prompts = read_prompts()

    classifier = load_classifier()
    print(f"1. Encoder-only: {ENCODER} ({count(classifier.model)})")
    print("   reads each whole line at once, left and right, and gives ONE label")
    for line, (label, score) in zip(prompts["feedback"], classify(classifier, prompts["feedback"])):
        print(f"   {label:<9}{score:.3f}  {line}")
    print("   It only knows two labels. A suggestion still has to be POSITIVE or NEGATIVE.\n")

    t5_tokenizer, t5 = load_t5()
    notice = prompts["notice"][0]
    summary = summarise(t5_tokenizer, t5, notice)
    print(f"2. Encoder-decoder: {ENCODER_DECODER} ({count(t5)})")
    print("   the encoder reads the notice, the decoder writes a summary token by token, looking back at it")
    print(f"   notice  ({len(notice.split())} words): {notice}")
    print(f"   summary ({len(summary.split())} words): {summary}")
    print(f"   Does the summary keep the new opening time (7:30)? {'7:30' in summary}")
    print("   Check every summary yourself: a small model can drop the main point and still sound fine.\n")

    tokenizer, decoder = load_decoder()
    prompt = prompts["continue"][0]
    print(f"3. Decoder-only: HuggingFaceTB/SmolLM2-135M ({count(decoder)})")
    print("   only looks back, and just keeps writing (greedy, 20 new tokens)")
    print(f"   {prompt!r} -> {continue_text(tokenizer, decoder, prompt)!r}\n")

    print("Which family did what:")
    print(family_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
