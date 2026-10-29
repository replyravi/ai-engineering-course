"""
explain.py - ask Gemini to explain OUR attention numbers to a fresher (step 14).

  1. Run DistilBERT on the pronoun sentence and collect real numbers (same as step 12).
  2. Put those numbers in a prompt.
  3. Ask the model in .env (Gemini by default) for a three-sentence explanation.

The numbers come from your computer. Only the explanation comes from Gemini, and Gemini only
sees the short prompt printed below.

Run:  python explain.py      (needs your free Gemini key in the .env file at the repo root)
"""

from langchain_core.messages import HumanMessage, SystemMessage

from llm import MissingAPIKeyError, get_llm

SYSTEM = "You are a patient teacher. You explain AI to college freshers in simple, friendly words."


def gather_facts(tokenizer=None, model=None) -> dict:
    """Real numbers from the local model: the best head for 'they' -> 'students' and its top 3 tokens."""
    from encoder import load_encoder, read_sentences
    from real_attention import FROM_WORD, TO_WORD, attention_maps, favourite_targets, find_head, top_attended

    if model is None:
        tokenizer, model = load_encoder()
    sentence = read_sentences()["pronoun"][0]
    tokens, weights = attention_maps(sentence, tokenizer, model)
    layer, head, _ = find_head(weights, tokens, FROM_WORD, TO_WORD)
    layers, heads = weights.shape[:2]
    return {
        "sentence": sentence,
        "from_word": FROM_WORD,
        "to_word": TO_WORD,
        "layer": layer,
        "head": head,
        "top": top_attended(weights, tokens, FROM_WORD, layer, head),
        "heads_agreeing": favourite_targets(weights, tokens, FROM_WORD)[TO_WORD],
        "total_heads": layers * heads,
    }


def build_messages(facts: dict) -> list:
    top = ", ".join(f"{token} {weight:.2f}" for token, weight in facts["top"])
    question = (
        f"I ran a small encoder model, DistilBERT, on this sentence:\n"
        f"\"{facts['sentence']}\"\n"
        f"In layer {facts['layer']}, head {facts['head']}, the token \"{facts['from_word']}\" gave its "
        f"attention (out of 1.00) mostly to: {top}.\n"
        f"In {facts['heads_agreeing']} of the {facts['total_heads']} heads, \"{facts['to_word']}\" was the token "
        f"\"{facts['from_word']}\" looked at most.\n\n"
        "Explain to a fresher, in exactly three short sentences, what these top-attended tokens show. "
        "Use only the numbers above and do not invent new ones."
    )
    return [SystemMessage(SYSTEM), HumanMessage(question)]


def explain(facts: dict, llm=None) -> str:
    if llm is None:
        llm = get_llm()
    reply = llm.invoke(build_messages(facts))
    return reply.text   # .text, not .content: Gemini 3 returns content as a list of blocks


def main(llm=None, facts=None) -> int:
    if facts is None:
        facts = gather_facts()
    messages = build_messages(facts)
    print("Real numbers from your computer, sent to the model as this prompt:\n")
    print(messages[1].content + "\n")

    if llm is None:
        try:
            llm = get_llm()
        except (MissingAPIKeyError, ValueError) as error:
            print(str(error).strip())   # friendly help (no key, or a typo in LLM_PROVIDER), not a traceback
            return 1
    model_name = getattr(llm, "model", None) or type(llm).__name__
    print(f"{model_name} explains:\n")
    print(explain(facts, llm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
