"""
gemini_limit.py - hit the output limit on purpose and read the finish reason (step 13).

Our small decoder had two stop rules: the end-of-text token, or our max-tokens limit (step 6).
Cloud models have the same two. Here we ask for a long answer but allow only 20 output tokens,
so the answer is cut off and the finish reason says why.

Run:  python gemini_limit.py      (needs your free Gemini key in the .env file at the repo root)
"""

from langchain_core.messages import HumanMessage, SystemMessage

from gemini_stream import finish_reason, hit_limit, token_line
from llm import MissingAPIKeyError, get_llm, get_model_name, get_provider

LIMIT = 20
QUESTION = "In about 150 words, explain how a large language model writes an answer, step by step."


def limit_settings(provider: str, limit: int = LIMIT) -> dict:
    """The output-limit setting has a different name for local Ollama models."""
    if provider == "ollama":
        return {"num_predict": limit}
    return {"max_tokens": limit}       # Gemini (max_output_tokens), OpenAI and Claude all accept max_tokens


def main(llm=None) -> int:
    try:
        provider = get_provider()
        print(f"Provider: {provider}   Model: {get_model_name(provider)}")
        if llm is None:
            llm = get_llm(**limit_settings(provider))
    except (MissingAPIKeyError, ValueError) as error:
        print(f"\n{str(error).strip()}")   # friendly help (no key, or a typo in LLM_PROVIDER), not a traceback
        return 1

    print(f"Output limit: {LIMIT} tokens\n")
    print(f"You asked: {QUESTION}\n")
    reply = llm.invoke([SystemMessage("You are a patient teacher."), HumanMessage(QUESTION)])
    print(f"AI replied:\n{reply.text or '(no visible text)'}\n")
    print(token_line(reply))
    print(f"Finish reason: {finish_reason(reply)}")
    if hit_limit(reply):
        print("The limit fired: the model stopped because we said so, not because it was done.")
        if not reply.text:
            print("No visible text at all. Look at the thinking count above: thinking tokens are output tokens too.")
    else:
        print("The model finished before the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
