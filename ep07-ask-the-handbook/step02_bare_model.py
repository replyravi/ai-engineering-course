"""
step02_bare_model.py - Step 2: ask the model about Dabbaloop WITHOUT the handbook.

Dabbaloop is a company we invented, so no model has ever read its rules.
Watch what the model does when it cannot know the answer.

    python step02_bare_model.py

Author: Ravi Shanker Singh (TechWithRavi)
"""

from llm import MissingAPIKeyError, get_llm

QUESTION = "Dabbaloop is a meal-box delivery service in Pune. What is its refund window for a missed lunch?"


def main():
    try:
        llm = get_llm()
    except MissingAPIKeyError as error:
        print(str(error).strip())
        raise SystemExit(1)
    print(f"Question: {QUESTION}\n")
    print(llm.invoke(QUESTION).text)   # .text, not .content (Gemini 3+ returns a list of blocks)


if __name__ == "__main__":
    main()
