"""
old_tasks.py - why tutorials from 2024 and 2025 crash on transformers v5 (step 3).

Older tutorials wrote pipeline("summarization"), pipeline("translation", ...) and
pipeline("visual-question-answering"). transformers v5 removed those task names.
A small chat model through "text-generation" now does summaries and translations, and a
vision-language model through "image-text-to-text" answers questions about pictures.

We call pipeline(task) WITHOUT a model on purpose: the task name is checked first,
so nothing is downloaded. (With model="..." the library would fetch the model's config first.)

Run:  python old_tasks.py            (add --list to see every task name your version knows)
"""

import sys

import common  # noqa: F401  (reads the course .env first)

OLD_TO_NEW = {
    "summarization": "text-generation + a small chat model",
    "translation": "text-generation + a small chat model",
    "visual-question-answering": "image-text-to-text + a vision-language model",
}


def try_old_task(task: str) -> tuple[bool, str]:
    """(still_works, message). Never downloads anything: the removed names fail before any model is fetched."""
    import transformers
    from transformers import pipeline

    if int(transformers.__version__.split(".")[0]) < 5:
        # Version 4 still knows these names, and pipeline(task) would download a default model of about 1 GB.
        return True, "still works in this version"
    try:
        pipeline(task)
    except KeyError as error:
        return False, str(error.args[0])
    return True, "still works in this version"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    import transformers
    from transformers.pipelines import get_supported_tasks

    print(f"transformers {transformers.__version__}\n")
    print(f"{'old task name':<28}{'in this version':<22}use instead")
    messages = {}
    for task, replacement in OLD_TO_NEW.items():
        works, message = try_old_task(task)
        messages[task] = message
        status = "works" if works else "removed (KeyError)"
        print(f"{task:<28}{status:<22}{replacement}")

    print('\nThe library\'s own message for "summarization":')
    print(f"  {messages['summarization']}")
    print("\nAlso renamed in v5: torch_dtype=... is now dtype=... (the old name still works, with a warning).")

    tasks = get_supported_tasks()
    print(f"\nThis version knows {len(tasks)} task names, including text-generation and image-text-to-text.")
    if "--list" in argv:
        print("  " + "\n  ".join(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
