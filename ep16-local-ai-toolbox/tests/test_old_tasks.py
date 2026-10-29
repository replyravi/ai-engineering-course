"""
tests/test_old_tasks.py - the task names that transformers v5 removed (step 3).

These run offline: a removed task name fails before anything is downloaded.
"""

import pytest

from old_tasks import OLD_TO_NEW, main, try_old_task


@pytest.mark.parametrize("task, new_task", [
    ("summarization", "text-generation"),
    ("translation", "text-generation"),
    ("visual-question-answering", "image-text-to-text"),
])
def test_old_task_names_are_removed_with_a_hint(task, new_task):
    works, message = try_old_task(task)

    assert works is False
    assert "has been removed" in message and new_task in message
    assert new_task in OLD_TO_NEW[task]


def test_the_report_prints_one_line_per_old_task(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert out.count("removed (KeyError)") == 3
    assert "torch_dtype=... is now dtype=..." in out
