"""
tests/test_cli.py - the command line of app.py, the menu, and the arguments of every skill (step 12).

No model is loaded here: we replace the part that runs a skill with a recorder,
so these tests check only how the command line is understood.
"""

import pytest

import app
import compare
from skills import ask_photo, search, translate
from skills.ask_photo import answer_matches


def test_help_lists_the_four_commands(capsys):
    assert app.main(["--help"]) == 0
    out = capsys.readouterr().out
    for command in ("summarise", "translate", "ask-photo", "search"):
        assert command in out


def test_unknown_command_is_refused_with_the_help(capsys):
    assert app.main(["dance"]) == 2
    assert "Unknown command: dance" in capsys.readouterr().out


def test_each_command_hands_the_rest_of_the_line_to_its_skill(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "run", lambda command, args: calls.append((command, args)) or 0)

    app.main(["search", "train ticket", "-k", "2"])
    app.main(["ask-photo", "photos/desk.jpg", "What colour is the mug?"])

    assert calls == [("search", ["train ticket", "-k", "2"]),
                     ("ask-photo", ["photos/desk.jpg", "What colour is the mug?"])]


def test_run_finds_the_skill_module(monkeypatch):
    monkeypatch.setattr(search, "main", lambda args: f"search got {args}")

    assert app.run("search", ["--matrix"]) == "search got ['--matrix']"


def test_menu_runs_a_choice_and_quits():
    typed = iter(["4", "train ticket", "1", "", "q"])
    calls, shown = [], []

    app.menu(read=lambda prompt: next(typed), write=shown.append,
             runner=lambda command, args: calls.append((command, args)))

    assert calls == [("search", ["train ticket"]), ("summarise", [])]
    assert shown[-1] == "Bye!"


def test_menu_explains_a_wrong_choice_and_stops_at_end_of_input():
    typed = iter(["9"])

    def read(prompt):
        for answer in typed:
            return answer
        raise EOFError              # nothing left to type, like pressing Ctrl+D

    shown = []
    app.menu(read=read, write=shown.append, runner=lambda *args: None)

    assert "Please type 1, 2, 3, 4 or q." in shown
    assert shown[-1] == "Bye!"


def test_menu_arguments_for_each_skill():
    assert app.menu_args("search", "") == ["--matrix"]
    assert app.menu_args("translate", "Good morning") == ["--text", "Good morning"]
    assert app.menu_args("ask-photo", "Is there a plant?")[1] == "Is there a plant?"
    assert app.menu_args("summarise", "") == []


def test_skill_parsers_understand_their_options():
    args = search.build_parser().parse_args(["train ticket", "-k", "2"])
    assert (args.query, args.k, args.matrix) == ("train ticket", 2, False)

    args = translate.build_parser().parse_args(["--text", "Good morning", "--to", "Tamil"])
    assert (args.text, args.to) == ("Good morning", "Tamil")

    args = ask_photo.build_parser().parse_args(["photos/desk.jpg", "Q1?", "Q2?", "--truth", "5"])
    assert (args.photo, args.questions, args.truth) == ("photos/desk.jpg", ["Q1?", "Q2?"], "5")


def test_compare_parser_and_its_switches():
    args = compare.build_parser().parse_args(["ask-photo", "photos/desk.jpg", "How many?", "--truth", "5"])
    assert (args.skill, args.question, args.truth, args.local_only) == ("ask-photo", "How many?", "5", False)

    assert compare.build_parser().parse_args(["summarise", "--local-only"]).local_only is True
    with pytest.raises(SystemExit):        # you cannot skip both sides
        compare.build_parser().parse_args(["summarise", "--local-only", "--cloud-only"])


@pytest.mark.parametrize("answer, truth, expected", [
    ("There are 4 pencils on the desk.", "5", False),
    ("I can see five pencils.", "5", True),
    ("There are 5 pencils.", "five", True),
    ("The mug is red.", "red", True),
])
def test_answer_matches(answer, truth, expected):
    assert answer_matches(answer, truth) is expected
