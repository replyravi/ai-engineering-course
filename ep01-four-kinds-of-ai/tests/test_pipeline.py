"""
tests/test_pipeline.py - the local parts: data, cleaning, the split, classic ML,
clustering, the scoreboard, predict.py and check_setup.py.

Everything here runs offline. Clustering uses fake_embeddings(), so no model is downloaded.
"""

import csv
import shutil
import subprocess

import pytest

import check_setup
import compare
import predict
from b_supervised import top_words, train_model
from c_clusters import find_groups
from c_clusters import main as clusters_main
from common import LABELS, clean_text, load_and_split, load_feedback, split_feedback
from testing.fakes import fake_embeddings


# ---------------------------------------------------------------------------
# Data, cleaning and the split (steps 6 and 8)
# ---------------------------------------------------------------------------

def test_feedback_file_has_40_rows_and_3_labels():
    df = load_feedback()
    assert len(df) == 40
    assert sorted(df["label"].unique()) == sorted(LABELS)
    assert df["id"].is_unique
    assert (df["text"].str.len() > 0).all()
    assert set(df["note"]) <= {"", "sarcasm", "negation"}


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("Please DON'T stop the pav bhaji!!", "please do not stop the pav bhaji"),
        ("The coffee wasn't hot.", "the coffee was not hot"),
        ("I can't   find a seat", "i can not find a seat"),
        ("Open till 10 pm?", "open till 10 pm"),
    ],
)
def test_clean_text(raw, clean):
    assert clean_text(raw) == clean


def test_split_is_30_and_10_with_every_label_in_test():
    train, test = load_and_split()
    assert (len(train), len(test)) == (30, 10)
    assert set(train["id"]).isdisjoint(test["id"])
    assert set(test["label"]) == set(LABELS)


def test_split_is_identical_on_every_run():
    first = load_and_split()[1]["id"].tolist()
    second = load_and_split()[1]["id"].tolist()
    assert first == second


def test_a_different_seed_gives_a_different_split():
    df = load_feedback()
    assert split_feedback(df, seed=42)[1]["id"].tolist() != split_feedback(df, seed=7)[1]["id"].tolist()


# ---------------------------------------------------------------------------
# Approach B: TF-IDF + logistic regression (step 9), on a tiny fixture
# ---------------------------------------------------------------------------

TINY_TEXTS = [
    "food was cold again", "cold tea again today", "waited so long cold rice",
    "tasty food thank you", "friendly staff tasty dosa", "thank you clean tables",
    "please add more seats", "please open a second counter", "could you add juice",
]
TINY_LABELS = ["complaint"] * 3 + ["praise"] * 3 + ["suggestion"] * 3


def test_supervised_model_learns_from_a_tiny_fixture():
    model = train_model(TINY_TEXTS, TINY_LABELS)
    assert model.predict(["the rice was cold again"])[0] == "complaint"
    assert model.predict(["please add a salad"])[0] == "suggestion"
    assert model.predict(["thank you, very tasty"])[0] == "praise"


def test_top_words_gives_five_words_per_label():
    words = top_words(train_model(TINY_TEXTS, TINY_LABELS))
    assert sorted(words) == sorted(LABELS)
    assert all(len(w) == 5 for w in words.values())
    assert "cold" in words["complaint"]


def test_filler_words_are_dropped_but_not_and_please_stay():
    model = train_model(TINY_TEXTS + ["the chutney was not fresh"], TINY_LABELS + ["complaint"])
    vocabulary = set(model.named_steps["tfidf"].get_feature_names_out())
    assert "the" not in vocabulary and "was" not in vocabulary
    assert {"not", "please", "not fresh"} <= vocabulary


# ---------------------------------------------------------------------------
# Approach C: clustering (step 10), with fake embeddings
# ---------------------------------------------------------------------------

def test_kmeans_finds_three_groups():
    vectors = fake_embeddings().embed_documents([f"sentence {i}" for i in range(12)])
    assert len(set(find_groups(vectors).labels_)) == 3


def test_cluster_script_prints_groups_and_saves_the_picture(tmp_path, capsys):
    picture = tmp_path / "clusters.png"
    df = clusters_main(embeddings=fake_embeddings(), picture=picture)
    assert set(df["group"]) == {1, 2, 3}
    assert picture.exists() and picture.stat().st_size > 10_000
    output = capsys.readouterr().out
    assert "Group 1" in output and "peek at the hidden labels" in output


# ---------------------------------------------------------------------------
# The scoreboard (step 15)
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {"approach": "A Keyword rules", "ring": "AI (rules)", "right": 6, "ms": 0.01,
     "labels": "no", "training": "no", "internet": "no", "era": "1950s-80s"},
    {"approach": "D Gemini zero-shot", "ring": "GenAI", "right": None, "ms": None,
     "labels": "no", "training": "no", "internet": "yes", "era": "2020s"},
]


def test_scoreboard_format():
    table = compare.format_scoreboard(SAMPLE_ROWS, total=10).splitlines()
    for column in ["Approach", "Accuracy", "Labels", "Training", "Internet", "ms/row", "Idea from"]:
        assert column in table[0]
    assert set(table[1]) == {"-"}
    assert "6/10 (60%)" in table[2] and "0.010" in table[2]
    assert table[3].count("needs live run") == 2       # accuracy and ms/row


def test_scoreboard_has_four_rows_and_waits_for_gemini(tmp_path):
    train, test = load_and_split()
    rows = compare.build_rows(train, test, embeddings=fake_embeddings(), gemini_file=tmp_path / "none.csv")
    assert [r["approach"][0] for r in rows] == ["A", "B", "C", "D"]
    assert all(r["right"] is not None for r in rows[:3])
    assert rows[3]["right"] is None and rows[3]["ms"] is None


def test_saved_gemini_run_is_used_when_ids_match(tmp_path):
    _, test = load_and_split()
    saved = tmp_path / "gemini.csv"
    with open(saved, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label", "gemini", "reason", "ms"])
        for i, (_, row) in enumerate(test.iterrows()):
            guess = row["label"] if i < 8 else "praise" if row["label"] != "praise" else "complaint"
            writer.writerow([row["id"], row["text"], row["label"], guess, "a reason", "500"])
    assert compare.saved_gemini_score(test, saved) == (8, 500.0)


# ---------------------------------------------------------------------------
# predict.py: save, predict and log (step 14), into a temp folder
# ---------------------------------------------------------------------------

def test_predict_trains_once_then_loads_and_logs(tmp_path, capsys):
    model_file, log_file = tmp_path / "model.joblib", tmp_path / "logs" / "predictions.csv"

    assert predict.main(["the", "sambar", "was", "cold"], model_file, log_file) == 0
    first = capsys.readouterr().out
    assert "trained one and saved it" in first and "-> complaint" in first

    assert predict.main(["please add more chairs"], model_file, log_file) == 0
    assert "Loaded the saved model" in capsys.readouterr().out

    with open(log_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["text"] for r in rows] == ["the sambar was cold", "please add more chairs"]
    assert rows[0]["label"] == "complaint"
    assert 0 < float(rows[0]["confidence"]) <= 1


def test_predict_without_a_sentence_shows_usage(tmp_path, capsys):
    assert predict.main([], tmp_path / "m.joblib", tmp_path / "p.csv") == 2
    assert "Usage" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# check_setup.py (step 4)
# ---------------------------------------------------------------------------

def test_mask_hides_the_middle_of_a_key():
    assert check_setup.mask("dummy-key-for-tests-0123456789") == "dumm...6789"
    assert check_setup.mask("short") == "*****"


def test_env_ignored_check_reads_gitignore_in_a_zip_download(tmp_path):
    (tmp_path / ".gitignore").write_text("# secrets\n.env\n", encoding="utf-8")
    assert check_setup.ignored_by_git(tmp_path / ".env").startswith("yes")
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    assert check_setup.ignored_by_git(tmp_path / ".env").startswith("NO")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_env_ignored_check_asks_git_in_a_clone(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    assert check_setup.ignored_by_git(tmp_path / ".env") == "yes"


@pytest.fixture
def no_keys(monkeypatch):
    for name in ["LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(name, raising=False)


def test_check_setup_reports_a_missing_key(no_keys, capsys):
    assert check_setup.main() == 1
    output = capsys.readouterr().out
    assert "API key ................. MISSING" in output
    assert "https://aistudio.google.com/apikey" in output


def test_check_setup_never_prints_the_full_key(no_keys, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests-0123456789")
    check_setup.main()
    output = capsys.readouterr().out
    assert "GEMINI_API_KEY found (dumm...6789)" in output
    assert "dummy-key-for-tests-0123456789" not in output


def test_check_setup_reports_a_typo_in_the_provider(no_keys, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "gemni")
    assert check_setup.main() == 1
    assert "Provider ................ WRONG" in capsys.readouterr().out
