"""
tests/test_models_report.py - the licence and size report, built from SAVED cards (step 2).

tests/sample_model_cards.json is a frozen copy of what models.py saved on 2026-10-26,
so these tests run offline and do not change when the Hub changes.
"""

import json
import shutil
from types import SimpleNamespace

import pytest

import models
from lab_fixtures import SAMPLE_CARDS_FILE


@pytest.fixture
def cards():
    return models.load_cards(SAMPLE_CARDS_FILE)


@pytest.mark.parametrize("licence, name, expected", [
    ("apache-2.0", None, "yes (keep the notice)"),
    ("mit", None, "yes (keep the notice)"),
    ("cc-by-nc-4.0", None, "NO: non-commercial"),
    ("other", "flux-1-dev-non-commercial-license", "NO: non-commercial"),
    ("other", "qwen-research", "read it (qwen-research)"),
    (None, None, "read it (unknown)"),
])
def test_business_use_labels(licence, name, expected):
    assert models.business_use(licence, name) == expected


def test_report_lists_our_three_models_first_with_their_licences(cards):
    text = models.report(cards)
    ours, others = text.split("Models from older tutorials")

    for name in ("Qwen2.5-0.5B-Instruct", "SmolVLM-256M-Instruct", "all-MiniLM-L6-v2"):
        assert name in ours
    assert ours.count("apache-2.0") == 3
    assert "nllb-200-distilled-600M" in others


def test_non_commercial_and_gated_models_are_labelled(cards):
    lines = models.report(cards).splitlines()

    nllb = next(line for line in lines if "nllb-200" in line)
    flux = next(line for line in lines if "FLUX.1-dev" in line)
    assert "NO: non-commercial" in nllb
    assert "NO: non-commercial" in flux and "YES" in flux          # gated
    assert any(line.startswith("Gated =") for line in lines)


def test_sizes_are_readable(cards):
    text = models.report(cards, cache={"Qwen/Qwen2.5-0.5B-Instruct": 999_600_000})

    assert "988 MB" in text          # the Qwen weights file
    assert "1.0 GB" in text          # what it takes in the cache folder
    assert models.size_text(None) == "-" and models.params_text(494_032_768) == "494.0 M"


def test_weights_bytes_prefers_safetensors_and_ignores_subfolders():
    files = [SimpleNamespace(rfilename="model.safetensors", size=100),
             SimpleNamespace(rfilename="pytorch_model.bin", size=100),
             SimpleNamespace(rfilename="onnx/model.onnx", size=999),
             SimpleNamespace(rfilename="transformer/part-1.safetensors", size=999)]

    assert models.weights_bytes(files) == 100
    assert models.weights_bytes([SimpleNamespace(rfilename="pytorch_model.bin", size=42)]) == 42


def test_saved_report_needs_no_internet(monkeypatch, capsys):
    monkeypatch.setattr(models, "CARDS_FILE", SAMPLE_CARDS_FILE)

    assert models.main(["--saved"]) == 0
    assert "From the saved" in capsys.readouterr().out


def test_without_the_hub_it_falls_back_to_the_saved_file(monkeypatch, capsys, tmp_path):
    saved = tmp_path / "model_cards.json"
    shutil.copy(SAMPLE_CARDS_FILE, saved)
    monkeypatch.setattr(models, "CARDS_FILE", saved)

    assert models.main([]) == 0        # the Hub cannot be reached in tests
    out = capsys.readouterr().out
    assert "Could not reach the Hub" in out and "Qwen2.5-0.5B-Instruct" in out
    assert json.loads(saved.read_text(encoding="utf-8")) == json.loads(SAMPLE_CARDS_FILE.read_text(encoding="utf-8"))
