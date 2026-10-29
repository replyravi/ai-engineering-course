"""
tests/test_tokens.py - the real DistilBERT tokenizer (steps 1 to 4).

These run offline once the tokenizer is downloaded, and SKIP with the fix if it is not.
"""

import tokens
from encoder import read_sentences
from model_fixtures import tiktoken_ready, tokenizer  # noqa: F401


def test_cls_first_sep_last(tokenizer):
    ids = tokenizer("Filter coffee here is the best on campus.")["input_ids"]
    assert ids[0] == 101 and ids[-1] == 102


def test_vocabulary_and_special_ids(tokenizer):
    assert len(tokenizer) == 30522
    special = {t: tokenizer.convert_tokens_to_ids(t) for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]}
    assert special == {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}


def test_unknown_words_are_cut_into_pieces_not_borrowed(tokenizer):
    splits = dict(tokens.split_words("The biryani was far too oily.", tokenizer))
    assert splits["biryani"] == ["bi", "##rya", "##ni"]
    assert splits["oily"] == ["oil", "##y"]


def test_unk_only_as_a_last_resort(tokenizer):
    assert tokenizer.tokenize("🍛 biryani") == ["[UNK]", "bi", "##rya", "##ni"]


def test_padding_gives_equal_rows_and_a_mask(tokenizer):
    batch = tokenizer(read_sentences()["short"], padding=True)
    first, second = batch["input_ids"]
    assert len(first) == len(second) == 11
    assert batch["attention_mask"][1] == [1] * 7 + [0] * 4
    assert second[7:] == [0, 0, 0, 0]                 # [PAD] has ID 0


def test_token_counts_english_vs_hindi(tokenizer, tiktoken_ready):
    english, hindi = read_sentences()["english_hindi"]
    assert tokens.count_tokens(english, tokenizer) == {
        "WordPiece (DistilBERT, BERT)": 9, "cl100k_base (gpt-4)": 11, "o200k_base (gpt-4o)": 10}
    assert tokens.count_tokens(hindi, tokenizer) == {
        "WordPiece (DistilBERT, BERT)": 16, "cl100k_base (gpt-4)": 33, "o200k_base (gpt-4o)": 11}


def test_steps_print(tokenizer, capsys):
    for step in ["pieces", "ids", "pad"]:
        tokens.STEPS[step](tokenizer)
    out = capsys.readouterr().out
    assert "takeaway = take + ##away" in out
    assert "Vocabulary size: 30,522" in out
    assert "Both rows now have 11 tokens" in out
