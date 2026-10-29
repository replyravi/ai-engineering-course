"""
tests/test_cloud_steps.py - steps 6 and 9 (Gemini) with a FAKE model: no key, no internet.

gemini_style_reply(...) answers the way Gemini 3 does (content is a list of blocks),
so these tests also prove we read replies with .text.
The photo tests use the sample drawings, drawn fresh in a temporary folder (tmp_path),
so your own photos in photos/ never make them fail.
"""

import base64
import io
import json

from langchain_core.messages import SystemMessage
from PIL import Image

import caption
import decoder_cloud
from common import OWNER_SYSTEM, find_photos
from testing.fakes import fake_llm, gemini_style_reply
from zoo_fixtures import draw_sample_photos

KEY_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL"]
LOCAL = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "parameters": 494032768, "seconds": 0.8,
         "reply": "Sorry for the delay in your order."}


def no_keys(monkeypatch):
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def decode(block) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(block["base64"])))


# --- step 6: decoder_cloud.py -------------------------------------------------

def test_ask_reads_gemini_style_replies_as_plain_text():
    llm = fake_llm(gemini_style_reply("Sorry the coffee was lukewarm. We will wipe every cup."))
    assert decoder_cloud.ask(llm) == "Sorry the coffee was lukewarm. We will wipe every cup."
    first_message = llm.received[0][0]
    assert isinstance(first_message, SystemMessage) and first_message.content == OWNER_SYSTEM


def test_the_rough_problem_check():
    assert decoder_cloud.mentions_problem("Sorry the coffee was lukewarm.") == "yes (coffee, lukewarm)"
    assert decoder_cloud.mentions_problem("Sorry for the delay in your order.") == "no"


def test_comparison_rows_put_both_models_side_by_side():
    cloud = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "seconds": 1.2, "reply": "Sorry, cold coffee."}
    rows = {row[0]: row[1:] for row in decoder_cloud.comparison_rows(LOCAL, cloud)}
    assert rows["Parameters"] == ["494,032,768", "not published"]
    assert rows["Runs on"] == ["this computer", "Google's servers"]
    assert rows["Names the real problem?"] == ["no", "yes (coffee)"]


def test_comparison_rows_without_step_5():
    cloud = {"provider": "ollama", "model": "qwen3.5:4b", "seconds": 3.0, "reply": "Sorry!"}
    rows = {row[0]: row[1:] for row in decoder_cloud.comparison_rows(None, cloud)}
    assert rows["Model"] == ["run decoder_local.py first", "qwen3.5:4b"]
    assert rows["Runs on"][1] == "this computer (Ollama)"


def test_decoder_cloud_main_with_a_fake_model(capsys, tmp_path, monkeypatch):
    local_file = tmp_path / "local_decoder.json"
    local_file.write_text(json.dumps(LOCAL), encoding="utf-8")
    monkeypatch.setattr(decoder_cloud, "LOCAL_RESULT_FILE", local_file)
    llm = fake_llm(gemini_style_reply("Sorry your coffee was lukewarm. We now wipe every cup."))
    assert decoder_cloud.main(llm=llm) == 0
    out = capsys.readouterr().out
    assert "Same prompt, two models" in out
    assert "Sorry your coffee was lukewarm." in out


def test_decoder_cloud_without_a_key_prints_friendly_help(capsys, monkeypatch):
    no_keys(monkeypatch)
    assert decoder_cloud.main() == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out


# --- step 9: caption.py -------------------------------------------------------

def test_image_block_is_a_small_jpeg(tmp_path):
    block = caption.image_block(draw_sample_photos(tmp_path) / "cup.jpg")
    assert block["type"] == "image" and block["mime_type"] == "image/jpeg"
    assert base64.b64decode(block["base64"])[:2] == b"\xff\xd8"   # every JPEG file starts with these bytes
    assert max(decode(block).size) <= caption.MAX_SIDE


def test_a_big_photo_is_shrunk_and_keeps_its_shape(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "orange").save(path)
    assert decode(caption.image_block(path)).size == (1024, 683)


def test_a_phone_photo_is_turned_the_right_way_up(tmp_path):
    path = tmp_path / "phone.jpg"
    image = Image.new("RGB", (400, 200), "green")
    exif = image.getexif()
    exif[0x0112] = 6                    # "rotate 90 degrees when you show me", as phones store it
    image.save(path, exif=exif)
    assert decode(caption.image_block(path)).size == (200, 400)


def test_one_message_carries_text_and_the_photo(tmp_path):
    message = caption.build_message(draw_sample_photos(tmp_path) / "plant.jpg")
    assert [block["type"] for block in message.content] == ["text", "image"]


def test_caption_returns_the_models_sentence(tmp_path):
    llm = fake_llm(gemini_style_reply("A green plant in a clay pot on a wooden table."))
    plant = draw_sample_photos(tmp_path) / "plant.jpg"
    assert caption.caption(plant, llm) == "A green plant in a clay pot on a wooden table."
    sent = llm.received[0][0].content
    assert sent[1]["type"] == "image"


def test_caption_main_with_a_fake_model(capsys, tmp_path, monkeypatch):
    photos = draw_sample_photos(tmp_path / "photos")
    monkeypatch.setattr(caption, "HERE", tmp_path)                        # tmp_path plays the episode folder
    monkeypatch.setattr(caption, "find_photos", lambda: find_photos(photos))
    llm = fake_llm(gemini_style_reply("A white cup of tea on a saucer."))
    assert caption.main([], llm=llm) == 0
    out = capsys.readouterr().out
    assert "Photo: photos/book.jpg" in out           # the first photo by name
    assert "A white cup of tea on a saucer." in out


def test_caption_main_with_a_missing_photo(capsys):
    assert caption.main(["no_such_photo.jpg"], llm=fake_llm("unused")) == 1
    assert "No photo found" in capsys.readouterr().out


def test_caption_without_a_key_prints_friendly_help(capsys, monkeypatch, tmp_path):
    no_keys(monkeypatch)
    assert caption.main([str(draw_sample_photos(tmp_path) / "cup.jpg")]) == 1
    assert "https://aistudio.google.com/apikey" in capsys.readouterr().out


def test_a_connection_error_gives_a_hint_not_a_traceback(capsys, monkeypatch):
    import httpx

    class Unreachable:
        def invoke(self, messages):
            raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert decoder_cloud.main(llm=Unreachable()) == 1
    out = capsys.readouterr().out
    assert "Could not reach the model: [Errno 61] Connection refused" in out
    assert "ollama pull qwen3.5:4b" in out
