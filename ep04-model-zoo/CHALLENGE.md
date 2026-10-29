# Your Turn: add an ear to the zoo

Today's zoo reads text, writes text and looks at photos. It cannot listen yet.
Add an **audio foundation model**: record a ten-second voice note, turn it into text with a tiny
Whisper-style speech-to-text model on your own laptop, and give that model a card in the zoo.

Post your transcript line and your new report row in the comments, for example:
`whisper-tiny: "Two cutting chai and one bun maska, please." (1.8 s)`

## The task

1. **Record yourself** for about ten seconds: a tea-stall order, a reminder, anything. Record only your own voice.
2. **Save it as a WAV file** in this folder, for example `voice_note.wav`: **16000 Hz, mono, 16-bit**.
   Free ways: Audacity (**File > Export Audio**, choose WAV, sample rate 16000, one channel), or, if you have ffmpeg:
   `ffmpeg -i note.m4a -ar 16000 -ac 1 voice_note.wav`
3. **Write `speech_to_text.py`.** Load [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) with
   `pipeline("automatic-speech-recognition", ...)` on the CPU, transcribe your file, and print the text, the time
   it took and the parameter count (`count_parameters` from `common.py`).
4. **Read its model card** and add a fifth card to `model_cards.json`: name, id, maker, family `speech-to-text`,
   inputs, outputs, parameters (your own count), context, weights, licence, business use, download size, released,
   source and today's date as `checked`. Run `python model_cards_schema.py` until it says
   `5 cards in model_cards.json, all passed the schema check.`
5. **Add a job** to `JOBS` in `zoo_report.py`, like `("Turn a voice note into text", "openai/whisper-tiny", "speech_to_text.py")`,
   and run `python zoo_report.py`. You should see `6 jobs, 5 model cards.`
6. **Keep the tests green.** Run `python -m pytest`. The card and report tests in `tests/test_cards.py` read the
   counts from your files, so your new card is checked too (a missing `source` or a wrong `family` fails a test).

## Done when

- [ ] `python speech_to_text.py` prints a transcript of your voice note
- [ ] `python model_cards_schema.py` passes with 5 cards
- [ ] `python zoo_report.py` shows 6 jobs
- [ ] `python -m pytest` passes

<details>
<summary>Hint (a skeleton, not the full answer)</summary>

The pipeline can open `.mp3` or `.m4a` files only if the ffmpeg program is installed. To stay free of that, read
the WAV yourself with Python's built-in `wave` module and pass the numbers in a dictionary:

```python
import wave

import numpy as np
from transformers import pipeline


def read_wav(path):
    """The audio as numbers between -1 and 1, and its sample rate."""
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)   # 16-bit WAV
        if f.getnchannels() == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)                       # stereo -> mono
    return samples.astype(np.float32) / 32768, rate


asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
audio, rate = read_wav("voice_note.wav")
result = asr({"raw": audio, "sampling_rate": rate})
print(result["text"])
```

If the rate is not 16000, transformers needs the `torchaudio` package to resample it. Exporting at 16000 Hz is simpler.

</details>

## Think about it

- Whisper writes text. Does that make it a generative model? It writes words, but it is not inventing them.
  Which box would you tick for `generates` on its card, and why?
- Which family is it: encoder, decoder, or both? The model card tells you.
- whisper-tiny is the smallest size of its family. Try a bigger one later, and add its card too. What do you gain,
  and what does it cost in download size and time?

## Stretch

- Send the same voice note to Gemini (it accepts audio input: see the `inputs` on its card) and compare the two
  transcripts in one table, like `decoder_cloud.py` does for text. Remember that the free tier may use what you send
  to improve Google's products.
- Add a test that is skipped until whisper-tiny is downloaded, with `needs_model` from `tests/zoo_fixtures.py`.

No solutions are published yet. A worked solution will be added to the course repo after the episode is live,
so try it yourself first.
