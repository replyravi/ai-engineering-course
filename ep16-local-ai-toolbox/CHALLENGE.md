# Your Turn: give Tiffin Toolbox a fifth skill

Pick **one** of the two challenges. Both keep the rule of this episode: read the model card first,
measure, and compare local with cloud honestly.

Post your result in the comments, for example:
`listen: <words right> of <words said>, <seconds> s for a <length>-second voice note, whisper-tiny`

---

## Challenge A: `python app.py listen` (speech to text)

Record a short voice note on your phone (one or two sentences, your own voice only), and turn it into text
on your laptop with a small Whisper-style model.

### Steps

1. **Read the card first.** Add `("openai/whisper-tiny", "listen to a voice note")` to `OUR_MODELS` in
   `models.py` and run `python models.py`. On 2026-10-26 the Hub listed it as Apache-2.0 with a 151 MB weights file.
   Is that a licence you may build on?
2. Convert your recording to a 16 kHz mono WAV file (Whisper models expect 16,000 samples per second).
   Make a folder for it first (`mkdir voice`, the same command on Windows), then use the free tool ffmpeg:
   `ffmpeg -i my_note.m4a -ar 16000 -ac 1 voice/my_note.wav`
3. Create `skills/listen.py` with the same shape as the other skills: a `load_...()` function wrapped in
   `lazy_load(...)`, a `listen(path)` function that returns the text and the seconds, and a `main(argv)`.
   The pipeline task is `"automatic-speech-recognition"` (it is still in transformers v5: run
   `python old_tasks.py --list` to check).
4. Add `"listen"` to `COMMANDS` in `app.py`.
5. Write down what you said, then count how many words came back right. Try one sentence with a Hindi word in it.
6. Add a test in `tests/test_cli.py` that checks `python app.py listen my_note.wav` reaches your skill
   (replace `app.run` with a recorder, like the other CLI tests).

<details>
<summary>Hint (the skeleton, not the full answer)</summary>

```python
import wave

import numpy as np

def read_wav(path):
    """A 16-bit mono WAV file as numbers between -1 and 1, plus its sample rate."""
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768, rate

def load_listener():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device())

audio, rate = read_wav(path)
result = lazy_load("listen", load_listener)({"raw": audio, "sampling_rate": rate})
print(result["text"])
```

</details>

**Stretch:** add a `listen` column to `python compare.py` that sends the same WAV file to Gemini
(an audio content block). Which one gets your name right?

---

## Challenge B: Ollama as a third column

Ollama runs bigger open models on your laptop with one command. Put it next to the Hugging Face model and Gemini.

### Steps

1. Install Ollama from https://ollama.com/download and run `ollama pull qwen3.5:4b` (a few GB: check your free disk first).
2. Run the comparison with Ollama in the cloud column, no code change needed:
   - macOS / Linux: `LLM_PROVIDER=ollama python compare.py translate`
   - Windows PowerShell: `$env:LLM_PROVIDER="ollama"; python compare.py translate`
   The table now says `your laptop (Ollama)` and "data left your laptop? no".
3. Now make it a real third column: in `compare.py`, build a second model with
   `init_chat_model("ollama:qwen3.5:4b")` (from `langchain.chat_models`) and add its row next to the local and Gemini rows.
4. Run all three on the Hindi translation. Does a 4-billion-parameter local model fix what the 0.5-billion one got wrong?
   How much slower is it, and how much memory does it take (`memory_check.py` shows the idea)?

<details>
<summary>Hint</summary>

`cloud_row(skill, llm, item, question)` already works with any LangChain chat model. You only need a second
`llm` and a second call. The engine label is the first field of `Row`: call it `"ollama"`.

</details>

## Think about it

- The 0.5B chat model writes Hindi letters that do not make sense. Is that a bug in our code, or a limit of the model?
  How would you find out before you build on a model? (Hint: the card's language list, and a two-minute test.)
- Your notes never left your laptop in the local columns. Which of your real notes would you never paste into a cloud model?
- A smaller model that is right 8 times out of 10 and runs offline, or a cloud model that is right 10 times out of 10?
  It depends on the job. Write down one job for each.

---

The challenge models are suggestions. Their sizes and licences were read from the Hugging Face Hub on 2026-10-26:
check the cards again before you download. The Challenge A hint was tried once on a computer-made voice note
(3.4 seconds, 16 kHz WAV): whisper-tiny wrote the sentence back word for word, and printed a few harmless
`[transformers]` notices about its settings. Challenge B was not run for this repo, so you are the first to measure it.
