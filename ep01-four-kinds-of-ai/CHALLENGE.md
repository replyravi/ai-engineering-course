# Your Turn: let a big model do the labelling

In step 9 a person labelled all 30 training rows, and the trained model scored **5 of 10** on the test rows.
Labelling by hand is slow and boring. What if a big AI model does the labelling for you?

**The task:** keep the human labels on only the **10 test rows** (they stay our answer key).
Let **Gemini label the other 30 rows** with structured output, train approach B on Gemini's labels,
and score it on the same 10 test rows. Is it better or worse than step 9's 5 of 10?

This is not semi-supervised learning yet: every training label comes from Gemini, none from a person.
The semi-supervised version (a person labels a few training rows, a model labels the rest) is the first stretch goal.

Post your number in the comments, for example: `B on Gemini labels: 6/10 (Gemini agreed with my labels on 27 of 30 rows)`.

## Steps

1. Create a new file `f_gemini_labels.py` in this folder.
2. Load and split the data exactly as before: `train, test = load_and_split()` from `common.py`.
3. Pretend the 30 training rows have no labels. Ask Gemini to label them with `classify(train["text"])`
   from `d_genai.py`. That is one short call per row, so only 30 small calls, but run it once, not in a loop.
   Your free-tier limits are on Google's rate-limit page: https://aistudio.google.com/rate-limit
4. Count how often Gemini agrees with the human labels (this is a fun extra number).
5. Train approach B on the Gemini labels: `train_model(train["clean"], gemini_labels)` from `b_supervised.py`.
6. Predict the 10 test rows and print the accuracy next to step 9's result.

<details>
<summary>Hint (the skeleton, not the full answer)</summary>

```python
from b_supervised import train_model
from common import load_and_split
from d_genai import classify

train, test = load_and_split()

results = classify(train["text"])                     # [(FeedbackLabel, ms), ...]
gemini_labels = [answer.label for answer, _ in results]

agree = ...                                           # how many match train["label"]?
model = train_model(train["clean"], gemini_labels)
right = ...                                           # compare model.predict(test["clean"]) with test["label"]
print(f"B on Gemini labels: {right}/10 (Gemini agreed with my labels on {agree} of 30 rows)")
```

</details>

## Think about it

- Gemini labelled 30 rows in seconds. A person would take several minutes. When is that trade worth it?
- If Gemini disagrees with your label on a sarcastic row, who is right? Read the row again.
- This trick has a name in industry: a big model labels data so a small, fast, cheap model can be trained on it.

## Stretch goals

- **Semi-supervised, for real:** keep human labels on 10 of the 30 training rows, let Gemini label
  the other 20, train on the mix, and compare all three results.
- **More data beats clever tricks:** write 20 more feedback rows of your own in `data/feedback.csv`,
  run `b_supervised.py` again and watch the test accuracy. (Your split will change, so your numbers will
  not match the video any more. That is fine. Also change the `40` in
  `test_feedback_file_has_40_rows_and_3_labels` in `tests/test_pipeline.py` to your new row count.)
- **Write a test:** add a test to `tests/test_genai.py` that runs your script with
  `fake_llm(structured_reply(...))` replies, so it works offline like the others.
