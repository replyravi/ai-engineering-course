# Your Turn: one batch, two lengths, zero leaks

In step 4 the tokenizer padded `Chai, please!` with four `[PAD]` tokens so it could sit in one batch next to
`Filter coffee here is the best on campus.`, and it gave us an attention mask: `1 1 1 1 1 1 1 0 0 0 0`.
In steps 9 to 11 our NumPy attention read one sentence at a time.

**The task:** run both sentences through `multi_head_attention()` from `attention.py` as **one batch**, and prove
three things:

1. The padded positions get **exactly zero** attention weight. Not "small": `== 0.0`.
2. Every row of weights still sums to 1.
3. The short sentence gives the **same output** in the batch as when it runs alone, without padding.
   (So padding changes nothing for the real tokens. That is the whole point of the mask.)

Post your result in the comments, for example: `padded weights exactly zero: True, same as alone: True (max difference 7e-16)`.

## Steps

1. Add a small function `padding_mask(attention_mask)` to `attention.py`. It takes the tokenizer's
   `attention_mask` (shape `(batch, tokens)`, ones and zeros) and returns a mask that `scaled_dot_product_attention`
   understands. Think about the shape: the score grid of each sentence is `(queries, keys)`, and the padding
   is a property of the **keys** (columns), the same for every query.
2. Create `challenge.py`. Tokenize the two `[short]` lines from `data/sentences.txt` with `padding=True`.
3. Turn the token IDs into vectors. Use DistilBERT's own input vectors (token row + position), exactly what
   its first attention block receives: `model.embeddings(batch["input_ids"])`. Shape `(2, 11, 768)`.
4. Make one set of weights with `random_block_params(768, 3072, rng)` and run `multi_head_attention(...)`
   with `num_heads=12` and your mask.
5. Check the three things above with `==`, `np.allclose` and `print`.
6. Add a test to `tests/test_attention.py` that checks your `padding_mask` on a tiny hand-made batch, so it runs offline.

<details>
<summary>Hint (the skeleton, not the full answer)</summary>

```python
import numpy as np
import torch

from attention import multi_head_attention, padding_mask, random_block_params
from encoder import load_encoder, read_sentences

tokenizer, model = load_encoder()
lines = read_sentences()["short"]
batch = tokenizer(lines, padding=True, return_tensors="pt")
with torch.inference_mode():
    X = model.embeddings(batch["input_ids"]).numpy()          # (2, 11, 768)

params = random_block_params(768, 3072, np.random.default_rng(42))
W = [params[name] for name in ("Wq", "Wk", "Wv", "Wo")]
out, weights = multi_head_attention(X, *W, num_heads=12, mask=padding_mask(batch["attention_mask"]))

print(weights.shape)                          # (2, 12, 11, 11): sentence, head, query, key
print((weights[1, :, :, 7:] == 0.0).all())    # sentence 2, every head, every query, the 4 [PAD] keys
# now run lines[1] alone (no padding) and compare out[1, :7] with its output
```

</details>

## Think about it

- `padding_mask` needs one new axis. Which one, and why does NumPy then "broadcast" it over every query and every head?
- The `[PAD]` **rows** (queries 7 to 10 of sentence 2) still get an output. Is that a problem? Who reads those rows later?
- Episode 03 adds a second kind of mask, the causal mask, to the same `mask=` argument. What would it look like for 4 tokens?

## Stretch goals

- **Check the real model:** run DistilBERT on the same padded batch with `output_attentions=True` (step 12 shows how).
  Are its weights on the `[PAD]` keys exactly zero too, in all 6 layers and 12 heads?
- **Left or right?** Tokenizers can pad on the left (`tokenizer.padding_side = "left"`). Does your mask still work?
- **Break it on purpose:** pass `mask=None` and measure how much the short sentence's output changes when padding leaks in.
