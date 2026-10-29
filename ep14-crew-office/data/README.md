# data/ - invented sample data, not real companies or prices

`python make_prices.py` (Step 9) writes two files here:

- `prices.csv` - 250 weekdays of closing prices for three invented tickers, DEMO1, DEMO2 and DEMO3,
  made by a seeded random generator. Same seed, same numbers, on every computer.
- `news.csv` - a few invented headlines on the days the generator added a big move.

None of these companies, prices or headlines are real. They exist so we can practise reading
numbers safely. Nothing here is investment advice. Both files are git-ignored: you make them yourself.
