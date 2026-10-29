"""
tools/ - the custom tools of our two crews. Each file is one tool, tested on its own first.

    tone.py         check_tone            Crew A (Reviewer)   Step 3
    files.py        read_lead_file        Crew A (Researcher) Step 4
                    read_data_file        Crew B (Analyst)    Step 12
    brief_check.py  check_against_brief   Crew A (Reviewer)   Step 6
    indicators.py   price_indicators      Crew B (Analyst)    Step 10
    backtest.py     paper_backtest        Crew B (Strategy)   Step 11
    calculator.py   calculator            Crew B (all three)  Step 12, the safe calculator from Episode 11

Run one tool by hand, with no agent and no model, from the episode folder:

    python -m tools.tone "Congratulations on your new outlet."
    python -m tools.indicators DEMO1

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""
