"""
tools/make_food_safety_pdf.py - writes handbook/food_safety.pdf (a 3-page PDF).

You do NOT need to run this: the PDF is already in the handbook folder.
It is here so you can see how the PDF was made, and make your own PDFs
for the challenge. It uses matplotlib, which is already in requirements.txt.

Run (from the episode folder):  python tools/make_food_safety_pdf.py

Author: Ravi Shanker Singh (TechWithRavi). The company and every rule in it are invented.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # no window, just files
matplotlib.rcParams["pdf.fonttype"] = 42    # real text inside the PDF, so a loader can read it
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "handbook" / "food_safety.pdf"
TITLE = "Dabbaloop Kitchen Food Safety and Allergen Guide"

# Each page is a list of ("h", heading) or ("p", paragraph) or ("b", bullet) items.
PAGES = [
    [
        ("title", TITLE),
        ("p", "Version 2.0, October 2026. Issued by the Dabbaloop kitchen team. "
              "Dabbaloop is an invented company used for teaching; every rule here is made up."),
        ("h", "1. Our kitchen"),
        ("p", "All Dabbaloop meals are cooked in one central kitchen in Kothrud, Pune. Lunch cooking "
              "starts at 7:30 am and every lunch box is packed and sealed by 11:15 am. Dinner cooking "
              "starts at 4:30 pm, after the dinner cut-off for changes has passed."),
        ("h", "2. Hygiene rules we follow every day"),
        ("b", "Every cook and packer wears a hair cap, gloves and an apron, and washes hands every "
              "30 minutes and after handling raw vegetables."),
        ("b", "Fridge and freezer temperatures are checked and written in a log three times a day. "
              "Fridges must stay at 5 degrees C or colder."),
        ("b", "Drinking and cooking water is RO purified and tested every month by an outside lab."),
        ("b", "Frying oil is used for one day only and then thrown away. We never top up old oil."),
        ("b", "Every steel meal box is washed in water at 80 degrees C and air dried before reuse."),
        ("b", "The whole kitchen is deep cleaned every Sunday, which is why there are no Sunday deliveries."),
    ],
    [
        ("h", "3. Allergens in our food"),
        ("p", "Our menu labels show five allergens. Milk: paneer, curd, ghee and butter. Gluten: wheat "
              "phulkas and some sabzis thickened with wheat flour. Peanuts: sabudana, poha and several "
              "Maharashtrian sabzis. Sesame: til chutney and some gravies. Tree nuts: the cashew gravy "
              "that is usually served on Fridays."),
        ("p", "Our kitchen is not nut-free. Peanuts and cashews are used in the same kitchen as every "
              "other dish, so we cannot promise that any meal is free from traces of nuts. If a nut "
              "allergy is severe, please do not rely on our meals."),
        ("p", "Jain thalis are cooked in separate vessels marked with a green band, but in the same "
              "kitchen and by the same team."),
        ("h", "4. Allergy notes and red stickers"),
        ("p", "If you add an allergy note under Profile, then Food preferences, the packer checks your "
              "meal against the menu of the day and puts a red sticker on the bag. The red sticker means "
              "we have read your note. It is not a medical guarantee."),
        ("h", "5. Labels on the lid"),
        ("p", "Every meal box lid has a printed label with the date, the meal (lunch or dinner), the thali "
              "name and the allergens in that day's menu."),
    ],
    [
        ("h", "6. Storing and reheating at home"),
        ("b", "Eat your meal within 3 hours of delivery. If you will eat later, move the food to the "
              "fridge within 1 hour of delivery and finish it the same day."),
        ("b", "Never put the steel box in a microwave. Move the food to a microwave-safe bowl and heat it "
              "until it is steaming hot all the way through."),
        ("b", "Curd and raita should not stay out of the fridge for more than 2 hours, and for less than "
              "1 hour on hot summer days."),
        ("b", "Do not keep leftovers for the next day's lunch."),
        ("h", "7. If something seems wrong"),
        ("p", "If the food smells sour, looks unusual or the seal on the lid is broken, do not eat it. "
              "Take a photo, keep the food and report it in the app within 2 hours of delivery. The kitchen "
              "team calls you back within 1 hour, and the meal is credited in full."),
        ("h", "8. Kitchen visits"),
        ("p", "Customers can visit the kitchen on the first Saturday of every month between 3:00 pm and "
              "5:00 pm. Book a visit in the app. Each group can have up to 6 people, and children under "
              "12 must come with an adult."),
    ],
]

# Layout on an A4 page (sizes in inches and points)
PAGE_W, PAGE_H = 8.27, 11.69
LEFT, TOP = 0.09, 0.93          # as fractions of the page
WRAP = 88                       # characters per line at 10 pt


def draw_page(fig, items, page_no, total):
    y = TOP
    for kind, text in items:
        if kind == "title":
            fig.text(LEFT, y, text, fontsize=16, weight="bold")
            y -= 0.035
            continue
        if kind == "h":
            y -= 0.012
            fig.text(LEFT, y, text, fontsize=12.5, weight="bold")
            y -= 0.026
            continue
        prefix = "- " if kind == "b" else ""
        lines = textwrap.wrap(text, width=WRAP - len(prefix))
        for i, line in enumerate(lines):
            lead = prefix if i == 0 else " " * len(prefix)
            fig.text(LEFT, y, lead + line, fontsize=10)
            y -= 0.018
        y -= 0.008
    fig.text(LEFT, 0.04, f"{TITLE} - page {page_no} of {total}", fontsize=8, color="0.4")


def main():
    OUT.parent.mkdir(exist_ok=True)
    metadata = {"Title": TITLE, "Author": "Dabbaloop kitchen team (invented)", "CreationDate": None}
    with PdfPages(OUT, metadata=metadata) as pdf:
        for number, items in enumerate(PAGES, start=1):
            fig = Figure(figsize=(PAGE_W, PAGE_H))
            draw_page(fig, items, number, len(PAGES))
            pdf.savefig(fig)
    print(f"Wrote {OUT.name} with {len(PAGES)} pages")


if __name__ == "__main__":
    main()
