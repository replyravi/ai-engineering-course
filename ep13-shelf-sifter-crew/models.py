"""
models.py - Step 10: the SHAPES of our data, written as Pydantic classes.

A shape is a promise: "a Pick always has a title, a price, a rating, a url and a reason".
CrewAI uses Picks to turn the Analyst's answer into a real Python object (output_pydantic),
and Pydantic refuses anything that breaks the promise, for example a rating of 7.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

from pydantic import BaseModel, Field


class Book(BaseModel):
    """One book as the Scout scraped it from the shelf page."""

    title: str
    price: float = Field(description="price in pounds (GBP), as shown on the practice site")
    rating: int = Field(ge=1, le=5, description="star rating, 1 to 5")
    stock: str = Field(description="availability text, e.g. 'In stock'")
    url: str = Field(description="the book's own page on the practice site")


class Pick(BaseModel):
    """One book the Analyst recommends."""

    title: str = Field(description="the exact title from the Scout's list")
    price: float = Field(description="the exact price from the Scout's list")
    rating: int = Field(ge=1, le=5, description="the exact star rating from the Scout's list")
    url: str = Field(description="the exact url from the Scout's list")
    reason: str = Field(description="one short sentence: why this book is good value for this reader")


class Picks(BaseModel):
    """The Analyst's typed answer: up to three best-value books for one reader."""

    category: str = Field(description="the shelf, e.g. travel")
    interest: str = Field(description="what the reader likes, copied from the task")
    picks: list[Pick] = Field(min_length=1, max_length=3, description="best value first")
