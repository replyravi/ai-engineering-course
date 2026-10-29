"""
test_robots.py - permission and politeness, tested offline (Steps 2, 3 and 6).

robots.txt rules come from a saved file, and every "download" is a fake, so no request
ever leaves your computer.
"""

import pytest

from robots_check import SAMPLE_ROBOTS, USER_AGENT, is_allowed, load_robots, parse_robots, robots_url
from scraper import PoliteFetcher
from shelf_helpers import TRAVEL_URL, FakeGet, FakeResponse

ROBOTS = "https://books.toscrape.com/robots.txt"


def test_robots_txt_lives_at_the_root_of_the_site():
    assert robots_url(TRAVEL_URL) == ROBOTS


def test_saved_rules_allow_the_shelf_but_not_the_basket():
    rules = parse_robots(SAMPLE_ROBOTS.read_text(encoding="utf-8"))

    assert is_allowed(rules, TRAVEL_URL)
    assert not is_allowed(rules, "https://books.toscrape.com/basket/")
    assert not is_allowed(rules, "https://books.toscrape.com/accounts/login/")
    assert rules.crawl_delay(USER_AGENT) == 2


@pytest.mark.parametrize("status, allowed", [(404, True), (401, False), (403, False), (503, False)])
def test_what_each_robots_status_means(status, allowed):
    # 404 = no rules at all. 401/403 = no robots allowed. 5xx = the site has trouble: wait.
    rules, seen = load_robots(TRAVEL_URL, get=FakeGet({ROBOTS: FakeResponse(status)}))

    assert seen == status
    assert is_allowed(rules, TRAVEL_URL) is allowed


def test_fetcher_refuses_other_sites_without_sending_a_request():
    get = FakeGet({})
    fetcher = PoliteFetcher(get=get, sleep=lambda s: None, log=lambda m: None)

    with pytest.raises(ValueError, match="only visits books.toscrape.com"):
        fetcher.fetch("http://127.0.0.1:8000/admin")
    assert get.calls == []


def test_fetcher_obeys_a_disallow_rule():
    get = FakeGet({ROBOTS: FakeResponse(200, "User-agent: *\nDisallow: /catalogue/\n")})
    fetcher = PoliteFetcher(get=get, sleep=lambda s: None, log=lambda m: None)

    with pytest.raises(PermissionError, match="robots.txt does not allow"):
        fetcher.fetch(TRAVEL_URL)
    assert get.calls == [ROBOTS]          # it asked for permission and stopped there


def test_fetcher_waits_two_seconds_between_requests_and_says_who_it_is():
    get = FakeGet({TRAVEL_URL: FakeResponse(200, "<html>£12.50</html>")})
    naps = []
    fetcher = PoliteFetcher(get=get, sleep=naps.append, clock=lambda: 100.0, log=lambda m: None)

    fetcher.fetch(TRAVEL_URL)    # robots.txt (404 = no rules), then the page
    fetcher.fetch(TRAVEL_URL)    # robots.txt is remembered, so only the page

    assert get.calls == [ROBOTS, TRAVEL_URL, TRAVEL_URL]
    assert naps == [2.0, 2.0]    # the clock never moves in this test, so each wait is the full 2 s
    assert get.last_headers == {"User-Agent": USER_AGENT}


def test_fetcher_uses_a_longer_crawl_delay_when_the_site_asks():
    get = FakeGet({ROBOTS: FakeResponse(200, "User-agent: *\nCrawl-delay: 5\n"),
                   TRAVEL_URL: FakeResponse(200, "<html></html>")})
    naps = []
    fetcher = PoliteFetcher(get=get, sleep=naps.append, clock=lambda: 100.0, log=lambda m: None)

    fetcher.fetch(TRAVEL_URL)

    assert naps == [5.0]


def test_fetcher_reads_the_pound_sign_correctly():
    # The real server sends "text/html" with no charset, so requests guesses ISO-8859-1 ("Â£").
    get = FakeGet({TRAVEL_URL: FakeResponse(200, "<p class='price_color'>£12.50</p>", "text/html")})
    fetcher = PoliteFetcher(get=get, sleep=lambda s: None, log=lambda m: None)

    assert "£12.50" in fetcher.fetch(TRAVEL_URL)
