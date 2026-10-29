"""
tests/test_crawl_mock.py - crawl.py (Step 11) with a MOCKED Apify client and a fake robots.txt.

No internet and no Apify account: the fake client returns a real, typed apify-client 3 Run
object (built from a dict, like the API would send) and three dataset items.
"""

import json

import pytest
from apify_client._models import Run          # the typed object .call() returns in apify-client 3
from mandi_fixtures import no_keys  # noqa: F401  (pytest fixture)

import crawl

START = "https://example.org/rule-book/"

ITEMS = [
    {"url": START, "metadata": {"title": "Rule book"}, "text": "How to read the board.", "markdown": "# ..."},
    {"url": START + "fees.html", "metadata": {"title": "Buyer charges"}, "text": "Commission is 4 percent."},
    {"url": START + "empty.html", "metadata": {"title": "Empty"}, "text": "   "},          # skipped: no text
    {"url": START + "units.html", "metadata": {}, "text": "One onion bag holds 50 kg."},
]


def typed_run(status="SUCCEEDED") -> Run:
    return Run.model_validate({
        "id": "RUN123", "actId": "ACT", "userId": "USER", "startedAt": "2026-10-26T10:00:00Z", "status": status,
        "meta": {"origin": "API"}, "stats": {"computeUnits": 0.0123, "runTimeSecs": 41.5},
        "options": {"build": "latest", "timeoutSecs": 300, "memoryMbytes": 1024, "diskMbytes": 2048},
        "buildId": "B", "defaultKeyValueStoreId": "KV", "defaultDatasetId": "DS42", "defaultRequestQueueId": "RQ",
        "usageTotalUsd": 0.0031,
    })


class FakeApify:
    """Stands in for ApifyClient: records what our code asked for."""

    def __init__(self, token, run="default", items=ITEMS):
        self.token, self.items, self.calls = token, items, []
        self.run = typed_run() if run == "default" else run
        self.dataset_ids = []

    def actor(self, actor_id):
        self.actor_id = actor_id
        return self

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return self.run

    def dataset(self, dataset_id):
        self.dataset_ids.append(dataset_id)
        return self

    def iterate_items(self):
        return iter(self.items)


def robots(status, text=""):
    """A fake web: robots.txt answers with `status`; every other page answers 200."""
    def get(url, timeout=10):
        return (status, text) if url.endswith("/robots.txt") else (200, "<html></html>")
    return get


def test_the_run_input_has_the_caps():
    run_input = crawl.build_run_input(START)
    assert run_input["startUrls"] == [{"url": START}]
    assert run_input["maxCrawlPages"] == 3 and run_input["maxResults"] == 3
    assert run_input["maxCrawlDepth"] == 1 and run_input["maxConcurrency"] == 1
    assert run_input["respectRobotsTxtFile"] is True
    assert run_input["crawlerType"] == "cheerio"


def test_robots_txt_lives_at_the_root_of_the_site():
    assert crawl.robots_url("https://replyravi.github.io/ai-engineering-course/x/site/") == \
        "https://replyravi.github.io/robots.txt"


@pytest.mark.parametrize("status, text, allowed", [
    (404, "", True),                                          # no robots.txt: no rules
    (403, "", False),                                         # "no robots"
    (503, "", False),                                         # the site has trouble: wait
    (200, "User-agent: *\nDisallow: /rule-book/\n", False),   # a rule that says no
    (200, "User-agent: *\nDisallow: /private/\n", True),
])
def test_check_robots(status, text, allowed):
    assert crawl.check_robots(START, get=robots(status, text)) == (allowed, status)


def test_run_crawl_reads_the_typed_run_and_keeps_three_pages():
    client = FakeApify("token")
    run, pages = crawl.run_crawl(client, START)
    assert client.actor_id == "apify/website-content-crawler"
    assert client.dataset_ids == ["DS42"]                     # run.default_dataset_id, an attribute
    assert client.calls[0]["memory_mbytes"] == 1024
    assert client.calls[0]["run_input"]["maxCrawlPages"] == 3
    assert [p["url"] for p in pages] == [START, START + "fees.html", START + "units.html"]
    assert pages[2] == {"url": START + "units.html", "title": "", "text": "One onion bag holds 50 kg."}
    assert crawl.describe_cost(run) == "Run time 42 s, 0.0123 compute units, cost $0.0031"


def test_call_can_return_none():
    with pytest.raises(crawl.CrawlError, match="returned no run"):
        crawl.run_crawl(FakeApify("token", run=None), START)


def test_a_failed_run_is_reported_with_its_console_link():
    with pytest.raises(crawl.CrawlError, match="status FAILED.*actors/runs/RUN123"):
        crawl.run_crawl(FakeApify("token", run=typed_run("FAILED")), START)


def test_main_saves_crawl_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("APIFY_API_TOKEN", "test-token")
    monkeypatch.setattr(crawl, "OUT_FILE", tmp_path / "crawl.json")
    clients = []

    def factory(token):
        clients.append(FakeApify(token))
        return clients[-1]

    assert crawl.main(["--url", START], client_factory=factory, get=robots(404)) == 0
    out = capsys.readouterr().out
    assert "Saved crawl.json: 3 pages" in out
    assert "test-token" not in out                            # the token is never printed
    assert clients[0].token == "test-token"
    saved = json.loads((tmp_path / "crawl.json").read_text(encoding="utf-8"))
    assert saved["start_url"] == START and len(saved["pages"]) == 3
    assert saved["about"].endswith("run RUN123.")


def test_no_token_means_no_apify_call(no_keys, capsys):
    called = []
    assert crawl.main(["--url", START], client_factory=called.append, get=robots(404)) == 1
    assert "No APIFY_API_TOKEN found" in capsys.readouterr().out
    assert called == []


def test_robots_says_no_so_we_stop_before_apify(monkeypatch, capsys):
    monkeypatch.setenv("APIFY_API_TOKEN", "test-token")
    called = []
    blocked = robots(200, "User-agent: *\nDisallow: /\n")
    assert crawl.main(["--url", START], client_factory=called.append, get=blocked) == 1
    assert "robots.txt says no" in capsys.readouterr().out
    assert called == []


def test_check_only_never_needs_a_token(no_keys, capsys):
    assert crawl.main(["--url", START, "--check-only"], get=robots(404)) == 0
    out = capsys.readouterr().out
    assert "Start page:   HTTP 200, online" in out
    assert "Check only: Apify was not called." in out


def test_the_fixture_has_the_shape_crawl_py_saves():
    data = json.loads(crawl.FIXTURE.read_text(encoding="utf-8"))
    assert set(data) == {"about", "start_url", "max_pages", "pages"}
    assert "Not a recorded Apify run" in data["about"]
    assert len(data["pages"]) == data["max_pages"] == 3
    assert all(set(page) == {"url", "title", "text"} and page["url"].startswith(data["start_url"])
               for page in data["pages"])
