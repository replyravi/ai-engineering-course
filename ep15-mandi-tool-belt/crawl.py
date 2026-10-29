"""
crawl.py - Step 11: crawl at most 3 pages with an Apify Actor and save them as crawl.json.

An Actor is a ready-made program that runs on Apify's servers. We use Apify's own
"Website Content Crawler": we give it a start page and caps, it visits the pages and
returns their clean text in a dataset. We keep only url, title and text.

    python crawl.py --check-only          robots.txt + the exact input we would send (no account needed)
    python crawl.py                       the real crawl (needs APIFY_API_TOKEN in the root .env)
    python crawl.py --url https://...     crawl another start page (only a site that allows it!)

Rules we follow on every crawl:
  - only a site that allows it: our own rule-book pages (site/ in this folder, published by Ravi)
  - robots.txt is checked first, and the Actor is told to respect it too
  - caps: 3 pages, 1 link deep, 1 page at a time, 1 GB of memory, 5 minutes at most
  - the token stays in .env and is never printed

No Apify account? Skip this step: ingest.py reads fixtures/crawl.json, an offline copy of the same pages.

Author: Ravi Shanker Singh (TechWithRavi)
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import llm  # noqa: F401  (importing llm reads the ONE root .env, where APIFY_API_TOKEN lives)

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "crawl.json"
FIXTURE = HERE / "fixtures" / "crawl.json"

ACTOR_ID = "apify/website-content-crawler"
# Where Ravi publishes the three pages of site/ (GitHub Pages of the course repo).
DEFAULT_START_URL = "https://replyravi.github.io/ai-engineering-course/ep15-mandi-tool-belt/site/"

MAX_PAGES = 3                         # the cap: the start page + 2 linked pages
MEMORY_MB = 1024                      # less memory = fewer compute units = less credit used
RUN_TIMEOUT = timedelta(minutes=5)    # the run is stopped after 5 minutes, whatever happens
USER_AGENT = "mandi-tool-belt/1.0 (learning project; https://github.com/replyravi/ai-engineering-course)"

NO_TOKEN_HELP = """No APIFY_API_TOKEN found in your .env file.

This step is optional. To crawl for real with Apify's free plan (check today's terms at https://apify.com/pricing):
  1. Sign up at https://console.apify.com
  2. Open the API & Integrations page (https://console.apify.com/settings/integrations)
     and copy your personal API token
  3. Add it to the ONE .env file at the repo root:  APIFY_API_TOKEN=paste-your-token-here
     (never paste it into code, a screenshot or a chat)

No account? Nothing is lost: ingest.py uses fixtures/crawl.json, an offline copy of the same pages."""


# ---------------------------------------------------------------------------
# 1. Permission first: robots.txt (the same rules as Episode 13's robots_check.py)
# ---------------------------------------------------------------------------

def robots_url(page_url: str) -> str:
    """robots.txt always lives at the root of the site (scheme + host), never in a sub-folder."""
    parts = urlsplit(page_url)
    return f"{parts.scheme}://{parts.netloc}/robots.txt"


def http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    """GET a URL with our User-Agent. Returns (HTTP status, text). Raises OSError when there is no answer."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:      # 404, 403, 500 ... are answers too
        return error.code, ""


def check_robots(page_url: str, get=http_get) -> tuple[bool, int]:
    """May our User-Agent fetch page_url? Returns (allowed, HTTP status of robots.txt).

    200 -> follow the rules in the file;  401/403 -> the site says no robots;
    other 4xx (like 404) -> no robots.txt, so no rules;  5xx -> the site has trouble: do not crawl now.
    `get` can be swapped for a fake in tests, so tests never use the internet.
    """
    status, text = get(robots_url(page_url))
    parser = RobotFileParser()
    if status == 200:
        parser.parse(text.splitlines())
    elif status in (401, 403) or status >= 500:
        parser.disallow_all = True
    else:
        parser.allow_all = True
    return parser.can_fetch(USER_AGENT, page_url), status


# ---------------------------------------------------------------------------
# 2. The Actor's input: start page + caps (field names from the Actor's input schema)
# ---------------------------------------------------------------------------

def build_run_input(start_url: str, max_pages: int = MAX_PAGES) -> dict:
    return {
        "startUrls": [{"url": start_url}],
        "maxCrawlPages": max_pages,          # the page cap (the default is 9,999,999: effectively no cap)
        "maxCrawlDepth": 1,                  # only pages linked straight from the start page
        "maxResults": max_pages,             # never store more pages than that
        "maxConcurrency": 1,                 # one page at a time: polite to the site
        "crawlerType": "cheerio",            # plain HTTP, no browser: the cheapest; our pages need no JavaScript
        "respectRobotsTxtFile": True,        # the Actor checks robots.txt as well
        "proxyConfiguration": {"useApifyProxy": False},   # our own site does not need a proxy
    }


# ---------------------------------------------------------------------------
# 3. Run the Actor and read its dataset (apify-client 3: typed Run objects, not dicts)
# ---------------------------------------------------------------------------

class CrawlError(RuntimeError):
    """The crawl did not finish with pages we can use. The message says why."""


def items_to_pages(items) -> list[dict]:
    """Keep only what we need from each dataset item: url, title and text. Skip empty pages."""
    pages = []
    for item in items:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        title = (item.get("metadata") or {}).get("title") or ""
        pages.append({"url": item.get("url", ""), "title": title, "text": text})
    return pages


def run_crawl(client, start_url: str, max_pages: int = MAX_PAGES):
    """Start the Actor, wait for it, read its pages. Returns (run, pages)."""
    run = client.actor(ACTOR_ID).call(
        run_input=build_run_input(start_url, max_pages),
        memory_mbytes=MEMORY_MB,
        run_timeout=RUN_TIMEOUT,
    )
    if run is None:                                       # v3: call() can return None
        raise CrawlError("Apify returned no run (call() gave None). Check the Actor id and try again.")
    if run.status != "SUCCEEDED":
        raise CrawlError(f"The run ended with status {run.status}. Open it in the Apify console: "
                         f"https://console.apify.com/actors/runs/{run.id}")
    items = client.dataset(run.default_dataset_id).iterate_items()   # v3: an attribute, not run["defaultDatasetId"]
    pages = items_to_pages(items)[:max_pages]
    if not pages:
        raise CrawlError("The crawl finished but found no text. Is the start page online?")
    return run, pages


def save_pages(pages: list[dict], start_url: str, about: str, path=None) -> Path:
    """Write the pages as JSON: the same shape as fixtures/crawl.json."""
    path = Path(path or OUT_FILE)
    data = {"about": about, "start_url": start_url, "max_pages": MAX_PAGES, "pages": pages}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def describe_cost(run) -> str:
    """What the run used, from the typed Run object (the console shows the same numbers)."""
    stats = run.stats
    units = f"{stats.compute_units:.4f} compute units" if stats and stats.compute_units is not None else "compute units: n/a"
    usd = f"${run.usage_total_usd:.4f}" if run.usage_total_usd is not None else "not reported yet"
    seconds = f"{stats.run_time_secs:.0f} s" if stats and stats.run_time_secs is not None else "n/a"
    return f"Run time {seconds}, {units}, cost {usd}"


# ---------------------------------------------------------------------------
# 4. The command line
# ---------------------------------------------------------------------------

def main(argv=None, client_factory=None, get=http_get) -> int:
    parser = argparse.ArgumentParser(description="Crawl at most 3 pages with Apify and save crawl.json.")
    parser.add_argument("--url", default=os.getenv("CRAWL_START_URL", "").strip() or DEFAULT_START_URL,
                        help="start page (default: CRAWL_START_URL in .env, else Ravi's rule-book pages)")
    parser.add_argument("--check-only", action="store_true",
                        help="check robots.txt and show the Actor input, without calling Apify")
    args = parser.parse_args(argv)

    print(f"Start page:   {args.url}")
    try:
        allowed, status = check_robots(args.url, get=get)
    except OSError as error:
        print(f"Could not download {robots_url(args.url)}: {error}")
        print("No answer means no permission. Check your internet connection and try again.")
        return 1
    print(f"robots.txt:   {robots_url(args.url)} -> HTTP {status}, allowed for us: {allowed}")
    if not allowed:
        print("robots.txt says no. We do not crawl this site.")
        return 1

    print(f"Actor:        {ACTOR_ID}   memory {MEMORY_MB} MB, stopped after "
          f"{int(RUN_TIMEOUT.total_seconds() // 60)} minutes")
    print("Actor input (the caps are in here):")
    print(json.dumps(build_run_input(args.url), indent=2))

    if args.check_only:
        try:
            page_status, _ = get(args.url)
        except OSError as error:
            print(f"Start page:   no answer ({error})")
            return 1
        verdict = "online" if page_status == 200 else "NOT online yet (publish site/ first, see the README)"
        print(f"Start page:   HTTP {page_status}, {verdict}")
        print("Check only: Apify was not called.")
        return 0

    token = os.getenv("APIFY_API_TOKEN", "").strip()
    if not token or token.startswith("paste-your"):
        print("\n" + NO_TOKEN_HELP)
        return 1

    if client_factory is None:
        from apify_client import ApifyClient
        client_factory = ApifyClient
    print(f"\nRunning the Actor (at most {MAX_PAGES} pages). Waiting for it to finish ...")
    try:
        run, pages = run_crawl(client_factory(token), args.url)
    except CrawlError as error:
        print(f"Crawl failed: {error}")
        return 1
    except Exception as error:      # wrong token, no internet, out of free credit ...
        print(f"Apify call failed: {type(error).__name__}: {error}")
        return 1

    for number, page in enumerate(pages, start=1):
        print(f"  {number}. {page['url']}  {len(page['text']):,} characters  \"{page['title']}\"")
    about = f"Crawled with {ACTOR_ID} by crawl.py on {date.today().isoformat()}, run {run.id}."
    path = save_pages(pages, args.url, about)
    print(f"Saved {path.name}: {len(pages)} pages")
    print(describe_cost(run))
    print(f"See the run and its cost in the console: https://console.apify.com/actors/runs/{run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
