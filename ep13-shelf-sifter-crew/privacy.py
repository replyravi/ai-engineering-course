"""
privacy.py - switch off CrewAI's telemetry, execution traces and update check.

Import this BEFORE crewai (tools.py and crew.py do it on their first line):

    import privacy  # noqa: F401

Why? Without these switches CrewAI 1.15 sends anonymous usage data, offers to collect
execution traces of your first crew run, and asks PyPI for a newer version on every
verbose run (then prints an "Update Available" box). Our course pins exact versions,
so we do not want any of that.

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import os

import llm  # noqa: F401  (reads the root .env first, so the values in YOUR .env win)

SWITCHES = {
    "CREWAI_DISABLE_TELEMETRY": "true",      # no anonymous usage data
    "OTEL_SDK_DISABLED": "true",             # the OpenTelemetry library CrewAI uses for it, off too
    "CREWAI_TRACING_ENABLED": "false",       # no execution traces, and no question about them
    "CREWAI_DISABLE_VERSION_CHECK": "true",  # no "A new version of CrewAI is available!" box
}

for name, value in SWITCHES.items():
    os.environ.setdefault(name, value)
