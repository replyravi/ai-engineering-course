"""
privacy.py - switch off CrewAI's telemetry, execution traces and update check, and ONNX Runtime's usage data.

Import this BEFORE crewai and before any local model (every program and tool here that loads
one does it on its first line):

    import privacy  # noqa: F401

Why? Without these switches CrewAI 1.15 sends anonymous usage data, offers to collect
execution traces of your first crew run, and asks PyPI for a newer version on every
verbose run (then prints an "Update Available" box). Our course pins exact versions,
so we do not want any of that.

ONNX Runtime runs both local models of this episode. Its official builds have usage data
(telemetry) turned ON by default, and keep a device id in your user folder (on a Mac:
~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime/). ORT_DISABLE_TELEMETRY=1,
set before ONNX Runtime starts, stops the uploader, the events and the device id (ONNX Runtime's
docs/Privacy.md on GitHub, checked 2026-10-26; on our Mac no device id was written with it).

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

import os

import llm  # noqa: F401  (reads the root .env first, so the values in YOUR .env win)

SWITCHES = {
    "CREWAI_DISABLE_TELEMETRY": "true",      # no anonymous usage data
    "OTEL_SDK_DISABLED": "true",             # the OpenTelemetry library CrewAI uses for it, off too
    "CREWAI_TRACING_ENABLED": "false",       # no execution traces, and no question about them
    "CREWAI_DISABLE_VERSION_CHECK": "true",  # no "A new version of CrewAI is available!" box
    "ORT_DISABLE_TELEMETRY": "1",            # no usage data from ONNX Runtime (the local models)
}

for name, value in SWITCHES.items():
    os.environ.setdefault(name, value)
