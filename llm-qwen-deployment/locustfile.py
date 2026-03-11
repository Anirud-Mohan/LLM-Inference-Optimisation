"""Load test for the Qwen moral story endpoint.

Works for both pipelines — set RUN_TAG to distinguish output files

Per-request metrics are written to:
    results/request_metrics_<RUN_TAG>.csv
Columns: timestamp, pod_url, success, response_time_ms, completion_tokens, tokens_per_sec
"""

import csv
import itertools
import os
import random
import threading
import time
from pathlib import Path

import requests as _requests
from dotenv import load_dotenv
from locust import HttpUser, between, events, task

from generate_prompts import generate_prompts

load_dotenv()

# ── Pod URLs ───────────────────────────────────────────────────────────────────
POD_URLS: list[str] = []
for key in ("POD_1_URL", "POD_2_URL"):
    url = os.getenv(key, "").strip()
    if url:
        POD_URLS.append(url)

if not POD_URLS:
    raise RuntimeError(
        "Set POD_1_URL (and optionally POD_2_URL) in your .env file "
        "or as environment variables."
    )

# Round-robin pod selection — guarantees exactly even request distribution
_pod_cycle = itertools.cycle(range(len(POD_URLS)))
_pod_lock  = threading.Lock()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
RUN_TAG    = os.getenv("RUN_TAG", "run").strip()
PROMPTS    = generate_prompts()

print(f"[Locust] run_tag : {RUN_TAG}")
print(f"[Locust] pods    : {POD_URLS}")
print(f"[Locust] model   : {MODEL_NAME}")
print(f"[Locust] prompts : {len(PROMPTS):,}")

# ── Per-request CSV ────────────────────────────────────────────────────────────
METRICS_DIR  = Path("results")
METRICS_DIR.mkdir(exist_ok=True)
METRICS_FILE = METRICS_DIR / f"request_metrics_{RUN_TAG}.csv"

_csv_file   = METRICS_FILE.open("w", newline="")
_csv_writer = csv.writer(_csv_file)
_csv_writer.writerow([
    "timestamp", "pod_url", "success",
    "response_time_ms", "completion_tokens", "tokens_per_sec",
])


@events.quitting.add_listener
def _close_csv(environment, **kwargs):
    _csv_file.flush()
    _csv_file.close()


# ── Locust user ────────────────────────────────────────────────────────────────
class StoryUser(HttpUser):
    """Simulated user requesting children's moral stories.

    Distributes requests across available pods by random selection,
    giving approximate data-parallel load balancing across 2 pods.
    """

    wait_time = between(1, 3)

    @task
    def generate_story(self):
        with _pod_lock:
            pod_url = POD_URLS[next(_pod_cycle)]
        prompt  = random.choice(PROMPTS)
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0,
        }
        url = f"{pod_url.rstrip('/')}/v1/chat/completions"

        t0 = time.perf_counter()
        success           = False
        completion_tokens = 0
        tokens_per_sec    = 0.0

        try:
            resp       = _requests.post(url, json=payload, timeout=120)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            success    = resp.status_code == 200

            if success:
                try:
                    data              = resp.json()
                    usage             = data.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)
                    if elapsed_ms > 0 and completion_tokens > 0:
                        tokens_per_sec = (completion_tokens / elapsed_ms) * 1000
                except Exception:
                    success = False

            self.environment.events.request.fire(
                request_type="POST",
                name="/v1/chat/completions",
                response_time=elapsed_ms,
                response_length=len(resp.content),
                exception=None if success else Exception(f"HTTP {resp.status_code}"),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.environment.events.request.fire(
                request_type="POST",
                name="/v1/chat/completions",
                response_time=elapsed_ms,
                response_length=0,
                exception=exc,
            )

        _csv_writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            pod_url,
            success,
            round(elapsed_ms, 2),
            completion_tokens,
            round(tokens_per_sec, 2),
        ])
