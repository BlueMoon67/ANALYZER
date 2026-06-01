"""
feed.py — Read JSONL event files produced by detect.py and POST to the API.

Supports two modes:
  batch   — Read a complete .jsonl file and send all events immediately.
  stream  — Tail the file and send events as they arrive (for Part E dashboard).

Usage:
    # Batch ingest after clip processing
    python pipeline/feed.py --file events/STORE_BLR_002_CAM_ENTRY_01.jsonl

    # Stream mode (simulated real-time for dashboard)
    python pipeline/feed.py --file events/STORE_BLR_002_CAM_ENTRY_01.jsonl --stream

    # Ingest all .jsonl files in a directory
    python pipeline/feed.py --dir events/ --api http://localhost:8000
"""

import argparse
import json
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("feed")

DEFAULT_API   = "http://localhost:8000"
BATCH_SIZE    = 200          # events per POST request
STREAM_DELAY  = 0.067        # ~15fps replay delay in seconds


def parse_args():
    p = argparse.ArgumentParser(description="Feed JSONL events into Store Intelligence API")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Single .jsonl file to ingest")
    g.add_argument("--dir",  type=str, help="Directory of .jsonl files (batch ingest all)")
    p.add_argument("--api",    type=str, default=DEFAULT_API)
    p.add_argument("--stream", action="store_true",
                   help="Stream mode: tail file and send events with real-time delay")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def post_events(api_base: str, events: list[dict]) -> dict:
    url  = f"{api_base}/events/ingest"
    body = json.dumps({"events": events}).encode()
    req  = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        logger.error("HTTP %s posting events: %s", exc.code, body_text[:300])
        return {"accepted": 0, "rejected": len(events), "duplicate": 0}
    except Exception as exc:
        logger.error("Request failed: %s", exc)
        return {"accepted": 0, "rejected": len(events), "duplicate": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(path: str, api: str, batch_size: int) -> tuple[int, int]:
    total_accepted = 0
    total_rejected = 0
    buffer: list[dict] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                buffer.append(event)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line: %s", exc)
                continue

            if len(buffer) >= batch_size:
                result = post_events(api, buffer)
                total_accepted += result.get("accepted", 0)
                total_rejected += result.get("rejected", 0)
                logger.info(
                    "Batch sent: accepted=%d rejected=%d duplicate=%d",
                    result.get("accepted", 0),
                    result.get("rejected", 0),
                    result.get("duplicate", 0),
                )
                buffer = []

    # Flush remainder
    if buffer:
        result = post_events(api, buffer)
        total_accepted += result.get("accepted", 0)
        total_rejected += result.get("rejected", 0)

    return total_accepted, total_rejected


# ─────────────────────────────────────────────────────────────────────────────
# Stream mode (tail + replay)
# ─────────────────────────────────────────────────────────────────────────────

def stream_file(path: str, api: str, batch_size: int):
    """
    Reads the file line by line with a small delay between events,
    simulating real-time clip replay. Used for the live dashboard (Part E).
    """
    logger.info("Stream mode: replaying %s at ~15fps", path)
    buffer: list[dict] = []
    total = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                buffer.append(event)
                total += 1
            except json.JSONDecodeError:
                continue

            if len(buffer) >= batch_size:
                result = post_events(api, buffer)
                logger.info(
                    "Stream batch: accepted=%d | total_sent=%d",
                    result.get("accepted", 0), total,
                )
                buffer = []
                time.sleep(STREAM_DELAY * batch_size)  # pace to ~real-time

    if buffer:
        post_events(api, buffer)

    logger.info("Stream complete — %d events replayed.", total)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.dir:
        files = sorted(Path(args.dir).glob("*.jsonl"))
        if not files:
            logger.error("No .jsonl files found in %s", args.dir)
            return
        for f in files:
            logger.info("Ingesting: %s", f)
            acc, rej = ingest_file(str(f), args.api, args.batch_size)
            logger.info("Done %s → accepted=%d rejected=%d", f.name, acc, rej)
    else:
        if args.stream:
            stream_file(args.file, args.api, args.batch_size)
        else:
            acc, rej = ingest_file(args.file, args.api, args.batch_size)
            logger.info("Ingest complete → accepted=%d rejected=%d", acc, rej)


if __name__ == "__main__":
    main()
