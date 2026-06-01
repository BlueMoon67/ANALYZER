#!/usr/bin/env bash
# run.sh — One command to process all CCTV clips and emit events to the API.
#
# Usage:
#   ./pipeline/run.sh [--store STORE_ID] [--clips-dir clips/] [--api-url http://localhost:8000]
#
# What it does:
#   1. Loops over all stores defined in store_layout.json
#   2. Runs detect.py on each store's 3 camera clips
#   3. Streams the resulting JSONL events into POST /events/ingest
#
# Requirements:
#   - Python 3.11+ with ultralytics, opencv-python, numpy installed
#   - clips/ directory with the CCTV footage
#   - store_layout.json in the current directory
#   - API server running (docker compose up)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ── Defaults ─────────────────────────────────────────────────────────────────
CLIPS_DIR="${CLIPS_DIR:-${PROJECT_ROOT}/clips}"
LAYOUT_FILE="${LAYOUT_FILE:-${PROJECT_ROOT}/store_layout.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/events}"
API_URL="${API_URL:-http://localhost:8000}"
WEIGHTS="${WEIGHTS:-yolov8n.pt}"
BATCH_SIZE=200       # events per ingest batch
MAX_WORKERS=1        # set >1 to parallelise clips (requires GPU / fast CPU)

# ── Colours for terminal output ───────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Parse args ────────────────────────────────────────────────────────────────
TARGET_STORE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --store)       TARGET_STORE="$2"; shift 2 ;;
    --clips-dir)   CLIPS_DIR="$2"; shift 2 ;;
    --api-url)     API_URL="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --weights)     WEIGHTS="$2"; shift 2 ;;
    *) log_error "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

# ── Preflight checks ──────────────────────────────────────────────────────────
log_info "Checking dependencies..."

python3 -c "import cv2, numpy" 2>/dev/null || {
  log_error "Missing python deps. Run: pip install opencv-python numpy ultralytics"
  exit 1
}

log_info "Checking API health at ${API_URL}/health ..."
for i in {1..10}; do
  if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
    log_info "API is up."
    break
  fi
  if [[ $i -eq 10 ]]; then
    log_error "API not reachable after 10 attempts. Is docker compose running?"
    exit 1
  fi
  log_warn "Waiting for API... (attempt $i/10)"
  sleep 3
done

# ── Discover stores ───────────────────────────────────────────────────────────
if [[ -n "${TARGET_STORE}" ]]; then
  STORES=("${TARGET_STORE}")
else
  # Discover from clips directory
  mapfile -t STORES < <(find "${CLIPS_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
fi

if [[ ${#STORES[@]} -eq 0 ]]; then
  log_error "No stores found under ${CLIPS_DIR}"
  exit 1
fi

log_info "Processing ${#STORES[@]} store(s): ${STORES[*]}"

# ── Main processing loop ──────────────────────────────────────────────────────
TOTAL_EVENTS=0
TOTAL_ERRORS=0

for STORE_ID in "${STORES[@]}"; do
  log_info "━━━ Processing store: ${STORE_ID} ━━━"

  STORE_CLIPS_DIR="${CLIPS_DIR}/${STORE_ID}"
  if [[ ! -d "${STORE_CLIPS_DIR}" ]]; then
    log_warn "No clip directory for ${STORE_ID}, skipping."
    continue
  fi

  STORE_OUTPUT_DIR="${OUTPUT_DIR}/${STORE_ID}"
  mkdir -p "${STORE_OUTPUT_DIR}"

  # Run detection pipeline
  log_info "Running detection pipeline for ${STORE_ID}..."
  python3 "${SCRIPT_DIR}/detect.py" \
    --store "${STORE_ID}" \
    --clips-dir "${CLIPS_DIR}" \
    --layout "${LAYOUT_FILE}" \
    --output-dir "${STORE_OUTPUT_DIR}" \
    --weights "${WEIGHTS}" \
    2>&1 | tee "${STORE_OUTPUT_DIR}/detect.log"

  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log_error "Detection failed for ${STORE_ID}. Check ${STORE_OUTPUT_DIR}/detect.log"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
    continue
  fi

  # Count events
  JSONL_FILES=("${STORE_OUTPUT_DIR}"/*.jsonl)
  STORE_EVENTS=0
  for f in "${JSONL_FILES[@]}"; do
    [[ -f "$f" ]] && STORE_EVENTS=$((STORE_EVENTS + $(wc -l < "$f")))
  done
  log_info "${STORE_ID}: generated ${STORE_EVENTS} events"
  TOTAL_EVENTS=$((TOTAL_EVENTS + STORE_EVENTS))

  # Ingest events into API in batches
  log_info "Ingesting ${STORE_EVENTS} events into ${API_URL}/events/ingest ..."
  python3 - <<PYTHON
import json, urllib.request, sys, math

jsonl_files = $(python3 -c "import glob, json; print(json.dumps(glob.glob('${STORE_OUTPUT_DIR}/*.jsonl')))")
batch_size = ${BATCH_SIZE}
api_url = "${API_URL}/events/ingest"
total_ok = 0
total_fail = 0

for fpath in jsonl_files:
    with open(fpath) as fh:
        events = [json.loads(line) for line in fh if line.strip()]

    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        payload = json.dumps({"events": batch}).encode()
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                ok = result.get("accepted", len(batch))
                fail = result.get("rejected", 0)
                total_ok += ok
                total_fail += fail
        except Exception as exc:
            print(f"  [WARN] Batch ingest failed: {exc}", file=sys.stderr)
            total_fail += len(batch)

print(f"  Ingested: {total_ok} accepted, {total_fail} rejected")
PYTHON

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
log_info "━━━ Pipeline Complete ━━━"
log_info "Total events generated : ${TOTAL_EVENTS}"
log_info "Store errors           : ${TOTAL_ERRORS}"

if [[ ${TOTAL_ERRORS} -gt 0 ]]; then
  log_warn "Some stores had errors. Check logs under ${OUTPUT_DIR}/"
fi

# Quick sanity check on metrics endpoint
log_info "Sanity check: GET ${API_URL}/stores/${STORES[0]}/metrics"
curl -sf "${API_URL}/stores/${STORES[0]}/metrics" | python3 -m json.tool | head -20 || true

echo ""
log_info "Done. Events are live in the API."
