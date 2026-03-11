#!/usr/bin/env bash
# =============================================================================
# SynthRAD2025 Task 1 - Training Data Download Script
# Dataset: https://doi.org/10.5281/zenodo.14918089
#
# Usage:
#   bash scripts/download_data.sh [--task1-only] [--anatomy HN|TH|AB]
#
# Requirements:
#   pip install zenodo-get   OR   use direct wget/curl with Zenodo API
# =============================================================================

set -euo pipefail

ZENODO_DOI="10.5281/zenodo.14918089"
DATA_DIR="data/raw"
TASK="task1"

# Parse args
ANATOMY_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --anatomy) ANATOMY_FILTER="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "$DATA_DIR"

echo "=============================================="
echo " SynthRAD2025 Data Download"
echo " DOI: $ZENODO_DOI"
echo " Destination: $DATA_DIR"
echo "=============================================="

# -------------------------------------------------
# Method 1: zenodo_get (recommended)
# -------------------------------------------------
if command -v zenodo_get &>/dev/null; then
    echo "[INFO] Using zenodo_get..."
    cd "$DATA_DIR"
    zenodo_get "$ZENODO_DOI"
    cd -

# -------------------------------------------------
# Method 2: Manual wget from Zenodo REST API
# -------------------------------------------------
else
    echo "[INFO] zenodo_get not found. Using Zenodo REST API..."
    echo "[INFO] Install zenodo_get with: pip install zenodo-get"
    echo ""

    RECORD_ID=$(echo "$ZENODO_DOI" | sed 's/10.5281\/zenodo\.//')
    API_URL="https://zenodo.org/api/records/${RECORD_ID}"

    echo "[INFO] Fetching file list from Zenodo record ${RECORD_ID}..."
    curl -s "$API_URL" | python3 -c "
import sys, json
record = json.load(sys.stdin)
files = record.get('files', [])
for f in files:
    name = f['key']
    url  = f['links']['self']
    size = f.get('size', 0) / (1024**3)
    print(f'{url}|{name}|{size:.2f}GB')
" > /tmp/synthrad_files.txt

    echo "[INFO] Available files:"
    cat /tmp/synthrad_files.txt | awk -F'|' '{printf "  %-60s %s GB\n", $2, $3}'
    echo ""

    # Filter and download
    while IFS='|' read -r url name size; do
        # Skip if anatomy filter set and doesn't match
        if [[ -n "$ANATOMY_FILTER" ]]; then
            case "$ANATOMY_FILTER" in
                HN) [[ "$name" != *"1HN"* && "$name" != *"head"* ]] && continue ;;
                TH) [[ "$name" != *"1TH"* && "$name" != *"thorax"* ]] && continue ;;
                AB) [[ "$name" != *"1AB"* && "$name" != *"abdomen"* ]] && continue ;;
            esac
        fi

        # Skip task2 files if task1 only
        [[ "$name" == *"task2"* || "$name" == *"2HN"* || "$name" == *"2TH"* || "$name" == *"2AB"* ]] && continue

        dest="$DATA_DIR/$name"
        if [[ -f "$dest" ]]; then
            echo "[SKIP] $name already exists"
            continue
        fi

        echo "[DOWN] $name (${size} GB)..."
        wget -q --show-progress -O "$dest" "$url"
    done < /tmp/synthrad_files.txt
fi

echo ""
echo "=============================================="
echo " Download complete. Extracting archives..."
echo "=============================================="

# Extract any zip/tar files
find "$DATA_DIR" -name "*.zip" | while read -r f; do
    echo "[UNZIP] $f"
    unzip -q -o "$f" -d "$DATA_DIR"
done

find "$DATA_DIR" -name "*.tar.gz" -o -name "*.tgz" | while read -r f; do
    echo "[UNTAR] $f"
    tar -xzf "$f" -C "$DATA_DIR"
done

echo ""
echo "[DONE] Data available at: $DATA_DIR"
echo ""
echo "Expected structure:"
echo "  data/raw/"
echo "  ├── 1HN/   (Head-and-Neck, ~340 cases)"
echo "  │   ├── 1HNA001/ → mr.mha, ct.mha, mask.mha"
echo "  │   └── ..."
echo "  ├── 1TH/   (Thorax, ~280 cases)"
echo "  └── 1AB/   (Abdomen, ~270 cases)"
