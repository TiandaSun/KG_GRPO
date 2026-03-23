#!/usr/bin/env bash
# Download ConceptNet assertions file for KG server
set -eo pipefail

DATA_DIR="data/raw"
FILENAME="conceptnet-assertions-5.7.0.csv.gz"
URL="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/${FILENAME}"

mkdir -p "$DATA_DIR"

if [[ -f "$DATA_DIR/$FILENAME" ]]; then
    echo "Already exists: $DATA_DIR/$FILENAME ($(du -h "$DATA_DIR/$FILENAME" | cut -f1))"
    exit 0
fi

echo "Downloading ConceptNet assertions..."
echo "URL: $URL"
echo "Target: $DATA_DIR/$FILENAME"

wget -q --show-progress -O "$DATA_DIR/$FILENAME" "$URL"

echo "Done: $(du -h "$DATA_DIR/$FILENAME" | cut -f1)"
# Quick sanity check
zcat "$DATA_DIR/$FILENAME" | head -3
echo "..."
echo "Total lines: $(zcat "$DATA_DIR/$FILENAME" | wc -l)"
