#!/bin/bash
# Download ConceptNet assertions and build graph cache.
#
# This downloads the raw ConceptNet data needed for Stage 1 (KG path extraction)
# and the KG server. Run once on a login node before training.
#
# Usage:
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh --skip-graph   # skip graph build (GPU node later)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SKIP_GRAPH=false
if [[ "${1:-}" == "--skip-graph" ]]; then
    SKIP_GRAPH=true
fi

echo "=== KG-Align-RL: Download Data ==="
echo "Project dir: $PROJECT_DIR"
echo ""

# --- 1. Create directories ---
mkdir -p data/raw data/processed data/eval

# --- 2. Download ConceptNet assertions ---
CONCEPTNET_URL="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CONCEPTNET_FILE="data/raw/conceptnet-assertions-5.7.0.csv.gz"

if [ -f "$CONCEPTNET_FILE" ]; then
    echo "[1/3] ConceptNet assertions already exist: $CONCEPTNET_FILE"
else
    echo "[1/3] Downloading ConceptNet assertions (~230MB)..."
    wget -q --show-progress -O "$CONCEPTNET_FILE" "$CONCEPTNET_URL"
    echo "      Saved to $CONCEPTNET_FILE"
fi
echo ""

# --- 3. Download HF models (cache only, no disk duplication) ---
echo "[2/3] Pre-downloading HF models to cache..."
HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
export HF_HOME
echo "      HF_HOME=$HF_HOME"

# Check if huggingface-cli is available
if command -v huggingface-cli &>/dev/null; then
    echo "      Downloading Qwen/Qwen2.5-1.5B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --quiet 2>/dev/null || \
        echo "      WARNING: Failed to download 1.5B model (will download on first use)"

    echo "      Downloading Qwen/Qwen2.5-7B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct --quiet 2>/dev/null || \
        echo "      WARNING: Failed to download 7B model (will download on first use)"
else
    echo "      huggingface-cli not found. Models will download on first use."
    echo "      Install with: pip install huggingface_hub"
fi
echo ""

# --- 4. Build ConceptNet graph cache (optional) ---
if [ "$SKIP_GRAPH" = true ]; then
    echo "[3/3] Skipping graph build (--skip-graph). Run the KG server once to build cache."
else
    GRAPH_CACHE="data/raw/conceptnet_graph_w2.0.pkl"
    if [ -f "$GRAPH_CACHE" ]; then
        echo "[3/3] Graph cache already exists: $GRAPH_CACHE"
    else
        echo "[3/3] Building ConceptNet graph (English, weight>=2.0)..."
        echo "      This parses the CSV and builds a NetworkX graph (~2min)."
        echo "      The graph is cached as a pickle for fast reloading."
        python -c "
from src_verl.kg_server.conceptnet_adapter import ConceptNetAdapter
adapter = ConceptNetAdapter('data/raw/conceptnet-assertions-5.7.0.csv.gz')
print(f'Graph built: {adapter.num_entities()} entities')
" 2>&1 || echo "      WARNING: Graph build failed. Will build on first KG server start."
    fi
fi

echo ""
echo "=== Download Complete ==="
echo ""
echo "Data files:"
ls -lh data/raw/ 2>/dev/null
echo ""
echo "Processed data (included in repo):"
ls -lh data/processed/*.jsonl 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Set up environment:  bash scripts/setup_env.sh"
echo "  2. Run tests:           python -m pytest tests/ -v"
echo "  3. Start KG server:     python src_verl/kg_server/server.py --kg conceptnet"
