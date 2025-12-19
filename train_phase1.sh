#!/bin/bash
# =============================================================================
# Phase 1 Training Script for Chroma-MoE Embedding Model
#
# Description:
#   Orchestrates the Phase 1 training process, including data preparation
#   and distributed training execution.
#
# Features:
#   - Automated dependency checking
#   - Data pre-downloading and caching
#   - Dynamic hardware resource optimization (OMP_NUM_THREADS)
#   - Robust error handling and logging
#
# Usage:
#   ./train_phase1.sh [NUM_GPUS] [CONFIG_FILE]
# 
# Example:
#   ./train_phase1.sh 8 config.yaml
# =============================================================================

# -----------------------------------------------------------------------------
# Strict Error Handling
# -----------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration & Defaults
# -----------------------------------------------------------------------------
DEFAULT_NUM_GPUS=2
DEFAULT_CONFIG="config.yaml"
CACHE_DIR="./data/pretraining_cache"
LOG_FILE="training_phase1.log"

# Arguments
NUM_GPUS="${1:-$DEFAULT_NUM_GPUS}"
CONFIG_FILE="${2:-$DEFAULT_CONFIG}"

# -----------------------------------------------------------------------------
# Logging Helper
# -----------------------------------------------------------------------------
log() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local level="$1"
    shift
    local message="$*"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
log "INFO" "Starting Phase 1 Training Workflow"
log "INFO" "Configuration: GPUs=$NUM_GPUS, Config=$CONFIG_FILE"

if ! command -v uv &> /dev/null; then
    log "ERROR" "'uv' tool is not installed. Please install it first."
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    log "ERROR" "Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# -----------------------------------------------------------------------------
# Resource Optimization
# -----------------------------------------------------------------------------
NUM_CPU_THREADS=$(nproc)
OMP_THREADS=$((NUM_CPU_THREADS / NUM_GPUS))
# Ensure at least 1 thread
[ "$OMP_THREADS" -lt 1 ] && OMP_THREADS=1

log "INFO" "System Resources: CPU Threads=$NUM_CPU_THREADS"
log "INFO" "Optimization: Setting OMP_NUM_THREADS=$OMP_THREADS"

export OMP_NUM_THREADS=$OMP_THREADS
export TOKENIZERS_PARALLELISM=false

# -----------------------------------------------------------------------------
# Step 1: Data Preparation
# -----------------------------------------------------------------------------
log "INFO" "Step 1/2: Preparing Datasets..."

# 1.1 Extract Wikipedia Data
log "INFO" "Running Wikipedia extraction (get_wikipedia_data.py)..."
if ! uv run python octo_embedding_model/get_wikipedia_data.py --max-pages 50000; then
    log "ERROR" "Wikipedia extraction failed."
    exit 1
fi

# 1.2 Download External Datasets (HuggingFace)
log "INFO" "Downloading/Verifying external datasets (download_datasets.py)..."
if ! uv run python octo_embedding_model/download_datasets.py \
    --config "$CONFIG_FILE" \
    --output-dir "$CACHE_DIR"; then
    log "ERROR" "Dataset download failed."
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Training Execution
# -----------------------------------------------------------------------------
log "INFO" "Step 2/2: Starting Distributed Training..."

CMD="uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    octo_embedding_model/train_phase1.py \
    --config $CONFIG_FILE \
    --local-data $CACHE_DIR"

log "INFO" "Executing: $CMD"

if $CMD; then
    log "INFO" "Training completed successfully."
else
    log "ERROR" "Training failed."
    exit 1
fi
