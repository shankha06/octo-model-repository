#!/bin/bash
# =============================================================================
# Phase 1 Training Script for Chroma-MoE Embedding Model
# 
# Automatically configures OMP_NUM_THREADS for optimal performance.
# Usage: ./train_phase1.sh [NUM_GPUS] [CONFIG_FILE]
# =============================================================================

set -e

# Configuration
NUM_GPUS="${1:-2}"  # Default to 2 GPUs
CONFIG_FILE="${2:-config.yaml}"

# Calculate optimal OMP_NUM_THREADS
# Formula: nb_cpu_threads / nproc_per_node
NUM_CPU_THREADS=$(nproc)
OMP_NUM_THREADS=$((NUM_CPU_THREADS / NUM_GPUS))

# Ensure at least 1 thread
if [ "$OMP_NUM_THREADS" -lt 1 ]; then
    OMP_NUM_THREADS=1
fi

echo "=============================================="
echo "Chroma-MoE Phase 1 Training"
echo "=============================================="
echo "CPU Threads: $NUM_CPU_THREADS"
echo "GPUs: $NUM_GPUS"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Config: $CONFIG_FILE"
echo "=============================================="

uv run python octo_embedding_model/get_wikipedia_data.py --max-pages 50000

# Export environment variables
export OMP_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizers warning

# Run training
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    octo_embedding_model/train_phase1.py \
    --config $CONFIG_FILE
