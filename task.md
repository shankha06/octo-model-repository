# Chroma-MoE Project Tasks

## Phase 1: Pre-training
- [x] Fix DDP "unused parameters" error (`find_unused_parameters=True`)
- [x] Fix missing FinMTEB dataset (replaced with ashraq/financial-news etc.)
- [x] Align Phase 1 datasets with Tokenizer (Ecom-niverse, Financial, FineWeb-Edu)
- [x] Optimize data loading (Streaming IterableDataset)
- [x] Optimize training speed (Flash Attention 2, Mixed Precision, TF32)
- [x] Implement dynamic random masking ratio (0.05 - max)
- [ ] Verify training run with optimizations
- [ ] Document Phase 1 training procedures

## Phase 2: Fine-tuning
- [ ] Prepare Phase 2 datasets (Contrastive)
- [ ] Implement Phase 2 training loop optimizations

## General
- [ ] Create comprehensive README.md
- [ ] Create evaluation script (`evaluate.py`)
