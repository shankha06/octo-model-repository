# Chroma-MoE Embedding Model

A domain-specific embedding model for **finance** and **e-commerce** with Multi-Head Latent Attention (MLA), Mixture-of-Experts (MoE), and Latent Attention Pooling.

## Features

- **Custom BPE Tokenizer**: 64K/96K vocabulary trained on e-commerce and financial data
- **Domain-Specific Vocabulary**: Terms like "EBITDA", "ROI", "HDMI", "polyester" as single tokens
- **Two-Phase Training**: Domain-adaptive pre-training + contrastive fine-tuning
- **MoE Architecture**: 64 routed experts for specialized domain processing
- **Multi-GPU Support**: DDP for cluster training with W&B logging

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/octo-model-repository.git
cd octo-model-repository

# Install with uv (recommended)
pip install uv && uv sync --all-extras --all-groups && uv run hf auth login

# Install evaluation dependencies
pip install mteb scikit-learn
```

### Configuration

Edit `config.yaml` to configure:
- Model architecture (debug vs full profiles)
- Training hyperparameters
- Dataset selection
- W&B logging

```yaml
# Use debug profile for local testing
active_model_profile: "debug"

# Tokenizer settings
tokenizer:
  path: "./models/tokenizer"
  vocab_size: 65536
```

---

## Training Pipeline

### Step 1: Train Custom Tokenizer

Train a BPE tokenizer on Ecom-niverse and EDGAR financial data:

```bash
# Full training (64K vocab)
uv run python octo_embedding_model/train_tokenizer.py \
    --vocab-size 65536 \
    --max-samples 2500000 \
    --output-dir ./models/tokenizer \

# Quick debug mode (1K samples)
uv run python octo_embedding_model/train_tokenizer.py \
    --vocab-size 65536 \
    --max-samples 2500000 \
    --output-dir ./models/tokenizer \
    --debug

# 96K vocabulary for larger model
uv run python octo_embedding_model/train_tokenizer.py \
    --vocab-size 98304 \
    --max-samples 7500000 \
    --output-dir ./models/tokenizer \

# Verify tokenizer
uv run python octo_embedding_model/train_tokenizer.py \
    --verify-only ./models/tokenizer
```

### Step 2: Phase 1 - Domain-Adaptive Pre-training

Pre-train with span masking MLM on domain-specific corpora:

```bash
# Single GPU (local testing)
uv run python octo_embedding_model/train_phase1.py --config config.yaml

# Multi-GPU cluster training with DDP
uv run torchrun --nproc_per_node=8 octo_embedding_model/train_phase1.py \
    --config config.yaml

# Validate DDP setup
uv run torchrun --nproc_per_node=2 octo_embedding_model/train_phase1.py \
    --config config.yaml --test-ddp
```

**Phase 1 Features:**
- T5-style span masking (15% mask ratio)
- MoE auxiliary load balancing loss
- Cosine learning rate with warmup
- FineWeb-Edu 15% mixture for grammatical competence

### Step 3: Phase 2 - Contrastive Fine-tuning

Fine-tune with InfoNCE loss on query-document pairs:

```bash
# Single GPU
uv run python octo_embedding_model/train_phase2.py \
    --config config.yaml \
    --pretrained-path ./checkpoints/checkpoint-100000.pt

# Multi-GPU cluster
torchrun --nproc_per_node=8 octo_embedding_model/train_phase2.py \
    --config config.yaml \
    --pretrained-path ./checkpoints/checkpoint-100000.pt
```

**Phase 2 Features:**
- InfoNCE loss with learnable temperature
- In-batch negatives + hard negative mining
- GradCache for large effective batch sizes
- Instruction-following format for queries

---

## Evaluation

### Evaluate on All Benchmarks

```bash
uv run python octo_embedding_model/evaluate.py \
    --model-path ./checkpoints/phase2/final_model.pt \
    --tokenizer-path ./models/tokenizer \
    --benchmark all
```

### Evaluate on Specific Benchmarks

```bash
# FinMTEB (finance)
uv run python octo_embedding_model/evaluate.py \
    --model-path ./checkpoints/phase2/final_model.pt \
    --benchmark finmteb

# MTEB (general)
uv run python octo_embedding_model/evaluate.py \
    --model-path ./checkpoints/phase2/final_model.pt \
    --benchmark mteb

# Amazon ESCI (e-commerce)
uv run python octo_embedding_model/evaluate.py \
    --model-path ./checkpoints/phase2/final_model.pt \
    --benchmark esci

# Debug mode (smaller sample sizes)
uv run python octo_embedding_model/evaluate.py \
    --model-path ./checkpoints/phase2/final_model.pt \
    --benchmark all \
    --debug
```

### Evaluation Metrics

| Benchmark | Metrics |
|-----------|---------|
| FinMTEB | Classification accuracy, retrieval nDCG |
| MTEB | Average score across retrieval tasks |
| Amazon ESCI | NDCG@10, MRR |

---

## Model Architecture

```
Chroma-MoE (3B parameters - full model)
├── Token Embeddings (vocab_size × hidden_size)
├── 28 Transformer Blocks
│   ├── Multi-Head Latent Attention (MLA)
│   │   ├── KV LoRA compression
│   │   └── RoPE positional encoding
│   └── Mixture-of-Experts (MoE)
│       ├── 64 routed experts
│       ├── 2 shared experts
│       └── Top-6 routing
└── Latent Attention Pooling → 4096-dim embedding
```

### Model Profiles

| Profile | Hidden Size | Layers | Experts | Parameters |
|---------|-------------|--------|---------|------------|
| `debug` | 512 | 4 | 8 | ~50M |
| `full` | 2048 | 28 | 64 | ~3B |

---

## Datasets

### Pre-training (Phase 1)
- **FinMTEB**: Finance domain text
- **FineWeb-Edu**: General grammatical competence (15% mixture)

### Contrastive Tuning (Phase 2)
- **Amazon ESCI**: E-commerce query-product pairs with hard negatives
- **ConvFinQA**: Financial Q&A
- **MS-MARCO**: General retrieval regularization

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suites
uv run pytest tests/test_training.py -v
uv run pytest tests/test_integration.py -v
uv run pytest tests/test_trainer_utils.py -v
```

---

## Project Structure

```
octo-model-repository/
├── config.yaml                      # Training configuration
├── octo_embedding_model/
│   ├── model_architecture.py        # Chroma-MoE model
│   ├── data_loader.py               # Dataset loading utilities
│   ├── trainer_utils.py             # Training utilities
│   ├── train_tokenizer.py           # BPE tokenizer training
│   ├── train_phase1.py              # Phase 1 pre-training
│   ├── train_phase2.py              # Phase 2 contrastive tuning
│   └── evaluate.py                  # Evaluation script
├── tests/
│   ├── test_data_loader.py
│   ├── test_trainer_utils.py
│   ├── test_training.py
│   └── test_integration.py          # Comprehensive integration tests
└── tokenizer/                       # Trained tokenizer (after training)
```

---

## W&B Logging

Training metrics are logged to Weights & Biases:

```yaml
# config.yaml
wandb:
  enabled: true
  project: "chroma-moe"
  entity: "your-entity"
  tags: ["embedding", "moe", "finance"]
```

Tracked metrics:
- Training loss
- Learning rate
- Contrastive accuracy
- Temperature (Phase 2)
- MoE load balance

---

## License

MIT License

## Citation

```bibtex
@misc{chromamoe2024,
  title={Chroma-MoE: Domain-Specific Embedding Model},
  author={Shankhadeep Roy},
  year={2025}
}
```
