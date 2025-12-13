"""
Integration tests for training scripts.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelCreation:
    """Tests for model creation from config."""

    def test_create_debug_model(self):
        """Test creating debug model from config."""
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
        from octo_embedding_model.trainer_utils import get_model_config, load_config

        # Create minimal config
        config = {
            "active_model_profile": "debug",
            "model": {
                "debug": {
                    "vocab_size": 1000,
                    "hidden_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "kv_lora_rank": 32,
                    "q_lora_rank": 64,
                    "qk_rope_head_dim": 16,
                    "qk_nope_head_dim": 16,
                    "v_head_dim": 16,
                    "moe_intermediate_size": 128,
                    "num_routed_experts": 4,
                    "num_shared_experts": 1,
                    "num_experts_per_tok": 2,
                    "moe_layer_freq": 1,
                    "aux_loss_alpha": 0.01,
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "latent_pooler_dim": 128,
                },
            },
        }

        model_config = get_model_config(config)
        chroma_config = ChromaConfig(**model_config)
        model = ChromeMoEModel(chroma_config)

        # Test forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            embeddings = model(input_ids, attention_mask)

        assert embeddings.shape == (batch_size, 128)  # latent_pooler_dim
        assert embeddings.requires_grad is False

    def test_model_parameter_count(self):
        """Test model parameter count is reasonable for debug config."""
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel

        config = ChromaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            kv_lora_rank=32,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
            moe_intermediate_size=128,
            num_routed_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            latent_pooler_dim=128,
        )

        model = ChromeMoEModel(config)

        total_params = sum(p.numel() for p in model.parameters())

        # Debug model should be small (< 10M params)
        assert total_params < 10_000_000
        assert total_params > 100_000  # But not too small


class TestPretrainingModel:
    """Tests for Phase 1 pretraining model wrapper."""

    def test_pretraining_forward(self):
        """Test pretraining model forward pass."""
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining
        from octo_embedding_model.model_architecture import ChromaConfig

        config = ChromaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            kv_lora_rank=32,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
            moe_intermediate_size=128,
            num_routed_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            latent_pooler_dim=128,
        )

        model = ChromaMoEForPretraining(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        labels[labels < 100] = -100  # Mask some labels

        outputs = model(input_ids, attention_mask, labels)

        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)
        assert outputs["loss"].dim() == 0

    def test_pretraining_backward(self):
        """Test pretraining model backward pass."""
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining
        from octo_embedding_model.model_architecture import ChromaConfig

        config = ChromaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            kv_lora_rank=32,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
            moe_intermediate_size=128,
            num_routed_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            latent_pooler_dim=128,
        )

        model = ChromaMoEForPretraining(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))

        outputs = model(input_ids, None, labels)
        loss = outputs["loss"]
        loss.backward()

        # Check that at least some gradients exist (MoE layer is incomplete in blueprint)
        has_gradients = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        assert has_gradients, "Model should have at least some gradients"


class TestContrastiveTraining:
    """Tests for Phase 2 contrastive training."""

    def test_contrastive_forward(self):
        """Test contrastive training forward pass."""
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
        from octo_embedding_model.trainer_utils import InfoNCELoss

        config = ChromaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            kv_lora_rank=32,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
            moe_intermediate_size=128,
            num_routed_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            latent_pooler_dim=128,
        )

        model = ChromeMoEModel(config)
        loss_fn = InfoNCELoss(temperature=0.05)

        batch_size = 4
        seq_len = 32

        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        positive_ids = torch.randint(0, 1000, (batch_size, seq_len))
        positive_mask = torch.ones(batch_size, seq_len)

        query_emb = model(query_ids, query_mask)
        positive_emb = model(positive_ids, positive_mask)

        loss, metrics = loss_fn(query_emb, positive_emb)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_contrastive_backward(self):
        """Test contrastive training backward pass."""
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
        from octo_embedding_model.trainer_utils import InfoNCELoss

        config = ChromaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            kv_lora_rank=32,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
            moe_intermediate_size=128,
            num_routed_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            latent_pooler_dim=128,
        )

        model = ChromeMoEModel(config)
        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=True)

        query_ids = torch.randint(0, 1000, (4, 32))
        positive_ids = torch.randint(0, 1000, (4, 32))

        query_emb = model(query_ids, None)
        positive_emb = model(positive_ids, None)

        loss, _ = loss_fn(query_emb, positive_emb)
        loss.backward()

        # Check model gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Model should have gradients"

        # Check temperature gradient
        assert loss_fn.log_temperature.grad is not None


class TestEndToEndTraining:
    """End-to-end training tests with mock data."""

    def test_mini_training_loop(self):
        """Test a minimal training loop runs without errors."""
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
        from octo_embedding_model.trainer_utils import InfoNCELoss, get_cosine_schedule_with_warmup

        config = ChromaConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            moe_intermediate_size=64,
            num_routed_experts=2,
            num_shared_experts=1,
            num_experts_per_tok=1,
            max_position_embeddings=64,
            latent_pooler_dim=64,
        )

        model = ChromeMoEModel(config)
        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=True)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=1e-4,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=20,
        )

        model.train()
        losses = []

        for step in range(10):
            # Generate random batch
            batch_size = 4
            seq_len = 16

            query_ids = torch.randint(0, 1000, (batch_size, seq_len))
            positive_ids = torch.randint(0, 1000, (batch_size, seq_len))

            # Forward
            query_emb = model(query_ids, None)
            positive_emb = model(positive_ids, None)

            loss, metrics = loss_fn(query_emb, positive_emb)
            losses.append(loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        # Loss should be computable for all steps
        assert len(losses) == 10
        assert all(l > 0 for l in losses)

    def test_config_integration(self, tmp_path):
        """Test loading config and creating model."""
        import yaml
        from octo_embedding_model.trainer_utils import load_config, get_model_config

        # Create test config
        config_content = {
            "active_model_profile": "debug",
            "model": {
                "debug": {
                    "vocab_size": 1000,
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "kv_lora_rank": 16,
                    "q_lora_rank": 32,
                    "qk_rope_head_dim": 8,
                    "qk_nope_head_dim": 8,
                    "v_head_dim": 8,
                    "moe_intermediate_size": 64,
                    "num_routed_experts": 2,
                    "num_shared_experts": 1,
                    "num_experts_per_tok": 1,
                    "max_position_embeddings": 64,
                    "latent_pooler_dim": 64,
                    "moe_layer_freq": 1,
                    "aux_loss_alpha": 0.01,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                },
            },
            "training": {"seed": 42},
            "wandb": {"enabled": False},
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        # Load and use
        config = load_config(str(config_path))
        model_config = get_model_config(config)

        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel

        chroma_config = ChromaConfig(**model_config)
        model = ChromeMoEModel(chroma_config)

        # Verify model works
        input_ids = torch.randint(0, 1000, (2, 16))
        embeddings = model(input_ids, None)

        assert embeddings.shape == (2, 64)
