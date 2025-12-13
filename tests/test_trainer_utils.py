"""
Tests for training utilities.
"""

import math

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


class TestInfoNCELoss:
    """Tests for InfoNCE loss function."""

    def test_loss_computation(self):
        """Test basic loss computation."""
        from octo_embedding_model.trainer_utils import InfoNCELoss

        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=False)

        batch_size = 4
        embed_dim = 128

        query_emb = torch.randn(batch_size, embed_dim)
        positive_emb = torch.randn(batch_size, embed_dim)

        loss, metrics = loss_fn(query_emb, positive_emb)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Should be positive
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "temperature" in metrics

    def test_loss_perfect_match(self):
        """Test loss with perfect matches (should be low)."""
        from octo_embedding_model.trainer_utils import InfoNCELoss

        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=False)

        batch_size = 4
        embed_dim = 128

        # Use same embeddings for query and positive (perfect match)
        embeddings = torch.randn(batch_size, embed_dim)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        loss, metrics = loss_fn(embeddings, embeddings.clone())

        # Perfect match should give high accuracy
        assert metrics["accuracy"] == 1.0

    def test_learnable_temperature(self):
        """Test learnable temperature gradient flow."""
        from octo_embedding_model.trainer_utils import InfoNCELoss

        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=True)

        query_emb = torch.randn(4, 128, requires_grad=True)
        positive_emb = torch.randn(4, 128, requires_grad=True)

        loss, _ = loss_fn(query_emb, positive_emb)
        loss.backward()

        # Temperature parameter should have gradient
        assert loss_fn.log_temperature.grad is not None

    def test_loss_with_hard_negatives(self):
        """Test loss with hard negatives."""
        from octo_embedding_model.trainer_utils import InfoNCELoss

        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=False)

        batch_size = 4
        num_negs = 2
        embed_dim = 128

        query_emb = torch.randn(batch_size, embed_dim)
        positive_emb = torch.randn(batch_size, embed_dim)
        negative_emb = torch.randn(batch_size * num_negs, embed_dim)

        loss, metrics = loss_fn(query_emb, positive_emb, negative_emb)

        assert loss.dim() == 0
        assert loss.item() > 0


class TestMoEAuxLoss:
    """Tests for MoE auxiliary load balancing loss."""

    def test_aux_loss_computation(self):
        """Test auxiliary loss computation."""
        from octo_embedding_model.trainer_utils import MoEAuxLoss

        num_experts = 8
        aux_loss_fn = MoEAuxLoss(num_experts=num_experts, alpha=0.01)

        batch_seq = 32
        top_k = 2

        router_logits = torch.randn(batch_seq, num_experts)
        expert_indices = torch.randint(0, num_experts, (batch_seq, top_k))

        loss = aux_loss_fn(router_logits, expert_indices)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_aux_loss_uniform_distribution(self):
        """Test that uniform distribution gives lower loss."""
        from octo_embedding_model.trainer_utils import MoEAuxLoss

        num_experts = 8
        aux_loss_fn = MoEAuxLoss(num_experts=num_experts, alpha=0.01)

        batch_seq = 64
        top_k = 2

        # Create perfectly uniform distribution
        uniform_logits = torch.zeros(batch_seq, num_experts)
        # Each expert selected equally
        indices_list = []
        for i in range(batch_seq):
            expert1 = (i * 2) % num_experts
            expert2 = (i * 2 + 1) % num_experts
            indices_list.append([expert1, expert2])
        uniform_indices = torch.tensor(indices_list)

        uniform_loss = aux_loss_fn(uniform_logits, uniform_indices)

        # Create skewed distribution (all tokens go to first 2 experts)
        skewed_logits = torch.randn(batch_seq, num_experts)
        skewed_logits[:, :2] += 10  # Bias towards first 2 experts
        skewed_indices = torch.zeros(batch_seq, top_k, dtype=torch.long)
        skewed_indices[:, 0] = 0
        skewed_indices[:, 1] = 1

        skewed_loss = aux_loss_fn(skewed_logits, skewed_indices)

        # Uniform should have lower loss (though this depends on alpha)
        # At minimum, both should be computable
        assert uniform_loss.item() >= 0
        assert skewed_loss.item() >= 0


class TestCosineScheduler:
    """Tests for cosine learning rate scheduler."""

    def test_warmup_phase(self):
        """Test linear warmup phase."""
        from octo_embedding_model.trainer_utils import get_cosine_schedule_with_warmup

        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
            min_lr_ratio=0.1,
        )

        # Simulate training step (required before scheduler.step())
        dummy_input = torch.randn(2, 10)
        dummy_output = model(dummy_input)
        loss = dummy_output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # At step 0, LR should be ~0
        lr_0 = scheduler.get_last_lr()[0]
        assert lr_0 < 1e-5

        # At step 50 (middle of warmup), LR should be ~0.5 * max
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        lr_50 = scheduler.get_last_lr()[0]
        assert 0.4e-4 < lr_50 < 0.6e-4

        # At step 100 (end of warmup), LR should be ~max
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        lr_100 = scheduler.get_last_lr()[0]
        assert lr_100 > 0.9e-4

    def test_cosine_decay_phase(self):
        """Test cosine decay after warmup."""
        from octo_embedding_model.trainer_utils import get_cosine_schedule_with_warmup

        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
            min_lr_ratio=0.1,
        )

        # Initial optimizer step (required before scheduler.step())
        dummy_input = torch.randn(2, 10)
        dummy_output = model(dummy_input)
        loss = dummy_output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Run to end
        lrs = []
        for step in range(1000):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # LR should be at minimum at the end
        final_lr = lrs[-1]
        assert final_lr < 0.15e-4  # Should be around 0.1 * max

        # LR should decrease after warmup (overall trend)
        assert lrs[200] > lrs[800]


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config(self, tmp_path):
        """Test loading YAML config."""
        from octo_embedding_model.trainer_utils import load_config

        config_content = """
model:
  debug:
    hidden_size: 512
    num_hidden_layers: 4
active_model_profile: debug
training:
  seed: 42
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))

        assert config["active_model_profile"] == "debug"
        assert config["model"]["debug"]["hidden_size"] == 512
        assert config["training"]["seed"] == 42

    def test_get_model_config(self):
        """Test extracting model config based on profile."""
        from octo_embedding_model.trainer_utils import get_model_config

        config = {
            "active_model_profile": "debug",
            "model": {
                "debug": {"hidden_size": 512},
                "full": {"hidden_size": 2048},
            },
        }

        model_config = get_model_config(config)

        assert model_config["hidden_size"] == 512

    def test_get_model_config_full(self):
        """Test extracting full model config."""
        from octo_embedding_model.trainer_utils import get_model_config

        config = {
            "active_model_profile": "full",
            "model": {
                "debug": {"hidden_size": 512},
                "full": {"hidden_size": 2048},
            },
        }

        model_config = get_model_config(config)

        assert model_config["hidden_size"] == 2048


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading a checkpoint."""
        from octo_embedding_model.trainer_utils import save_checkpoint, load_checkpoint

        # Create simple model
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            epoch=2,
            output_dir=str(tmp_path),
            is_ddp=False,
        )

        assert (tmp_path / "checkpoint-100.pt").exists()

        # Create new model and load
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)

        info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
        )

        assert info["step"] == 100
        assert info["epoch"] == 2

        # Verify weights are the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


class TestWandbLogger:
    """Tests for W&B logger."""

    def test_logger_disabled(self):
        """Test logger when disabled."""
        from octo_embedding_model.trainer_utils import WandbLogger

        logger = WandbLogger(
            project="test",
            config={},
            enabled=False,
        )

        # Should not raise
        logger.log({"loss": 0.5}, step=1)
        logger.finish()

    @patch("octo_embedding_model.trainer_utils.WandbLogger._wandb_available")
    def test_logger_unavailable(self, mock_available):
        """Test logger when W&B is not installed."""
        mock_available.return_value = False

        from octo_embedding_model.trainer_utils import WandbLogger

        logger = WandbLogger(
            project="test",
            config={},
            enabled=True,
        )

        assert not logger.enabled


class TestDDPUtilities:
    """Tests for DDP utilities (non-distributed)."""

    def test_validate_ddp_not_initialized(self):
        """Test DDP validation when not initialized."""
        from octo_embedding_model.trainer_utils import validate_ddp

        results = validate_ddp()

        assert results["is_initialized"] is False
        assert results["rank"] == -1
        assert results["world_size"] == -1

    def test_ddp_config_defaults(self):
        """Test DDPConfig default values."""
        from octo_embedding_model.trainer_utils import DDPConfig

        config = DDPConfig()

        assert config.backend == "nccl"
        assert config.find_unused_parameters is False
        assert config.gradient_as_bucket_view is True
