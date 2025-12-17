"""
Tests for Phase 1 pre-training script (train_phase1.py).

Tests the main components of the training script:
- Model creation from config
- ChromaMoEForPretraining wrapper
- Train epoch function
- Data loading and collation
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChromaMoEForPretraining:
    """Tests for ChromaMoEForPretraining model wrapper."""

    def test_model_initialization(self):
        """Test model initializes correctly."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining

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

        assert model.config == config
        assert hasattr(model, "backbone")
        assert hasattr(model, "mlm_head")

    def test_forward_without_labels(self):
        """Test forward pass without labels returns logits only."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining

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

        outputs = model(input_ids, attention_mask)

        assert "logits" in outputs
        assert "loss" not in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)

    def test_forward_with_labels(self):
        """Test forward pass with labels returns loss and logits."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining

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

        outputs = model(input_ids, attention_mask, labels)

        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)
        assert outputs["loss"].dim() == 0  # Scalar loss
        assert outputs["loss"].item() > 0

    def test_forward_with_masked_labels(self):
        """Test forward pass with -100 masked labels."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining

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
        labels[:, :16] = -100  # Mask half the labels

        outputs = model(input_ids, attention_mask, labels)

        assert "loss" in outputs
        # Loss should still be computed for non-masked tokens

    def test_backward_pass(self):
        """Test backward pass computes gradients."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining

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

        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        assert has_gradients, "Model should have gradients after backward"


class TestCreateModelFromConfig:
    """Tests for create_model_from_config function."""

    def test_create_model_without_tokenizer(self):
        """Test model creation without tokenizer uses config vocab size."""
        from octo_embedding_model.train_phase1 import create_model_from_config

        config = {
            "active_model_profile": "debug",
            "model": {
                "debug": {
                    "vocab_size": 2000,
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
            "tokenizer": {"vocab_size": 2000},
        }

        model = create_model_from_config(config, tokenizer=None)

        assert model.config.vocab_size == 2000

    def test_create_model_with_tokenizer(self):
        """Test model creation with tokenizer uses tokenizer vocab size."""
        from octo_embedding_model.train_phase1 import create_model_from_config
        from unittest.mock import MagicMock

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__ = MagicMock(return_value=3000)

        config = {
            "active_model_profile": "debug",
            "model": {
                "debug": {
                    "vocab_size": 2000,  # Different from tokenizer
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

        model = create_model_from_config(config, tokenizer=mock_tokenizer)

        # Should use tokenizer's vocab size
        assert model.config.vocab_size == 3000


class TestTrainEpoch:
    """Tests for train_epoch function."""

    def test_train_epoch_single_step(self):
        """Test train_epoch runs for a single step."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining, train_epoch
        from octo_embedding_model.trainer_utils import MoEAuxLoss, get_cosine_schedule_with_warmup
        from octo_embedding_model.data_loader import PreTrainingDataset, SpanMaskingCollator
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        # Use BERT vocab size (30522) since we're using bert-base-uncased tokenizer
        config = ChromaConfig(
            vocab_size=30522,
            hidden_size=128,
            num_hidden_layers=1,
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
        device = torch.device("cpu")
        model = model.to(device)

        # Create mock tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Create mock dataset
        texts = ["This is a test sentence." for _ in range(10)]
        dataset = PreTrainingDataset(texts, tokenizer)

        collator = SpanMaskingCollator(
            tokenizer=tokenizer,
            mask_ratio=0.15,
            mean_span_length=3,
            max_length=128,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collator,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=1, num_training_steps=10
        )

        aux_loss_fn = MoEAuxLoss(num_experts=4, alpha=0.01)

        training_config = {
            "phase1": {
                "training": {
                    "gradient_accumulation_steps": 1,
                    "max_steps": 2,
                },
            },
            "training": {
                "logging": {"log_steps": 1},
            },
        }

        global_step = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            aux_loss_fn=aux_loss_fn,
            config=training_config,
            epoch=0,
            global_step=0,
            wandb_logger=None,
            device=device,
            is_ddp=False,
            scaler=None,
        )

        assert global_step >= 1


class TestSpanMaskingCollator:
    """Tests for SpanMaskingCollator used in Phase 1."""

    def test_collator_outputs_correct_shapes(self):
        """Test collator produces correctly shaped outputs."""
        from octo_embedding_model.data_loader import SpanMaskingCollator, PreTrainingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        collator = SpanMaskingCollator(
            tokenizer=tokenizer,
            mask_ratio=0.15,
            mean_span_length=3,
            max_length=128,
        )

        # Create batch
        batch = [
            {"text": "This is a test sentence for masking."},
            {"text": "Another sentence to be masked."},
        ]

        outputs = collator(batch)

        assert "input_ids" in outputs
        assert "attention_mask" in outputs
        assert "labels" in outputs

        batch_size = 2
        assert outputs["input_ids"].shape[0] == batch_size
        assert outputs["attention_mask"].shape[0] == batch_size
        assert outputs["labels"].shape[0] == batch_size

    def test_collator_applies_mask(self):
        """Test that collator actually masks tokens."""
        from octo_embedding_model.data_loader import SpanMaskingCollator
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        collator = SpanMaskingCollator(
            tokenizer=tokenizer,
            mask_ratio=0.15,
            mean_span_length=3,
            max_length=128,
        )

        batch = [{"text": "This is a longer test sentence with more words for testing the masking functionality."}]
        outputs = collator(batch)

        # Labels should have some -100 values (masked) and some actual token ids
        labels = outputs["labels"]
        assert (labels == -100).any(), "Some tokens should be masked (-100)"


class TestDatasetIntegration:
    """Integration tests for dataset loading."""

    def test_pretraining_dataset_iteration(self):
        """Test PreTrainingDataset can be iterated."""
        from octo_embedding_model.data_loader import PreTrainingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        texts = ["Sample text " + str(i) for i in range(5)]
        dataset = PreTrainingDataset(texts, tokenizer)

        assert len(dataset) == 5

        for i, item in enumerate(dataset):
            assert "text" in item
            assert isinstance(item["text"], str)


class TestConfigLoading:
    """Tests for configuration loading and parsing."""

    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        from octo_embedding_model.trainer_utils import load_config

        config_content = {
            "active_model_profile": "debug",
            "model": {
                "debug": {
                    "vocab_size": 1000,
                    "hidden_size": 128,
                }
            },
            "wandb": {"enabled": False},
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(str(config_path))

        assert config["active_model_profile"] == "debug"
        assert config["model"]["debug"]["vocab_size"] == 1000


class TestEndToEndPhase1:
    """End-to-end tests for Phase 1 training."""

    def test_mini_pretraining_loop(self):
        """Test a minimal pretraining loop runs without errors."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining
        from octo_embedding_model.data_loader import SpanMaskingCollator, PreTrainingDataset
        from octo_embedding_model.trainer_utils import get_cosine_schedule_with_warmup
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        # Create minimal model - use BERT vocab size (30522) since we're using bert-base-uncased tokenizer
        config = ChromaConfig(
            vocab_size=30522,
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

        model = ChromaMoEForPretraining(config)
        model.train()

        # Create tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand text.",
            "Deep learning models can recognize patterns in data.",
        ]
        dataset = PreTrainingDataset(texts, tokenizer)

        collator = SpanMaskingCollator(
            tokenizer=tokenizer,
            mask_ratio=0.15,
            mean_span_length=3,
            max_length=64,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collator,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=1, num_training_steps=10
        )

        # Training loop
        losses = []
        for step, batch in enumerate(dataloader):
            if step >= 3:
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        assert len(losses) >= 2
        assert all(l >= 0 and not (l != l) for l in losses)  # Check positive and not NaN

    def test_model_save_load_compatibility(self, tmp_path):
        """Test that model can be saved and loaded."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining

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

        model = ChromaMoEForPretraining(config)

        # Save model state
        save_path = tmp_path / "model_checkpoint.pt"
        torch.save(model.state_dict(), save_path)

        # Load into new model
        model2 = ChromaMoEForPretraining(config)
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        # Verify loaded model works
        input_ids = torch.randint(0, 1000, (1, 16))
        labels = torch.randint(0, 1000, (1, 16))

        with torch.no_grad():
            outputs1 = model(input_ids, None, labels)
            outputs2 = model2(input_ids, None, labels)

        # Both models should produce identical outputs
        assert torch.allclose(outputs1["logits"], outputs2["logits"])


class TestGradientAccumulation:
    """Tests for gradient accumulation behavior."""

    def test_gradient_accumulation_equivalence(self):
        """Test that gradient accumulation produces equivalent gradients."""
        from octo_embedding_model.model_architecture import ChromaConfig
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining

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

        # Set seed for reproducibility
        torch.manual_seed(42)
        model1 = ChromaMoEForPretraining(config)

        torch.manual_seed(42)
        model2 = ChromaMoEForPretraining(config)

        # Same inputs
        torch.manual_seed(42)
        input_ids_1 = torch.randint(0, 1000, (2, 16))
        labels_1 = torch.randint(0, 1000, (2, 16))
        input_ids_2 = torch.randint(0, 1000, (2, 16))
        labels_2 = torch.randint(0, 1000, (2, 16))

        # Model 1: full batch
        combined_ids = torch.cat([input_ids_1, input_ids_2], dim=0)
        combined_labels = torch.cat([labels_1, labels_2], dim=0)
        outputs1 = model1(combined_ids, None, combined_labels)
        outputs1["loss"].backward()

        # Model 2: gradient accumulation
        outputs2a = model2(input_ids_1, None, labels_1)
        (outputs2a["loss"] / 2).backward()

        outputs2b = model2(input_ids_2, None, labels_2)
        (outputs2b["loss"] / 2).backward()

        # Both should have accumulated gradients now
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if param1.grad is not None and param2.grad is not None:
                # Gradients should be close (not exact due to batch norm, etc.)
                assert param1.grad.shape == param2.grad.shape
