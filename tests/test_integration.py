"""
Comprehensive integration tests for tokenizer and training pipeline.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenizerIntegration:
    """Tests for tokenizer training and integration."""

    def test_custom_tokenizer_trains_bpe(self, tmp_path):
        """Test BPE tokenizer training produces valid tokenizer."""
        from tokenizers import Tokenizer, models, trainers

        # Create a simple BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        # Train on sample text
        sample_texts = [
            "The company reported strong EBITDA growth with ROI exceeding expectations.",
            "This HDMI cable supports 4K resolution and is made of durable polyester.",
            "According to the 10-K filing, the company's WACC decreased by 50 basis points.",
            "SKU: ABC123 - Bluetooth wireless earbuds with USB-C charging.",
        ] * 50  # Repeat for more training data

        tokenizer.train_from_iterator(sample_texts, trainer=trainer)

        # Save tokenizer
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        assert tokenizer_path.exists()
        vocab = tokenizer.get_vocab()
        assert len(vocab) <= 1000

    def test_domain_terms_tokenization(self):
        """Test that domain terms can be tokenized consistently."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        domain_texts = [
            "EBITDA increased by 15% year-over-year",
            "HDMI 2.1 supports 8K resolution",
            "100% polyester with moisture-wicking technology",
        ]

        for text in domain_texts:
            tokens = tokenizer.tokenize(text)
            # Should tokenize without errors
            assert len(tokens) > 0
            # Should be able to encode/decode
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert len(decoded) > 0

    def test_tokenizer_vocab_size_in_model(self, tmp_path):
        """Test that model uses tokenizer's vocab size correctly."""
        from transformers import AutoTokenizer
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = len(tokenizer)

        config = ChromaConfig(
            vocab_size=vocab_size,
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

        # Model embedding should match tokenizer vocab
        assert model.embed_tokens.num_embeddings == vocab_size

        # Test forward pass with tokenized input
        text = "The company reported strong EBITDA growth."
        encoded = tokenizer(text, return_tensors="pt", max_length=32, padding="max_length", truncation=True)

        with torch.no_grad():
            embeddings = model(encoded["input_ids"], encoded["attention_mask"])

        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 64  # latent_pooler_dim


class TestEndToEndTrainingPipeline:
    """End-to-end tests for the full training pipeline."""

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a sample configuration for testing."""
        config = {
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
            "tokenizer": {
                "path": "./nonexistent_tokenizer",  # Will use fallback
                "vocab_size": 30522,  # bert-base-uncased size
                "fallback": "bert-base-uncased",
            },
            "training": {"seed": 42},
            "wandb": {"enabled": False},
            "phase1": {
                "training": {
                    "per_device_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                },
                "masking": {"mask_ratio": 0.15, "mean_span_length": 3},
            },
            "phase2": {
                "training": {
                    "per_device_batch_size": 2,
                },
                "contrastive": {"temperature": 0.05},
            },
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return str(config_path)

    def test_phase1_model_creation_with_tokenizer(self, sample_config):
        """Test Phase 1 model creation with tokenizer integration."""
        from transformers import AutoTokenizer
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining, create_model_from_config
        from octo_embedding_model.trainer_utils import load_config

        config = load_config(sample_config)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        model = create_model_from_config(config, tokenizer=tokenizer)

        assert model is not None
        assert isinstance(model, ChromaMoEForPretraining)
        # Vocab size should match tokenizer
        assert model.config.vocab_size == len(tokenizer)

    def test_phase1_forward_backward_pass(self, sample_config):
        """Test Phase 1 complete forward-backward with real tokenizer."""
        from transformers import AutoTokenizer
        from octo_embedding_model.train_phase1 import ChromaMoEForPretraining, create_model_from_config
        from octo_embedding_model.trainer_utils import load_config

        config = load_config(sample_config)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        model = create_model_from_config(config, tokenizer=tokenizer)

        # Prepare batch
        texts = [
            "The company reported strong EBITDA growth.",
            "HDMI cable with USB-C charging support.",
        ]
        encoded = tokenizer(
            texts,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Create labels (masked LM style)
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # Forward pass
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=labels,
        )

        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"].dim() == 0  # Scalar loss

        # Backward pass
        outputs["loss"].backward()

        # Check gradients flow
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                        for p in model.parameters() if p.requires_grad)
        assert has_grads, "Model should have gradients after backward"

    def test_phase2_contrastive_with_tokenizer(self, sample_config):
        """Test Phase 2 contrastive training with real tokenizer."""
        from transformers import AutoTokenizer
        from octo_embedding_model.train_phase2 import create_model_from_config
        from octo_embedding_model.trainer_utils import InfoNCELoss, load_config

        config = load_config(sample_config)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        model = create_model_from_config(config, tokenizer=tokenizer)
        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=True)

        # Prepare contrastive pairs
        queries = ["What is EBITDA?", "Find HDMI cables"]
        positives = ["EBITDA is earnings before interest taxes depreciation and amortization", 
                     "HDMI cable 4K high speed with ethernet"]

        query_encoded = tokenizer(queries, max_length=32, padding="max_length", 
                                  truncation=True, return_tensors="pt")
        positive_encoded = tokenizer(positives, max_length=32, padding="max_length", 
                                     truncation=True, return_tensors="pt")

        # Forward pass
        query_emb = model(query_encoded["input_ids"], query_encoded["attention_mask"])
        positive_emb = model(positive_encoded["input_ids"], positive_encoded["attention_mask"])

        loss, metrics = loss_fn(query_emb, positive_emb)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert "accuracy" in metrics
        assert "temperature" in metrics

        # Backward pass
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TestDataLoaderWithTokenizer:
    """Tests for data loading with tokenizer integration."""

    def test_span_masking_collator_with_bert_tokenizer(self):
        """Test span masking collator works with BERT tokenizer."""
        from transformers import AutoTokenizer
        from octo_embedding_model.data_loader import SpanMaskingCollator

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collator = SpanMaskingCollator(
            tokenizer=tokenizer,
            mask_ratio=0.15,
            mean_span_length=3,
            max_length=64,
        )

        batch = [
            {"text": "The company reported EBITDA of $1.5 billion for the fiscal year."},
            {"text": "This HDMI cable supports 4K resolution at 60Hz refresh rate."},
        ]

        output = collator(batch)

        assert "input_ids" in output
        assert "attention_mask" in output
        assert "labels" in output
        assert output["input_ids"].shape == (2, 64)

        # Check masking was applied
        mask_token_id = tokenizer.mask_token_id
        assert (output["input_ids"] == mask_token_id).any()

    def test_contrastive_collator_with_bert_tokenizer(self):
        """Test contrastive collator works with BERT tokenizer."""
        from transformers import AutoTokenizer
        from octo_embedding_model.data_loader import ContrastiveCollator

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collator = ContrastiveCollator(
            tokenizer=tokenizer,
            max_length=64,
            query_template="Instruct: {instruction}\nQuery: {query}",
        )

        batch = [
            {
                "query": "best HDMI cable",
                "positive": "Premium HDMI 2.1 cable 4K 120Hz",
                "instruction": "Find matching products",
            },
            {
                "query": "EBITDA calculation",
                "positive": "EBITDA equals operating income plus depreciation",
                "instruction": "Find relevant financial documents",
            },
        ]

        output = collator(batch)

        assert "query_input_ids" in output
        assert "positive_input_ids" in output
        assert output["query_input_ids"].shape == (2, 64)
        assert output["positive_input_ids"].shape == (2, 64)


class TestEnglishFiltering:
    """Tests for English language filtering."""

    def test_is_likely_english(self):
        """Test English detection function."""
        from octo_embedding_model.data_loader import is_likely_english

        english_texts = [
            "The company reported strong quarterly earnings with significant revenue growth.",
            "This product features Bluetooth 5.0 connectivity and USB-C charging.",
            "Available in sizes XS, S, M, L, XL with free shipping on orders over fifty dollars.",
        ]

        # Non-English texts need to be longer than 20 chars to be filtered
        non_english_texts = [
            "这是一个中文句子测试，用于检查语言过滤功能是否正常工作",
            "これは日本語のテキストです。言語フィルタリングのテストに使用されます",
            "هذا نص باللغة العربية لاختبار ميزة تصفية اللغة",
        ]

        for text in english_texts:
            assert is_likely_english(text), f"Should detect as English: {text}"

        for text in non_english_texts:
            assert not is_likely_english(text), f"Should not detect as English: {text}"

    def test_filter_english(self):
        """Test English filtering function."""
        from octo_embedding_model.data_loader import filter_english

        texts = [
            "English text about EBITDA and financial metrics for quarterly analysis.",
            "这是一个关于产品描述的中文句子，用于测试语言过滤功能",
            "Another English sentence with product SKU and detailed specifications.",
            "これは製品の説明に関する日本語の文章で、言語フィルタリングのテストに使用されます",
            "Final English text about USB-C charging and wireless connectivity features.",
        ]

        filtered = filter_english(texts)

        assert len(filtered) == 3
        assert all("English" in t or "SKU" in t or "USB" in t for t in filtered)


class TestGradientFlow:
    """Tests for gradient flow through the entire pipeline."""

    def test_gradient_flow_through_moe(self):
        """Test gradients flow through MoE layers."""
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel

        config = ChromaConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            moe_intermediate_size=64,
            num_routed_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            latent_pooler_dim=64,
        )

        model = ChromeMoEModel(config)
        model.train()

        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 16))
        embeddings = model(input_ids, None)

        # Create a simple loss
        loss = embeddings.sum()
        loss.backward()

        # Check embedding layer has gradients
        assert model.embed_tokens.weight.grad is not None
        assert model.embed_tokens.weight.grad.abs().sum() > 0

        # Check pooling head has gradients
        assert model.pooling_head.q_proj.weight.grad is not None

    def test_infonce_gradients(self):
        """Test InfoNCE loss produces proper gradients."""
        from octo_embedding_model.trainer_utils import InfoNCELoss

        loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=True)

        batch_size = 8
        embed_dim = 64

        query_emb = torch.randn(batch_size, embed_dim, requires_grad=True)
        positive_emb = torch.randn(batch_size, embed_dim, requires_grad=True)

        loss, metrics = loss_fn(query_emb, positive_emb)
        loss.backward()

        assert query_emb.grad is not None
        assert positive_emb.grad is not None
        assert loss_fn.log_temperature.grad is not None

        # Gradients should be non-zero
        assert query_emb.grad.abs().sum() > 0
        assert positive_emb.grad.abs().sum() > 0


class TestCheckpointingWithTokenizer:
    """Tests for checkpoint save/load with tokenizer info."""

    def test_save_load_checkpoint_preserves_vocab_size(self, tmp_path):
        """Test checkpoint preserves model configuration including vocab size."""
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
        from octo_embedding_model.trainer_utils import save_checkpoint, load_checkpoint

        vocab_size = 30522  # BERT vocab size

        config = ChromaConfig(
            vocab_size=vocab_size,
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            epoch=1,
            output_dir=str(tmp_path),
            is_ddp=False,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Verify state dict has correct embedding size
        embed_weight = checkpoint["model_state_dict"]["embed_tokens.weight"]
        assert embed_weight.shape[0] == vocab_size
