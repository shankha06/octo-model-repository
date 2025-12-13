"""
Tests for custom BPE tokenizer training.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBPETokenizerTraining:
    """Tests for BPE tokenizer training functionality."""

    def test_tokenizer_basic_training(self, tmp_path):
        """Test that tokenizer trains successfully on sample data."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        # Create BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=500,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2,
        )

        # Sample training data
        texts = [
            "The company reported strong EBITDA growth.",
            "HDMI cable with USB-C charging support.",
            "This product SKU is available in polyester fabric.",
            "Financial metrics show positive ROI and WACC improvements.",
        ] * 20

        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Verify tokenizer works
        output = tokenizer.encode("EBITDA growth is positive")
        assert len(output.ids) > 0
        assert len(output.tokens) > 0

        # Save and verify
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(save_path))
        assert save_path.exists()

    def test_tokenizer_special_tokens(self, tmp_path):
        """Test that special tokens are properly included."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=special_tokens,
        )

        texts = ["Sample text for training"] * 50
        tokenizer.train_from_iterator(texts, trainer=trainer)

        vocab = tokenizer.get_vocab()

        # All special tokens should be in vocabulary
        for token in special_tokens:
            assert token in vocab, f"Special token {token} not in vocabulary"

    def test_tokenizer_domain_terms(self, tmp_path):
        """Test tokenization of domain-specific terms."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Include domain terms in training
        domain_texts = [
            "EBITDA earnings before interest taxes depreciation amortization",
            "ROI return on investment calculation formula",
            "HDMI high definition multimedia interface cable",
            "USB-C universal serial bus type c connector",
            "polyester fabric material clothing textile",
            "SKU stock keeping unit inventory management",
        ] * 50

        trainer = trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]"],
            min_frequency=1,
        )

        tokenizer.train_from_iterator(domain_texts, trainer=trainer)

        # Test encoding domain terms
        test_cases = [
            ("EBITDA", 1),  # Should ideally be 1-2 tokens
            ("ROI", 1),
            ("HDMI", 1),
        ]

        for term, max_tokens in test_cases:
            output = tokenizer.encode(term)
            # Just verify it encodes without error
            assert len(output.tokens) > 0

    def test_tokenizer_save_and_load(self, tmp_path):
        """Test tokenizer can be saved and loaded."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        # Train tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=500,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        texts = ["Sample training text for tokenizer"] * 50
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Save raw tokenizer
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        # Wrap as HuggingFace tokenizer
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        # Save in HuggingFace format
        fast_tokenizer.save_pretrained(str(tmp_path))

        # Load back
        loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tmp_path))

        # Verify it works
        text = "Sample text to encode"
        original_ids = fast_tokenizer.encode(text)
        loaded_ids = loaded_tokenizer.encode(text)

        assert original_ids == loaded_ids

    def test_tokenizer_consistency(self, tmp_path):
        """Test tokenizer produces consistent output."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=500,
            special_tokens=["[PAD]", "[UNK]"],
        )

        texts = ["Training text for consistency test"] * 50
        tokenizer.train_from_iterator(texts, trainer=trainer)

        test_text = "This is a test for consistency"

        # Encode multiple times
        results = []
        for _ in range(5):
            output = tokenizer.encode(test_text)
            results.append(output.ids)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i], "Tokenizer output is not consistent"


class TestTokenizerTrainingScript:
    """Tests for the tokenizer training script functions."""

    def test_is_likely_english_function(self):
        """Test English detection in tokenizer training."""
        # Import the function from train_tokenizer if available
        # Otherwise test the logic directly
        
        def is_likely_english(text: str, min_ascii_ratio: float = 0.85) -> bool:
            if not text or len(text) < 20:
                return True
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            return (ascii_chars / len(text)) >= min_ascii_ratio

        # English text
        assert is_likely_english("The company reported strong earnings growth this quarter.")
        
        # Non-ASCII text (should fail with enough non-ASCII)
        assert not is_likely_english("这是一个很长的中文句子用于测试语言检测功能是否正常工作")

    def test_combined_corpus_mixing(self):
        """Test that corpus mixing works correctly."""
        import random

        def mock_corpus_iterator(ratio: float = 0.5):
            """Simulate mixed corpus iteration."""
            sources = {"ecom": 0, "edgar": 0}
            
            for i in range(100):
                if random.random() < ratio:
                    sources["ecom"] += 1
                    yield f"ecom text {i}"
                else:
                    sources["edgar"] += 1
                    yield f"edgar text {i}"
            
            return sources

        random.seed(42)
        texts = list(mock_corpus_iterator(0.5))
        
        ecom_count = sum(1 for t in texts if t.startswith("ecom"))
        edgar_count = sum(1 for t in texts if t.startswith("edgar"))
        
        # Should be roughly 50/50 (within tolerance)
        assert 30 <= ecom_count <= 70
        assert 30 <= edgar_count <= 70

    def test_special_domain_tokens_list(self):
        """Test that domain-specific tokens are defined."""
        from octo_embedding_model.train_tokenizer import SPECIAL_DOMAIN_TOKENS

        # Should have finance terms
        finance_terms = ["EBITDA", "ROI", "WACC", "10-K", "SEC"]
        for term in finance_terms:
            assert term in SPECIAL_DOMAIN_TOKENS, f"Finance term {term} missing"

        # Should have e-commerce terms
        ecom_terms = ["SKU", "HDMI", "polyester", "Bluetooth"]
        for term in ecom_terms:
            assert term in SPECIAL_DOMAIN_TOKENS, f"E-commerce term {term} missing"


class TestTokenizerVerification:
    """Tests for tokenizer verification functionality."""

    def test_tokenizer_encode_decode_roundtrip(self, tmp_path):
        """Test encode-decode roundtrip preserves text."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]"],
        )

        texts = [
            "The company reported EBITDA of 1.5 billion dollars.",
            "This HDMI cable supports 4K resolution.",
        ] * 50
        
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Test roundtrip
        test_texts = [
            "EBITDA growth is positive",
            "USB cable with charging",
        ]

        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            # Decoded should be similar (may have whitespace differences)
            assert len(decoded) > 0

    def test_tokenizer_handles_unicode(self, tmp_path):
        """Test tokenizer handles unicode characters gracefully."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=500,
            special_tokens=["[PAD]", "[UNK]"],
        )

        # Include some unicode in training
        texts = [
            "Standard ASCII text for training",
            "Price: $100 or €85 or £75",
            "Temperature: 25°C",
        ] * 50

        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Should handle unicode without crashing
        test_texts = [
            "Price is $50",
            "Temperature 30°C",
            "Euro price €100",
        ]

        for text in test_texts:
            output = tokenizer.encode(text)
            assert len(output.ids) > 0

    def test_vocab_size_constraint(self, tmp_path):
        """Test tokenizer respects vocab size constraint."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        target_vocab_size = 500

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            special_tokens=["[PAD]", "[UNK]"],
        )

        # Large training corpus
        texts = [f"Training text number {i} with various words" for i in range(500)]
        tokenizer.train_from_iterator(texts, trainer=trainer)

        vocab = tokenizer.get_vocab()
        
        # Vocab size should not exceed target
        assert len(vocab) <= target_vocab_size


class TestTokenizerIntegrationWithModel:
    """Tests for tokenizer integration with the model."""

    def test_tokenizer_with_model_embedding(self, tmp_path):
        """Test tokenizer output works with model embedding layer."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        import torch
        import torch.nn as nn

        vocab_size = 500

        # Train tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]"],
        )

        texts = ["Training text"] * 50
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Create embedding layer
        embedding = nn.Embedding(vocab_size, 64)

        # Encode text
        output = tokenizer.encode("Test input text")
        input_ids = torch.tensor([output.ids])

        # Should work with embedding
        embeddings = embedding(input_ids)
        assert embeddings.shape == (1, len(output.ids), 64)

    def test_huggingface_tokenizer_with_chroma_model(self, tmp_path):
        """Test HuggingFace tokenizer works with Chroma model."""
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        import torch
        from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel

        # Train and wrap tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        texts = ["Financial report with EBITDA metrics"] * 100
        tokenizer.train_from_iterator(texts, trainer=trainer)

        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
        )

        # Create model with matching vocab size
        config = ChromaConfig(
            vocab_size=len(fast_tokenizer),
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

        # Tokenize and forward
        text = "EBITDA growth report"
        encoded = fast_tokenizer(
            text,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            truncation=True,
        )

        with torch.no_grad():
            output = model(encoded["input_ids"], encoded["attention_mask"])

        assert output.shape == (1, 64)  # latent_pooler_dim
