"""
Tests for data loading utilities.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch


class TestSpanMaskingCollator:
    """Tests for SpanMaskingCollator."""

    def test_collator_output_shape(self):
        """Test that collator produces correct output shapes."""
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
            {"text": "This is a test sentence for span masking."},
            {"text": "Another example sentence to test the collator."},
        ]

        output = collator(batch)

        assert "input_ids" in output
        assert "attention_mask" in output
        assert "labels" in output

        assert output["input_ids"].shape == (2, 64)
        assert output["attention_mask"].shape == (2, 64)
        assert output["labels"].shape == (2, 64)

    def test_collator_applies_masking(self):
        """Test that masking is applied to inputs."""
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
            {"text": "This is a longer test sentence that should have some tokens masked."},
        ]

        output = collator(batch)

        # Check that some tokens are masked (replaced with mask token id)
        mask_token_id = tokenizer.mask_token_id
        assert (output["input_ids"] == mask_token_id).any(), "No mask tokens found"

        # Check that labels have -100 for non-masked positions
        assert (output["labels"] == -100).any(), "No ignored positions in labels"


class TestContrastiveCollator:
    """Tests for ContrastiveCollator."""

    def test_collator_output_keys(self):
        """Test that collator produces correct output keys."""
        from transformers import AutoTokenizer
        from octo_embedding_model.data_loader import ContrastiveCollator

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collator = ContrastiveCollator(
            tokenizer=tokenizer,
            max_length=64,
        )

        batch = [
            {"query": "What is machine learning?", "positive": "Machine learning is a subset of AI."},
            {"query": "How does NLP work?", "positive": "NLP uses algorithms to process text."},
        ]

        output = collator(batch)

        assert "query_input_ids" in output
        assert "query_attention_mask" in output
        assert "positive_input_ids" in output
        assert "positive_attention_mask" in output

    def test_collator_with_hard_negatives(self):
        """Test collator with hard negatives."""
        from transformers import AutoTokenizer
        from octo_embedding_model.data_loader import ContrastiveCollator

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collator = ContrastiveCollator(
            tokenizer=tokenizer,
            max_length=64,
        )

        batch = [
            {
                "query": "What is machine learning?",
                "positive": "Machine learning is a subset of AI.",
                "hard_negatives": ["Deep learning uses neural networks.", "Statistics is math."],
            },
        ]

        output = collator(batch)

        assert "negative_input_ids" in output
        assert "negative_attention_mask" in output
        assert output["negative_input_ids"].shape[0] == 2  # 2 hard negatives

    def test_collator_with_instruction(self):
        """Test collator applies instruction template."""
        from transformers import AutoTokenizer
        from octo_embedding_model.data_loader import ContrastiveCollator

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collator = ContrastiveCollator(
            tokenizer=tokenizer,
            max_length=128,
            query_template="Instruct: {instruction}\nQuery: {query}",
        )

        batch = [
            {
                "query": "machine learning",
                "positive": "ML is AI.",
                "instruction": "Find relevant documents",
            },
        ]

        output = collator(batch)

        # Decode to check instruction is included
        decoded = tokenizer.decode(output["query_input_ids"][0], skip_special_tokens=True)
        assert "instruct" in decoded.lower()


class TestPreTrainingDataset:
    """Tests for PreTrainingDataset."""

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        from octo_embedding_model.data_loader import PreTrainingDataset

        texts = ["text1", "text2", "text3"]
        dataset = PreTrainingDataset(texts)

        assert len(dataset) == 3

    def test_dataset_getitem(self):
        """Test dataset returns correct items."""
        from octo_embedding_model.data_loader import PreTrainingDataset

        texts = ["text1", "text2", "text3"]
        dataset = PreTrainingDataset(texts)

        item = dataset[0]
        assert item == {"text": "text1"}


class TestContrastiveDataset:
    """Tests for ContrastiveDataset."""

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        from octo_embedding_model.data_loader import ContrastiveDataset

        queries = ["q1", "q2"]
        positives = ["p1", "p2"]
        dataset = ContrastiveDataset(queries, positives)

        assert len(dataset) == 2

    def test_dataset_getitem(self):
        """Test dataset returns correct items."""
        from octo_embedding_model.data_loader import ContrastiveDataset

        queries = ["q1", "q2"]
        positives = ["p1", "p2"]
        hard_negatives = [["n1", "n2"], ["n3"]]
        instructions = ["i1", "i2"]

        dataset = ContrastiveDataset(queries, positives, hard_negatives, instructions)

        item = dataset[0]
        assert item["query"] == "q1"
        assert item["positive"] == "p1"
        assert item["hard_negatives"] == ["n1", "n2"]
        assert item["instruction"] == "i1"

    def test_dataset_assertion_mismatch(self):
        """Test dataset raises error on length mismatch."""
        from octo_embedding_model.data_loader import ContrastiveDataset

        queries = ["q1", "q2"]
        positives = ["p1"]  # Mismatch

        with pytest.raises(AssertionError):
            ContrastiveDataset(queries, positives)


class TestDatasetLoaders:
    """Tests for dataset loader functions."""

    @patch("octo_embedding_model.data_loader.load_dataset")
    def test_load_esci_mock(self, mock_load_dataset):
        """Test ESCI loading with mocked data."""
        from octo_embedding_model.data_loader import load_esci

        # Mock dataset
        mock_data = [
            {"query": "red shoes", "product_title": "Red Running Shoes", "esci_label": "exact"},
            {"query": "red shoes", "product_title": "Pink Sandals", "esci_label": "substitute"},
            {"query": "blue jacket", "product_title": "Blue Winter Jacket", "esci_label": "exact"},
        ]
        mock_load_dataset.return_value = mock_data

        dataset = load_esci(max_samples=10)

        assert len(dataset) > 0
        assert hasattr(dataset, "queries")
        assert hasattr(dataset, "positives")

    @patch("octo_embedding_model.data_loader.load_dataset")
    def test_load_convfinqa_mock(self, mock_load_dataset):
        """Test ConvFinQA loading with mocked data."""
        from octo_embedding_model.data_loader import load_convfinqa

        mock_data = [
            {"question": "What is the revenue?", "context": "Revenue was $1M."},
            {"question": "What is the profit?", "context": "Profit was $500K."},
        ]
        mock_load_dataset.return_value = mock_data

        dataset = load_convfinqa(max_samples=10)

        assert len(dataset) == 2
        assert dataset.queries[0] == "What is the revenue?"
