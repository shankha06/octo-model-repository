#!/usr/bin/env python3
"""
Tests for the Mistral training pipeline.

Run with: pytest tests/test_train_mistral.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from dataclasses import asdict

# Import functions and classes to test
import sys
sys.path.insert(0, '.')
from octo_embedding_model.train_mistral import (
    format_mistral_instruction,
    formatting_prompts_func,
    load_and_prepare_dataset,
    ModelConfig,
    QuantizationConfig,
    LoRAConfig,
    TrainConfig,
    NVFP4Config,
    get_bnb_config,
)


# ============================================================================
# Test Configuration Dataclasses
# ============================================================================

class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.torch_dtype == "bfloat16"
        assert config.device_map == "auto"
        assert config.trust_remote_code is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_id="custom/model",
            torch_dtype="float16",
            device_map="cuda:0"
        )
        assert config.model_id == "custom/model"
        assert config.torch_dtype == "float16"
        assert config.device_map == "cuda:0"


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""
    
    def test_default_values(self):
        """Test default quantization settings for QLoRA."""
        config = QuantizationConfig()
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_compute_dtype == "bfloat16"
        assert config.bnb_4bit_use_double_quant is True


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""
    
    def test_default_values(self):
        """Test default LoRA settings."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""
    
    def test_default_values(self):
        """Test default training settings."""
        config = TrainConfig()
        assert config.output_dir == "./results"
        assert config.num_train_epochs == 1
        assert config.per_device_train_batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.bf16 is True
        assert config.fp16 is False
        assert config.max_seq_length == 512


class TestNVFP4Config:
    """Tests for NVFP4Config dataclass."""
    
    def test_default_values(self):
        """Test default NVFP4 settings."""
        config = NVFP4Config()
        assert config.quant_algorithm == "nvfp4"
        assert config.policy == "per_tensor_symmetric"
        assert config.num_calibration_samples == 512


# ============================================================================
# Test Data Formatting Functions
# ============================================================================

class TestFormatMistralInstruction:
    """Tests for format_mistral_instruction function."""
    
    def test_dolly_format_with_context(self):
        """Test formatting Dolly-style data with context."""
        sample = {
            "instruction": "Summarize the following text.",
            "context": "The quick brown fox jumps over the lazy dog.",
            "response": "A fox jumps over a dog."
        }
        result = format_mistral_instruction(sample)
        
        assert "text" in result
        assert "<s>[INST]" in result["text"]
        assert "[/INST]" in result["text"]
        assert "</s>" in result["text"]
        assert "Summarize the following text." in result["text"]
        assert "Context:" in result["text"]
        assert "The quick brown fox" in result["text"]
        assert "A fox jumps over a dog." in result["text"]
    
    def test_dolly_format_without_context(self):
        """Test formatting Dolly-style data without context."""
        sample = {
            "instruction": "What is 2+2?",
            "context": "",
            "response": "4"
        }
        result = format_mistral_instruction(sample)
        
        assert "text" in result
        assert "Context:" not in result["text"]
        assert "What is 2+2?" in result["text"]
        assert "4" in result["text"]
    
    def test_alpaca_format(self):
        """Test formatting Alpaca-style data."""
        sample = {
            "instruction": "Translate to French.",
            "input": "Hello world",
            "output": "Bonjour le monde"
        }
        result = format_mistral_instruction(sample)
        
        assert "text" in result
        assert "Translate to French." in result["text"]
        assert "Hello world" in result["text"]
        assert "Bonjour le monde" in result["text"]
    
    def test_empty_context_is_ignored(self):
        """Test that empty or whitespace-only context is ignored."""
        sample = {
            "instruction": "Test instruction",
            "context": "   ",
            "response": "Test response"
        }
        result = format_mistral_instruction(sample)
        
        # Should not include "Context:" since context is just whitespace
        assert "Context:" not in result["text"]


class TestFormattingPromptsFunc:
    """Tests for formatting_prompts_func batch function."""
    
    def test_batch_formatting(self):
        """Test batch formatting of multiple examples."""
        examples = {
            "instruction": ["Q1", "Q2"],
            "context": ["C1", ""],
            "response": ["R1", "R2"]
        }
        results = formatting_prompts_func(examples)
        
        assert len(results) == 2
        assert all("<s>[INST]" in r for r in results)
        assert all("[/INST]" in r for r in results)
        assert all("</s>" in r for r in results)
    
    def test_empty_batch(self):
        """Test handling of empty batch."""
        examples = {
            "instruction": [],
            "context": [],
            "response": []
        }
        results = formatting_prompts_func(examples)
        assert results == []
    
    def test_handles_alpaca_format(self):
        """Test that batch function handles Alpaca format too."""
        examples = {
            "instruction": ["Instruction 1"],
            "input": ["Input 1"],
            "output": ["Output 1"]
        }
        results = formatting_prompts_func(examples)
        
        assert len(results) == 1
        assert "Instruction 1" in results[0]


# ============================================================================
# Test BitsAndBytes Config
# ============================================================================

class TestGetBnbConfig:
    """Tests for get_bnb_config function."""
    
    def test_creates_valid_config(self):
        """Test that get_bnb_config creates a valid BitsAndBytesConfig."""
        quant_config = QuantizationConfig()
        bnb_config = get_bnb_config(quant_config)
        
        # Check it's the right type
        from transformers import BitsAndBytesConfig
        assert isinstance(bnb_config, BitsAndBytesConfig)
    
    def test_respects_config_values(self):
        """Test that the config values are properly applied."""
        quant_config = QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype="float16"
        )
        bnb_config = get_bnb_config(quant_config)
        
        assert bnb_config.load_in_4bit is True
        assert bnb_config.bnb_4bit_quant_type == "fp4"


# ============================================================================
# Test Dataset Loading (with mocking)
# ============================================================================

class TestLoadAndPrepareDataset:
    """Tests for load_and_prepare_dataset function."""
    
    @patch('octo_embedding_model.train_mistral.load_dataset')
    def test_loads_huggingface_dataset(self, mock_load_dataset):
        """Test loading a HuggingFace dataset."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.column_names = ["instruction", "context", "response"]
        mock_load_dataset.return_value = mock_dataset
        
        dataset, formatting_func = load_and_prepare_dataset(
            "databricks/databricks-dolly-15k",
            use_formatting_func=True
        )
        
        mock_load_dataset.assert_called_once()
        assert formatting_func is not None
    
    @patch('octo_embedding_model.train_mistral.load_dataset')
    def test_loads_json_file(self, mock_load_dataset):
        """Test loading a JSON file."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset.column_names = ["instruction", "input", "output"]
        mock_load_dataset.return_value = mock_dataset
        
        dataset, _ = load_and_prepare_dataset(
            "data.json",
            use_formatting_func=True
        )
        
        mock_load_dataset.assert_called_with(
            "json", data_files="data.json", split="train", streaming=False
        )
    
    @patch('octo_embedding_model.train_mistral.load_dataset')
    def test_loads_jsonl_file(self, mock_load_dataset):
        """Test loading a JSONL file."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset.column_names = ["instruction", "input", "output"]
        mock_load_dataset.return_value = mock_dataset
        
        dataset, _ = load_and_prepare_dataset(
            "data.jsonl",
            use_formatting_func=True
        )
        
        mock_load_dataset.assert_called_with(
            "json", data_files="data.jsonl", split="train", streaming=False
        )


# ============================================================================
# Integration Tests (Optional - requires GPU)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIntegration:
    """Integration tests that require GPU."""
    
    def test_model_loading_smoke_test(self):
        """Smoke test for model loading (skipped without GPU)."""
        # This would be a minimal integration test
        pass


# ============================================================================
# CLI Argument Tests
# ============================================================================

class TestCLIArguments:
    """Tests for CLI argument parsing."""
    
    def test_parse_args_defaults(self):
        """Test that default arguments are set correctly."""
        from octo_embedding_model.train_mistral import parse_args
        
        with patch('sys.argv', ['train_mistral.py']):
            args = parse_args()
            
            assert args.data_path == "databricks/databricks-dolly-15k"
            assert args.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
            assert args.output_dir == "./output"
            assert args.do_train is False
            assert args.do_merge is False
            assert args.do_quantize is False
            assert args.do_evaluate is False
    
    def test_parse_args_with_flags(self):
        """Test argument parsing with flags."""
        from octo_embedding_model.train_mistral import parse_args
        
        with patch('sys.argv', ['train_mistral.py', '--do-train', '--do-merge']):
            args = parse_args()
            
            assert args.do_train is True
            assert args.do_merge is True
            assert args.do_quantize is False
    
    def test_parse_args_custom_paths(self):
        """Test custom path arguments."""
        from octo_embedding_model.train_mistral import parse_args
        
        with patch('sys.argv', [
            'train_mistral.py',
            '--data-path', 'custom_data.jsonl',
            '--model-id', 'custom/model',
            '--output-dir', './custom_output'
        ]):
            args = parse_args()
            
            assert args.data_path == "custom_data.jsonl"
            assert args.model_id == "custom/model"
            assert args.output_dir == "./custom_output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
