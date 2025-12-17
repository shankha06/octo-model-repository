#!/usr/bin/env python3
"""
Mistral Model Fine-tuning Pipeline with QLoRA and NVFP4 Quantization

This script provides a production-ready pipeline for:
1. Data preparation and formatting for Mistral instruction format
2. QLoRA fine-tuning with 4-bit quantization
3. LoRA adapter merging
4. NVFP4 quantization for TensorRT-LLM deployment
5. Model evaluation using lm-eval-harness

Usage:
    python train_mistral.py --config config.yaml
    python train_mistral.py --model-id mistralai/Mistral-7B-Instruct-v0.3 --data-path data.jsonl
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class QuantizationConfig:
    """Quantization configuration for QLoRA training."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"
    ])


@dataclass
class TrainConfig:
    """Training configuration parameters."""
    output_dir: str = "./results"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    logging_steps: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    max_seq_length: int = 512
    bf16: bool = True
    fp16: bool = False
    group_by_length: bool = True
    save_strategy: str = "epoch"


@dataclass
class NVFP4Config:
    """NVFP4 quantization configuration."""
    quant_algorithm: str = "nvfp4"
    policy: str = "per_tensor_symmetric"
    num_calibration_samples: int = 512


# ============================================================================
# Data Preparation
# ============================================================================

def format_mistral_instruction(sample: dict) -> dict:
    """
    Formats data into Mistral's instruction format.
    
    Supports both formats:
        - Dolly: {"instruction": "...", "context": "...", "response": "..."}
        - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    
    Output format:
        <s>[INST] instruction [/INST] response </s>
    """
    instruction = sample.get("instruction", "")
    
    # Handle both Dolly (context/response) and Alpaca (input/output) formats
    context = sample.get("context", sample.get("input", ""))
    response = sample.get("response", sample.get("output", ""))
    
    # Combine instruction and context if context exists
    if context and len(str(context).strip()) > 0:
        full_input = f"{instruction}\n\nContext:\n{context}"
    else:
        full_input = instruction
    
    # Format in Mistral style
    formatted_text = f"<s>[INST] {full_input.strip()} [/INST] {response.strip()} </s>"
    
    return {"text": formatted_text}


def formatting_prompts_func(examples: dict) -> list:
    """
    Batch formatting function for SFTTrainer.
    Formats examples into Mistral's instruction format.
    
    Args:
        examples: Dictionary with 'instruction', 'context', 'response' columns
    
    Returns:
        List of formatted text strings
    """
    output_texts = []
    
    # Handle batch processing
    instructions = examples.get('instruction', [])
    contexts = examples.get('context', examples.get('input', [''] * len(instructions)))
    responses = examples.get('response', examples.get('output', []))
    
    for instruction, context, response in zip(instructions, contexts, responses):
        # Combine instruction and context if context exists
        if context and len(str(context).strip()) > 0:
            full_input = f"{instruction}\n\nContext:\n{context}"
        else:
            full_input = instruction
        
        # Format using Mistral's special tokens
        text = f"<s>[INST] {full_input.strip()} [/INST] {response.strip()} </s>"
        output_texts.append(text)
    
    return output_texts


def load_and_prepare_dataset(
    data_path: str = "databricks/databricks-dolly-15k",
    split: str = "train",
    streaming: bool = False,
    use_formatting_func: bool = True,
) -> tuple:
    """
    Load and prepare dataset for training.
    
    Args:
        data_path: Path to dataset (JSON/JSONL file or HuggingFace dataset name)
                   Default: "databricks/databricks-dolly-15k"
        split: Dataset split to load
        streaming: Whether to use streaming mode for large datasets
        use_formatting_func: If True, return raw dataset + formatting func for SFTTrainer
                            If False, return pre-formatted dataset with 'text' column
    
    Returns:
        If use_formatting_func: Tuple of (dataset, formatting_prompts_func)
        Else: Formatted dataset with 'text' column
    """
    logger.info(f"Loading dataset from: {data_path}")
    
    # Determine dataset type and load
    if data_path.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=data_path, split=split, streaming=streaming)
    else:
        dataset = load_dataset(data_path, split=split, streaming=streaming)
    
    logger.info(f"Dataset loaded with {len(dataset) if not streaming else 'streaming'} samples")
    
    # For SFTTrainer, we can use formatting_func directly (more memory efficient)
    if use_formatting_func:
        logger.info("Returning dataset with formatting function for SFTTrainer")
        return dataset, formatting_prompts_func
    
    # Pre-format the dataset (alternative approach)
    logger.info("Formatting dataset for Mistral instruction format...")
    dataset = dataset.map(format_mistral_instruction, remove_columns=dataset.column_names)
    
    logger.info(f"Dataset prepared with {len(dataset) if not streaming else 'streaming'} samples")
    return dataset, None


# ============================================================================
# Model Loading
# ============================================================================

def get_bnb_config(config: QuantizationConfig) -> BitsAndBytesConfig:
    """Create BitsAndBytes configuration for 4-bit quantization."""
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def load_model_for_training(
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
) -> tuple:
    """
    Load model and tokenizer for QLoRA training.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_config.model_id}")
    
    # Create quantization config
    bnb_config = get_bnb_config(quant_config)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def apply_lora(model, lora_config: LoRAConfig):
    """Apply LoRA adapters to the model."""
    logger.info("Applying LoRA adapters...")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=lora_config.target_modules,
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, peft_config


# ============================================================================
# Training
# ============================================================================

def train_model(
    model,
    tokenizer,
    dataset: Dataset,
    peft_config: LoraConfig,
    train_config: TrainConfig,
    formatting_func=None,
) -> str:
    """
    Train the model using SFTTrainer.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for the model
        dataset: Training dataset
        peft_config: LoRA configuration
        train_config: Training configuration
        formatting_func: Optional formatting function for SFTTrainer
    
    Returns:
        Path to saved model
    """
    logger.info("Starting training...")
    
    # Create SFT config (trl 0.26+ API)
    sft_config = SFTConfig(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        logging_steps=train_config.logging_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        max_grad_norm=train_config.max_grad_norm,
        warmup_ratio=train_config.warmup_ratio,
        group_by_length=train_config.group_by_length,
        lr_scheduler_type=train_config.lr_scheduler_type,
        save_strategy=train_config.save_strategy,
        report_to="none",  # Disable wandb by default
        max_seq_length=train_config.max_seq_length,
        dataset_text_field="text" if formatting_func is None else None,
    )
    
    # Create trainer with appropriate data handling
    if formatting_func is not None:
        # Use formatting function (more memory efficient for large datasets)
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            formatting_func=formatting_func,
            processing_class=tokenizer,
            args=sft_config,
        )
    else:
        # Use pre-formatted dataset with 'text' column
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=sft_config,
        )
    
    # Train
    trainer.train()
    
    # Save model
    output_path = os.path.join(train_config.output_dir, "final_model")
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Model saved to: {output_path}")
    return output_path


# ============================================================================
# LoRA Merging
# ============================================================================

def merge_lora_weights(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
) -> str:
    """
    Merge LoRA adapter weights back into the base model.
    
    Args:
        base_model_id: HuggingFace model ID or path to base model
        adapter_path: Path to trained LoRA adapter
        output_path: Path to save merged model
    
    Returns:
        Path to merged model
    """
    logger.info("Merging LoRA weights into base model...")
    
    # Load base model in high precision
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    
    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Merged model saved to: {output_path}")
    return output_path


# ============================================================================
# NVFP4 Quantization
# ============================================================================

def quantize_to_nvfp4(
    model_path: str,
    output_path: str,
    calibration_data: Optional[List[str]] = None,
    config: Optional[NVFP4Config] = None,
) -> str:
    """
    Quantize model to NVFP4 format for TensorRT-LLM deployment.
    
    Args:
        model_path: Path to merged model
        output_path: Path to save quantized model
        calibration_data: List of calibration prompts
        config: NVFP4 quantization configuration
    
    Returns:
        Path to quantized checkpoint
    """
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        logger.error("nvidia-modelopt not installed. Install with: pip install nvidia-modelopt")
        raise
    
    config = config or NVFP4Config()
    
    logger.info("Loading model for NVFP4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Default calibration data if not provided
    if calibration_data is None:
        calibration_data = [
            "Explain the theory of relativity in simple terms.",
            "Write a Python function to sort a list using quicksort.",
            "What are the main causes of climate change?",
            "Summarize the key points of machine learning.",
            "How does a transformer neural network work?",
        ] * 100  # Repeat to get ~500 samples
    
    logger.info(f"Using {len(calibration_data)} calibration samples")
    
    # Define calibration loop
    def calibration_loop(model):
        for text in calibration_data[:config.num_calibration_samples]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)
            with torch.no_grad():
                model(**inputs)
    
    # Configure quantization
    quant_config = mtq.QuantizeConfig(
        quant_algorithm=config.quant_algorithm,
        policy=config.policy,
    )
    
    # Apply quantization
    logger.info("Applying NVFP4 quantization...")
    model = mtq.quantize(model, config=quant_config, forward_loop=calibration_loop)
    
    # Test generation
    logger.info("Testing quantized model...")
    test_input = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**test_input, max_new_tokens=20)
    logger.info(f"Test output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    
    # Save quantized model
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = os.path.join(output_path, "model_nvfp4.pth")
    mtq.save(model, checkpoint_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"NVFP4 model saved to: {output_path}")
    return output_path


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model_path: str,
    tasks: List[str] = None,
    num_fewshot: int = 0,
    batch_size: int = 4,
) -> dict:
    """
    Evaluate model using lm-eval-harness.
    
    Args:
        model_path: Path to model
        tasks: List of evaluation tasks
        num_fewshot: Number of few-shot examples
        batch_size: Evaluation batch size
    
    Returns:
        Evaluation results dictionary
    """
    try:
        from lm_eval import evaluator
    except ImportError:
        logger.error("lm-eval not installed. Install with: pip install lm-eval")
        raise
    
    tasks = tasks or ["hellaswag", "arc_easy"]
    
    logger.info(f"Evaluating model on tasks: {tasks}")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    
    logger.info("Evaluation Results:")
    for task, metrics in results.get("results", {}).items():
        logger.info(f"  {task}: {metrics}")
    
    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
    data_path: str = "databricks/databricks-dolly-15k",
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output_dir: str = "./output",
    do_train: bool = True,
    do_merge: bool = True,
    do_quantize: bool = False,
    do_evaluate: bool = False,
):
    """
    Run the complete fine-tuning pipeline.
    
    Args:
        data_path: Path to training data
        model_id: HuggingFace model ID
        output_dir: Base output directory
        do_train: Whether to run training
        do_merge: Whether to merge LoRA weights
        do_quantize: Whether to apply NVFP4 quantization
        do_evaluate: Whether to evaluate the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize configs
    model_config = ModelConfig(model_id=model_id)
    quant_config = QuantizationConfig()
    lora_config = LoRAConfig()
    train_config = TrainConfig(output_dir=os.path.join(output_dir, "training"))
    
    adapter_path = None
    merged_path = None
    nvfp4_path = None
    
    # Step 1: Training
    if do_train:
        logger.info("=" * 60)
        logger.info("STEP 1: Training")
        logger.info("=" * 60)
        
        # Load dataset (returns tuple: dataset, formatting_func)
        dataset, formatting_func = load_and_prepare_dataset(data_path)
        
        # Load model
        model, tokenizer = load_model_for_training(model_config, quant_config)
        
        # Apply LoRA
        model, peft_config = apply_lora(model, lora_config)
        
        # Train
        adapter_path = train_model(
            model, tokenizer, dataset, peft_config, train_config,
            formatting_func=formatting_func
        )
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Step 2: Merge LoRA weights
    if do_merge:
        logger.info("=" * 60)
        logger.info("STEP 2: Merging LoRA Weights")
        logger.info("=" * 60)
        
        adapter_path = adapter_path or os.path.join(output_dir, "training", "final_model")
        merged_path = os.path.join(output_dir, "merged")
        
        merge_lora_weights(model_id, adapter_path, merged_path)
        
        torch.cuda.empty_cache()
    
    # Step 3: NVFP4 Quantization
    if do_quantize:
        logger.info("=" * 60)
        logger.info("STEP 3: NVFP4 Quantization")
        logger.info("=" * 60)
        
        merged_path = merged_path or os.path.join(output_dir, "merged")
        nvfp4_path = os.path.join(output_dir, "nvfp4")
        
        quantize_to_nvfp4(merged_path, nvfp4_path)
        
        torch.cuda.empty_cache()
    
    # Step 4: Evaluation
    if do_evaluate:
        logger.info("=" * 60)
        logger.info("STEP 4: Evaluation")
        logger.info("=" * 60)
        
        eval_path = nvfp4_path or merged_path or adapter_path
        if eval_path:
            evaluate_model(eval_path)
        else:
            logger.warning("No model path available for evaluation")
    
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mistral Fine-tuning Pipeline with QLoRA and NVFP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train only
  python train_mistral.py --data-path data.jsonl --do-train
  
  # Full pipeline
  python train_mistral.py --data-path data.jsonl --do-train --do-merge --do-quantize --do-evaluate
  
  # Quantize existing model
  python train_mistral.py --model-id ./merged_model --do-quantize
        """
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="databricks/databricks-dolly-15k",
        required=False,
        help="Path to training data (JSON/JSONL file or HuggingFace dataset). Default: databricks/databricks-dolly-15k"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model ID or path"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for all artifacts"
    )
    
    # Pipeline steps
    parser.add_argument("--do-train", action="store_true", help="Run training step")
    parser.add_argument("--do-merge", action="store_true", help="Merge LoRA weights")
    parser.add_argument("--do-quantize", action="store_true", help="Apply NVFP4 quantization")
    parser.add_argument("--do-evaluate", action="store_true", help="Evaluate the model")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if any pipeline step is selected
    if not any([args.do_train, args.do_merge, args.do_quantize, args.do_evaluate]):
        print("=" * 60)
        print("No pipeline steps selected!")
        print("=" * 60)
        print("\nUse one or more of the following flags:")
        print("  --do-train     Run QLoRA fine-tuning")
        print("  --do-merge     Merge LoRA adapters into base model")
        print("  --do-quantize  Apply NVFP4 quantization")
        print("  --do-evaluate  Evaluate the model")
        print("\nExample:")
        print("  python train_mistral.py --do-train --do-merge")
        print()
        return
    
    # Run pipeline
    run_pipeline(
        data_path=args.data_path,
        model_id=args.model_id,
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_merge=args.do_merge,
        do_quantize=args.do_quantize,
        do_evaluate=args.do_evaluate,
    )


if __name__ == "__main__":
    main()