#!/usr/bin/env python3
"""
Mistral Model Fine-tuning Pipeline with QLoRA and NVFP4 Quantization
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True

@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class TrainConfig:
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
    max_seq_length: int = 2048 # Mistral supports up to 32k, but 2048 is efficient for fine-tuning
    bf16: bool = True
    save_strategy: str = "epoch"

# ============================================================================
# Data Preparation (The Fix)
# ============================================================================

def format_instruction(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Maps a single row to a formatted string.
    Mistral Format: <s>[INST] Instruction + Context [/INST] Response </s>
    """
    instruction = sample.get("instruction", "")
    response = sample.get("response", sample.get("output", ""))
    
    # Handle Context/Input being None or Empty
    context = sample.get("context", sample.get("input", ""))
    
    if context and isinstance(context, str) and len(context.strip()) > 0:
        full_input = f"{instruction}\n\nContext:\n{context}"
    else:
        full_input = instruction

    # Format strictly as a string
    text = f"<s>[INST] {full_input.strip()} [/INST] {response.strip()} </s>"
    
    return {"text": text}

def load_and_prepare_dataset(
    data_path: str,
    split: str = "train",
) -> Dataset:
    """
    Loads dataset and explicitly maps it to the 'text' format required by SFTTrainer.
    """
    logger.info(f"Loading dataset from: {data_path}")
    
    # Load dataset
    if data_path.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=data_path, split=split)
    else:
        dataset = load_dataset(data_path, split=split)

    logger.info(f"Loaded {len(dataset)} rows. Applying formatting...")

    # Explicitly map the formatting function BEFORE passing to Trainer.
    # This prevents the 'list object has no attribute endswith' error
    # by ensuring every row is strictly a string.
    formatted_dataset = dataset.map(
        format_instruction,
        remove_columns=dataset.column_names, # Remove raw columns to save RAM
        desc="Formatting dataset",
    )
    
    logger.info("Dataset formatting complete.")
    return formatted_dataset

# ============================================================================
# Training Logic
# ============================================================================

def train_model(
    model_config: ModelConfig,
    train_config: TrainConfig,
    quant_config: QuantizationConfig,
    dataset: Dataset,
):
    logger.info("Initializing Model & Tokenizer...")
    
    # 1. Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.load_in_4bit,
        bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
    )

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        device_map=model_config.device_map,
        attn_implementation="sdpa"  # Native PyTorch acceleration (No FlashAttn lib required)
    )

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for fp16/bf16 training

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # 5. Training Arguments (Using SFTConfig)
    sft_config = SFTConfig(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        logging_steps=train_config.logging_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        bf16=train_config.bf16,
        max_grad_norm=train_config.max_grad_norm,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        max_length=train_config.max_seq_length,
        dataset_text_field="text", # We pre-formatted the data to this field
        packing=False,             # Set to True if you want to pack multiple short examples into one sequence
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    logger.info("Starting Training...")
    trainer.train()

    # Save Adapter
    final_path = os.path.join(train_config.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Adapter saved to {final_path}")
    
    return final_path

# ============================================================================
# Merge & Quantize
# ============================================================================

def merge_and_quantize(
    model_id: str,
    adapter_path: str,
    output_dir: str,
    do_quantize: bool = False
):
    logger.info("Merging adapter into base model...")
    
    # 1. Merge
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    
    merged_path = os.path.join(output_dir, "merged_model")
    model.save_pretrained(merged_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"Merged model saved to {merged_path}")

    # 2. NVFP4 Quantization (Conditional)
    if do_quantize:
        try:
            import modelopt.torch.quantization as mtq
            logger.info("Starting NVFP4 Quantization...")
            
            # Calibration Data Generator
            calibration_data = ["Explain quantum physics", "Write a python script", "What is AI?"] * 20
            
            def calibration_loop(m):
                for txt in calibration_data:
                    inputs = tokenizer(txt, return_tensors="pt").to(m.device)
                    m(**inputs)

            quant_config = mtq.QuantizeConfig(quant_algorithm="fp4")
            model = mtq.quantize(model, config=quant_config, forward_loop=calibration_loop)
            
            nvfp4_path = os.path.join(output_dir, "nvfp4_model")
            mtq.save(model, os.path.join(nvfp4_path, "model.pth"))
            logger.info(f"NVFP4 Model saved to {nvfp4_path}")
            
        except ImportError:
            logger.error("nvidia-modelopt not found. Skipping NVFP4 export.")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-merge", action="store_true")
    args = parser.parse_args()

    # Configs
    model_conf = ModelConfig()
    train_conf = TrainConfig()
    quant_conf = QuantizationConfig()

    adapter_path = None

    if args.do_train:
        # Pre-process dataset explicitly to fix the "list" error
        dataset = load_and_prepare_dataset(args.data_path)
        adapter_path = train_model(model_conf, train_conf, quant_conf, dataset)
    
    if args.do_merge and adapter_path:
        merge_and_quantize(model_conf.model_id, adapter_path, train_conf.output_dir, do_quantize=True)

# # ============================================================================
# # CLI
# # ============================================================================

# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Mistral Fine-tuning Pipeline with QLoRA and NVFP4",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Train only
#   python train_mistral.py --data-path data.jsonl --do-train
  
#   # Full pipeline
#   python train_mistral.py --data-path data.jsonl --do-train --do-merge --do-quantize --do-evaluate
  
#   # Quantize existing model
#   python train_mistral.py --model-id ./merged_model --do-quantize
#         """
#     )
    
#     # Data arguments
#     parser.add_argument(
#         "--data-path",
#         type=str,
#         default="databricks/databricks-dolly-15k",
#         required=False,
#         help="Path to training data (JSON/JSONL file or HuggingFace dataset). Default: databricks/databricks-dolly-15k"
#     )
    
#     # Model arguments
#     parser.add_argument(
#         "--model-id",
#         type=str,
#         default="mistralai/Mistral-7B-Instruct-v0.3",
#         help="HuggingFace model ID or path"
#     )
    
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="./output",
#         help="Output directory for all artifacts"
#     )
    
#     # Pipeline steps
#     parser.add_argument("--do-train", action="store_true", help="Run training step")
#     parser.add_argument("--do-merge", action="store_true", help="Merge LoRA weights")
#     parser.add_argument("--do-quantize", action="store_true", help="Apply NVFP4 quantization")
#     parser.add_argument("--do-evaluate", action="store_true", help="Evaluate the model")
    
#     return parser.parse_args()


# def main():
#     """Main entry point."""
#     args = parse_args()
    
#     # Check if any pipeline step is selected
#     if not any([args.do_train, args.do_merge, args.do_quantize, args.do_evaluate]):
#         print("=" * 60)
#         print("No pipeline steps selected!")
#         print("=" * 60)
#         print("\nUse one or more of the following flags:")
#         print("  --do-train     Run QLoRA fine-tuning")
#         print("  --do-merge     Merge LoRA adapters into base model")
#         print("  --do-quantize  Apply NVFP4 quantization")
#         print("  --do-evaluate  Evaluate the model")
#         print("\nExample:")
#         print("  python train_mistral.py --do-train --do-merge")
#         print()
#         return
    
#     # Run pipeline
#     run_pipeline(
#         data_path=args.data_path,
#         model_id=args.model_id,
#         output_dir=args.output_dir,
#         do_train=args.do_train,
#         do_merge=args.do_merge,
#         do_quantize=args.do_quantize,
#         do_evaluate=args.do_evaluate,
#     )


# if __name__ == "__main__":
#     main()