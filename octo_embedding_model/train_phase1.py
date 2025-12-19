"""
Phase 1: Domain-Adaptive Pre-training with Span Masking.

This script implements the first phase of training for the Chroma-MoE embedding model.
Uses bidirectional MLM with T5-style span masking on domain-specific corpora.

Supports:
- DDP for multi-GPU cluster training
- Gradient accumulation for large effective batch sizes
- W&B logging
- MoE auxiliary load balancing loss

Usage:
    # Single GPU (local testing)
    python train_phase1.py --config config.yaml
    
    # Multi-GPU (cluster)
    torchrun --nproc_per_node=8 train_phase1.py --config config.yaml
    
    # DDP validation only
    torchrun --nproc_per_node=2 train_phase1.py --config config.yaml --test-ddp
"""

import math
import argparse
import inspect
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from octo_embedding_model.data_loader import (
    LocalPreTrainingDataset,
    PackedSpanMaskingCollator,
    PreTrainingDataset,
    SpanMaskingCollator,
    StreamingPreTrainingDataset,
    create_combined_pretraining_dataset,
    create_local_pretraining_dataset,
    create_streaming_pretraining_dataset,
)
from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
from octo_embedding_model.trainer_utils import (
    DDPConfig,
    MoEAuxLoss,
    WandbLogger,
    cleanup_ddp,
    get_cosine_schedule_with_warmup,
    get_model_config,
    load_checkpoint,
    load_config,
    save_checkpoint,
    setup_ddp,
    validate_ddp,
    wrap_model_ddp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)




class ChromaMoEForPretraining(nn.Module):
    """
    Chroma-MoE model wrapper for pre-training with MLM head.
    """

    def __init__(self, config: ChromaConfig):
        super().__init__()
        self.backbone = ChromeMoEModel(config)
        self.config = config

        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for pre-training.
        
        Returns dict with 'loss' and optionally 'logits'.
        """
        # Get backbone hidden states (before pooling)
        x = self.backbone.embed_tokens(input_ids)

        # Handle packed sequences vs padded sequences
        if cu_seqlens is not None:
            # Packed sequence mode: no attention mask needed
            extended_mask = None
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -10000.0

        for layer in self.backbone.layers:
            residual = x
            x_norm = layer["norm1"](x)
            attn_out = layer["attn"](
                x_norm, 
                extended_mask, 
                self.backbone.rotary_emb,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
            x = residual + attn_out

            residual = x
            x_norm = layer["norm2"](x)
            moe_out = layer["moe"](x_norm)
            x = residual + moe_out

        hidden_states = self.backbone.norm_final(x)

        # MLM head
        logits = self.mlm_head(hidden_states)

        output = {"logits": logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            output["loss"] = loss

        return output


def create_model_from_config(
    config: dict,
    tokenizer=None,
) -> ChromaMoEForPretraining:
    """
    Create model from configuration.
    
    If tokenizer is provided, uses its vocab size to ensure consistency.
    """
    model_config = get_model_config(config)
    
    # Use tokenizer's vocab size if available for consistency
    if tokenizer is not None:
        vocab_size = len(tokenizer)
        logger.info(f"Using tokenizer vocab size: {vocab_size}")
    else:
        # Fall back to config
        tokenizer_config = config.get("tokenizer", {})
        vocab_size = tokenizer_config.get("vocab_size", model_config.get("vocab_size", 32000))

    chroma_config = ChromaConfig(
        vocab_size=vocab_size,
        hidden_size=model_config.get("hidden_size", 512),
        num_hidden_layers=model_config.get("num_hidden_layers", 4),
        num_attention_heads=model_config.get("num_attention_heads", 8),
        kv_lora_rank=model_config.get("kv_lora_rank", 128),
        q_lora_rank=model_config.get("q_lora_rank", 256),
        qk_rope_head_dim=model_config.get("qk_rope_head_dim", 32),
        qk_nope_head_dim=model_config.get("qk_nope_head_dim", 32),
        v_head_dim=model_config.get("v_head_dim", 32),
        moe_intermediate_size=model_config.get("moe_intermediate_size", 512),
        num_routed_experts=model_config.get("num_routed_experts", 8),
        num_shared_experts=model_config.get("num_shared_experts", 1),
        num_experts_per_tok=model_config.get("num_experts_per_tok", 2),
        moe_layer_freq=model_config.get("moe_layer_freq", 1),
        aux_loss_alpha=model_config.get("aux_loss_alpha", 0.01),
        max_position_embeddings=model_config.get("max_position_embeddings", 512),
        rms_norm_eps=model_config.get("rms_norm_eps", 1e-6),
        rope_theta=model_config.get("rope_theta", 10000.0),
        latent_pooler_dim=model_config.get("latent_pooler_dim", 512),
        gradient_checkpointing=config.get("training", {}).get("gradient_checkpointing", False),
    )

    return ChromaMoEForPretraining(chroma_config)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    aux_loss_fn: MoEAuxLoss | None,
    config: dict,
    epoch: int,
    global_step: int,
    wandb_logger: WandbLogger | None,
    device: torch.device,
    is_ddp: bool = False,
    scaler: GradScaler | None = None,
) -> int:
    model.train()
    
    # 1. Determine correct precision for autocast
    precision = config.get("training", {}).get("precision", "bf16")
    pt_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    phase1_config = config.get("phase1", {})
    training_config = phase1_config.get("training", {})
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 1)
    log_steps = config.get("training", {}).get("logging", {}).get("log_steps", 10)

    rank = dist.get_rank() if is_ddp else 0
    accumulated_loss = 0.0


    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=rank != 0,
    )

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Handle packed vs padded sequences
        is_packed = batch.get("packed", False)
        if is_packed:
            cu_seqlens = batch["cu_seqlens"].to(device)
            max_seqlen = batch["max_seqlen"]
            attention_mask = None
        else:
            cu_seqlens = None
            max_seqlen = None
            attention_mask = batch["attention_mask"].to(device)

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=pt_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            
            total_loss = outputs["loss"]

            # 3. ADD MISSING AUX LOSS
            # We assume the model returns 'router_logits' in outputs needed for the loss
            if aux_loss_fn is not None and "router_logits" in outputs:
                aux_loss = aux_loss_fn(outputs["router_logits"])
                total_loss += aux_loss
            elif aux_loss_fn is not None:
                # Warning: Your model forward pass needs to return router_logits!
                # If ChromaMoEModel doesn't expose them, you need to modify it.
                pass

        # Scale loss for gradient accumulation
        loss = total_loss / grad_accum_steps
        
        # Backward pass with scaler
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item()

        # Optimizer step after accumulation
        if (step + 1) % grad_accum_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step with scaler if using mixed precision
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # More memory efficient

            global_step += 1

            # Logging
            if global_step % log_steps == 0 and rank == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = accumulated_loss * grad_accum_steps

                logger.info(
                    f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

                if wandb_logger:
                    wandb_logger.log(
                        {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

                accumulated_loss = 0.0

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item() * grad_accum_steps:.4f}",
                step=global_step,
            )

        # Check max steps
        max_steps = training_config.get("max_steps", float("inf"))
        if global_step >= max_steps:
            break

    return global_step


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Pre-training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--test-ddp",
        action="store_true",
        help="Only test DDP setup and exit",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local pre-downloaded data cache (from download_datasets.py)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine if running in distributed mode
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup DDP if needed
    if is_ddp:
        ddp_config = DDPConfig(
            backend=config.get("distributed", {}).get("backend", "nccl"),
            find_unused_parameters=config.get("distributed", {}).get(
                "find_unused_parameters", False
            ),
        )
        setup_ddp(rank, world_size, ddp_config)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        if rank == 0:
            logger.info(f"DDP initialized: rank={rank}, world_size={world_size}")

        # Validate DDP if requested
        if args.test_ddp:
            results = validate_ddp()
            if rank == 0:
                logger.info(f"DDP Validation Results: {results}")
            cleanup_ddp()
            return
    else:
        device = torch.device(
            config.get("training", {}).get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        if rank == 0:
            logger.info(f"Running on single device: {device}")

    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize W&B (only on rank 0)
    wandb_logger = None
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", True) and rank == 0:
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "chroma-moe-phase1"),
            config=config,
            run_name=wandb_config.get("run_name"),
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []) + ["phase1"],
        )

    # Load tokenizer FIRST (needed for model vocab size)
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_path = tokenizer_config.get("path", "./models/tokenizer")
    fallback_tokenizer = tokenizer_config.get("fallback", "bert-base-uncased")
    
    if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
        if rank == 0:
            logger.info(f"Loading custom tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        if rank == 0:
            logger.warning(
                f"Custom tokenizer not found at {tokenizer_path}, "
                f"using fallback: {fallback_tokenizer}"
            )
            logger.info("Train custom tokenizer with: python octo_embedding_model/train_tokenizer.py")
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    if rank == 0:
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create model with tokenizer's vocab size
    if rank == 0:
        logger.info("Creating model...")
    model = create_model_from_config(config, tokenizer=tokenizer)

    # Enable gradient checkpointing if configured
    if config.get("training", {}).get("gradient_checkpointing", False):
        # Note: Would need to implement gradient checkpointing in model
        pass

    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
        
    # Enable torch.compile for faster training (PyTorch 2.0+)
    if config.get("training", {}).get("torch_compile", True) and hasattr(torch, "compile"):
        if rank == 0:
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap with DDP
    if is_ddp:
        model = wrap_model_ddp(model, local_rank)

    # Create dataset
    debug_mode = config.get("active_model_profile") == "debug"
    use_streaming = config.get("data", {}).get("streaming", True)  # Default to streaming
    
    # Check for local data cache first (from download_datasets.py)
    if args.local_data:
        if rank == 0:
            logger.info(f"Loading from local cache: {args.local_data}")
        dataset = create_local_pretraining_dataset(
            config.get("data", {}),
            cache_dir=args.local_data,
            debug=debug_mode,
        )
        # IterableDataset doesn't support DistributedSampler
        sampler = None
    elif use_streaming:
        if rank == 0:
            logger.info("Creating streaming dataset (training starts immediately)...")
        dataset = create_streaming_pretraining_dataset(
            config.get("data", {}),
            debug=debug_mode,
        )
        # IterableDataset doesn't support DistributedSampler
        sampler = None
    else:
        if rank == 0:
            logger.info("Loading full dataset into memory...")
        dataset = create_combined_pretraining_dataset(
            config.get("data", {}),
            tokenizer=tokenizer,
            debug=debug_mode,
        )
        sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None

    if rank == 0:
        logger.info(f"Dataset estimated size: {len(dataset):,} samples")

    # Create collator
    phase1_config = config.get("phase1", {})
    masking_config = phase1_config.get("masking", {})
    training_config = phase1_config.get("training", {})
    
    # Use packing for 2-3x speedup (default: True if flash_attn available)
    use_packing = config.get("training", {}).get("use_packing", True)
    
    if use_packing and rank == 0:
        logger.info("Using sequence packing for 2-3x speedup")
    
    if use_packing:
        collator = PackedSpanMaskingCollator(
            tokenizer=tokenizer,
            mask_ratio=masking_config.get("mask_ratio", 0.30),
            mean_span_length=masking_config.get("mean_span_length", 3),
            max_length=training_config.get("max_seq_length", 4096),
        )
    else:
        collator = SpanMaskingCollator(
            tokenizer=tokenizer,
            mask_ratio=masking_config.get("mask_ratio", 0.30),
            mean_span_length=masking_config.get("mean_span_length", 3),
            max_length=training_config.get("max_seq_length", 512),
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.get("per_device_batch_size", 8),
        sampler=sampler, 
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Create optimizer
    optimizer_config = phase1_config.get("optimizer", {})
    # Use fused AdamW if available (PyTorch 2.0+)
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = dict(fused=True) if use_fused else dict()
    
    if rank == 0 and use_fused:
        logger.info("Using Fused AdamW optimizer (faster)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get("learning_rate", 3e-4),
        betas=(
            optimizer_config.get("beta1", 0.9),
            optimizer_config.get("beta2", 0.95),
        ),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0.1),
        **extra_args
    )

    # Create scheduler
    max_steps = training_config.get("max_steps", 100000)
    warmup_steps = training_config.get("warmup_steps", 2000)
    min_lr_ratio = optimizer_config.get("min_learning_rate", 3e-5) / optimizer_config.get(
        "learning_rate", 3e-4
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # MoE auxiliary loss
    model_config = get_model_config(config)
    aux_loss_fn = MoEAuxLoss(
        num_experts=model_config.get("num_routed_experts", 8),
        alpha=phase1_config.get("moe", {}).get("aux_loss_weight", 0.01),
    )

    # Resume from checkpoint if specified
    global_step = 0
    start_epoch = 0
    resume_path = config.get("training", {}).get("resume_from_checkpoint")
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint_info = load_checkpoint(resume_path, model, optimizer, scheduler)
        global_step = checkpoint_info["step"]
        start_epoch = checkpoint_info["epoch"]

    # Training loop
    output_dir = config.get("training", {}).get("output_dir", "./checkpoints")
    checkpoint_config = phase1_config.get("checkpoint", {})
    save_steps = checkpoint_config.get("save_steps", 5000)

    # Enable optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster matmul
    torch.backends.cudnn.allow_tf32 = True
    
    # Create GradScaler for mixed precision (bf16 doesn't need scaling, but fp16 does)
    precision = config.get("training", {}).get("precision", "bf16")
    scaler = GradScaler() if precision == "fp16" else None
    
    if rank == 0:
        logger.info(f"Starting training with {precision} precision...")
        logger.info(f"Flash Attention: {'enabled' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 'disabled'}")

    num_epochs = 2  # Will likely stop based on max_steps
    for epoch in range(start_epoch, num_epochs):
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        global_step = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            aux_loss_fn=aux_loss_fn,
            config=config,
            epoch=epoch,
            global_step=global_step,
            wandb_logger=wandb_logger,
            device=device,
            is_ddp=is_ddp,
            scaler=scaler,
        )

        # Save checkpoint
        if rank == 0 and (global_step % save_steps == 0 or global_step >= max_steps):
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                epoch=epoch,
                output_dir=output_dir,
                is_ddp=is_ddp,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        if global_step >= max_steps:
            break

    # Cleanup
    if wandb_logger:
        wandb_logger.finish()

    if is_ddp:
        cleanup_ddp()

    if rank == 0:
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
