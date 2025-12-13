"""
Phase 2: Contrastive Instruction Tuning.

This script implements the second phase of training for the Chroma-MoE embedding model.
Uses InfoNCE loss with in-batch negatives, hard negative mining, and GradCache
for memory-efficient large batch training.

Supports:
- DDP for multi-GPU cluster training
- In-batch negatives
- Hard negative mining (from ESCI substitutes)
- GradCache for simulating large batches
- W&B logging

Usage:
    # Single GPU (local testing)
    python train_phase2.py --config config.yaml
    
    # Multi-GPU (cluster)
    torchrun --nproc_per_node=8 train_phase2.py --config config.yaml
    
    # Resume from Phase 1 checkpoint
    python train_phase2.py --config config.yaml --pretrained-path ./checkpoints/checkpoint-100000.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from octo_embedding_model.data_loader import (
    ContrastiveCollator,
    create_combined_contrastive_dataset,
)
from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
from octo_embedding_model.trainer_utils import (
    DDPConfig,
    GradCache,
    InfoNCELoss,
    WandbLogger,
    cleanup_ddp,
    get_cosine_schedule_with_warmup,
    get_model_config,
    load_checkpoint,
    load_config,
    save_checkpoint,
    setup_ddp,
    wrap_model_ddp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_model_from_config(
    config: dict,
    tokenizer=None,
) -> ChromeMoEModel:
    """
    Create embedding model from configuration.
    
    If tokenizer is provided, uses its vocab size to ensure consistency.
    """
    model_config = get_model_config(config)
    
    # Use tokenizer's vocab size if available for consistency
    if tokenizer is not None:
        vocab_size = len(tokenizer)
        logger.info(f"Using tokenizer vocab size: {vocab_size}")
    else:
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
    )

    return ChromeMoEModel(chroma_config)


def gather_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Gather embeddings from all processes for in-batch negatives.
    
    This allows each process to see negatives from all other processes,
    effectively increasing the batch size for contrastive learning.
    """
    if not dist.is_initialized():
        return embeddings

    world_size = dist.get_world_size()
    if world_size == 1:
        return embeddings

    # Gather embeddings from all processes
    gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
    dist.all_gather(gathered, embeddings)

    # Concatenate (current process embeddings first for gradient flow)
    rank = dist.get_rank()
    gathered[rank] = embeddings  # Ensure gradients flow through current process
    return torch.cat(gathered, dim=0)


def train_epoch(
    model: ChromeMoEModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: InfoNCELoss,
    config: dict,
    epoch: int,
    global_step: int,
    wandb_logger: WandbLogger | None,
    device: torch.device,
    is_ddp: bool = False,
    grad_cache: GradCache | None = None,
) -> int:
    """Train for one epoch."""
    model.train()

    phase2_config = config.get("phase2", {})
    training_config = phase2_config.get("training", {})
    contrastive_config = phase2_config.get("contrastive", {})
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 1)
    log_steps = config.get("training", {}).get("logging", {}).get("log_steps", 10)
    hard_negative_weight = contrastive_config.get("hard_negative_weight", 1.0)

    rank = dist.get_rank() if is_ddp else 0
    accumulated_loss = 0.0
    accumulated_metrics = {
        "accuracy": 0.0,
        "temperature": 0.0,
        "mean_positive_sim": 0.0,
    }
    metrics_count = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=rank != 0,
    )

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        query_ids = batch["query_input_ids"].to(device)
        query_mask = batch["query_attention_mask"].to(device)
        positive_ids = batch["positive_input_ids"].to(device)
        positive_mask = batch["positive_attention_mask"].to(device)

        # Handle hard negatives if present
        negative_embeddings = None
        if "negative_input_ids" in batch:
            negative_ids = batch["negative_input_ids"].to(device)
            negative_mask = batch["negative_attention_mask"].to(device)

            with torch.no_grad():
                negative_embeddings = model(negative_ids, negative_mask)

        if grad_cache is not None:
            # Use GradCache for memory-efficient training
            loss, metrics = grad_cache.forward_backward(
                query_ids=query_ids,
                query_mask=query_mask,
                positive_ids=positive_ids,
                positive_mask=positive_mask,
                loss_fn=loss_fn,
            )
        else:
            # Standard forward pass
            query_embeddings = model(query_ids, query_mask)
            positive_embeddings = model(positive_ids, positive_mask)

            # Gather embeddings from all processes for in-batch negatives
            if is_ddp and contrastive_config.get("in_batch_negatives", True):
                positive_embeddings_gathered = gather_embeddings(positive_embeddings)
            else:
                positive_embeddings_gathered = positive_embeddings

            # Compute loss
            loss, metrics = loss_fn(
                query_embeddings=query_embeddings,
                positive_embeddings=positive_embeddings_gathered,
                negative_embeddings=negative_embeddings,
                hard_negative_weight=hard_negative_weight,
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

        accumulated_loss += loss.item() / grad_accum_steps
        for key in accumulated_metrics:
            if key in metrics:
                accumulated_metrics[key] += metrics[key]
        metrics_count += 1

        # Optimizer step after accumulation
        if (step + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % log_steps == 0 and rank == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = accumulated_loss

                log_metrics = {
                    "train/loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/epoch": epoch,
                }

                # Average accumulated metrics
                for key in accumulated_metrics:
                    if metrics_count > 0:
                        log_metrics[f"train/{key}"] = accumulated_metrics[key] / metrics_count

                logger.info(
                    f"Step {global_step} | Loss: {avg_loss:.4f} | "
                    f"Acc: {log_metrics.get('train/accuracy', 0):.4f} | LR: {lr:.2e}"
                )

                if wandb_logger:
                    wandb_logger.log(log_metrics, step=global_step)

                accumulated_loss = 0.0
                accumulated_metrics = {k: 0.0 for k in accumulated_metrics}
                metrics_count = 0

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{metrics.get('accuracy', 0):.4f}",
                step=global_step,
            )

    return global_step


def evaluate(
    model: ChromeMoEModel,
    dataloader: DataLoader,
    loss_fn: InfoNCELoss,
    device: torch.device,
    is_ddp: bool = False,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    rank = dist.get_rank() if is_ddp else 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=rank != 0):
            query_ids = batch["query_input_ids"].to(device)
            query_mask = batch["query_attention_mask"].to(device)
            positive_ids = batch["positive_input_ids"].to(device)
            positive_mask = batch["positive_attention_mask"].to(device)

            query_embeddings = model(query_ids, query_mask)
            positive_embeddings = model(positive_ids, positive_mask)

            loss, metrics = loss_fn(
                query_embeddings=query_embeddings,
                positive_embeddings=positive_embeddings,
            )

            total_loss += loss.item()
            total_accuracy += metrics.get("accuracy", 0)
            num_batches += 1

    # Aggregate across processes
    if is_ddp:
        metrics_tensor = torch.tensor(
            [total_loss, total_accuracy, num_batches],
            device=device,
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_accuracy, num_batches = metrics_tensor.tolist()

    return {
        "eval/loss": total_loss / max(num_batches, 1),
        "eval/accuracy": total_accuracy / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Contrastive Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="Path to Phase 1 pretrained checkpoint",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
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
            project=wandb_config.get("project", "chroma-moe-phase2"),
            config=config,
            run_name=wandb_config.get("run_name"),
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []) + ["phase2", "contrastive"],
        )

    # Load tokenizer FIRST (needed for model vocab size)
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_path = tokenizer_config.get("path", "./tokenizer")
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

    # Load pretrained weights from Phase 1 if provided
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        if rank == 0:
            logger.info(f"Loading pretrained weights from: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location="cpu")

        # Handle potential key mismatches (Phase 1 has MLM head, Phase 2 doesn't)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Filter out MLM head weights
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("mlm_head") and not k.startswith("backbone.")
        }
        # Try to load backbone weights if present
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in checkpoint.get("model_state_dict", checkpoint).items()
            if k.startswith("backbone.")
        }
        if backbone_state:
            model.load_state_dict(backbone_state, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params / 1e6:.2f}M")

    # Wrap with DDP
    if is_ddp:
        model = wrap_model_ddp(model, local_rank)

    # Create dataset
    if rank == 0:
        logger.info("Loading dataset...")

    debug_mode = config.get("active_model_profile") == "debug"
    dataset = create_combined_contrastive_dataset(
        config.get("data", {}),
        debug=debug_mode,
    )

    if rank == 0:
        logger.info(f"Dataset size: {len(dataset)} samples")

    # Create collator
    phase2_config = config.get("phase2", {})
    training_config = phase2_config.get("training", {})
    instruction_config = phase2_config.get("instruction", {})

    collator = ContrastiveCollator(
        tokenizer=tokenizer,
        max_length=training_config.get("max_seq_length", 512),
        query_template=instruction_config.get(
            "query_template", "Instruct: {instruction}\nQuery: {query}"
        ),
        passage_template=instruction_config.get("passage_template", "{passage}"),
    )

    # Create dataloader
    sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.get("per_device_batch_size", 4),
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Create loss function
    contrastive_config = phase2_config.get("contrastive", {})
    loss_fn = InfoNCELoss(
        temperature=contrastive_config.get("temperature", 0.05),
        learnable_temperature=contrastive_config.get("temperature_learnable", True),
    )
    loss_fn = loss_fn.to(device)

    # Create optimizer (include loss_fn parameters if temperature is learnable)
    optimizer_config = phase2_config.get("optimizer", {})
    params = list(model.parameters()) + list(loss_fn.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=optimizer_config.get("learning_rate", 1e-5),
        betas=(
            optimizer_config.get("beta1", 0.9),
            optimizer_config.get("beta2", 0.999),
        ),
        eps=optimizer_config.get("eps", 1e-8),
        weight_decay=optimizer_config.get("weight_decay", 0.01),
    )

    # Calculate training steps
    num_epochs = training_config.get("num_epochs", 3)
    steps_per_epoch = len(dataloader) // training_config.get("gradient_accumulation_steps", 1)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * training_config.get("warmup_ratio", 0.1))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1,
    )

    # GradCache for memory efficiency
    grad_cache = None
    grad_cache_config = phase2_config.get("grad_cache", {})
    if grad_cache_config.get("enabled", False):
        base_model = model.module if is_ddp else model
        grad_cache = GradCache(
            model=base_model,
            chunk_size=grad_cache_config.get("chunk_size", 16),
        )
        if rank == 0:
            logger.info("GradCache enabled for memory-efficient training")

    # Training loop
    output_dir = config.get("training", {}).get("output_dir", "./checkpoints")
    checkpoint_config = phase2_config.get("checkpoint", {})
    save_steps = checkpoint_config.get("save_steps", 1000)
    eval_steps = config.get("evaluation", {}).get("eval_steps", 1000)

    if rank == 0:
        logger.info("Starting contrastive training...")

    global_step = 0
    for epoch in range(num_epochs):
        if is_ddp:
            sampler.set_epoch(epoch)

        global_step = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            config=config,
            epoch=epoch,
            global_step=global_step,
            wandb_logger=wandb_logger,
            device=device,
            is_ddp=is_ddp,
            grad_cache=grad_cache,
        )

        # Evaluation at end of epoch
        if rank == 0:
            eval_metrics = evaluate(
                model=model.module if is_ddp else model,
                dataloader=dataloader,  # In practice, use separate val dataloader
                loss_fn=loss_fn,
                device=device,
                is_ddp=False,  # Eval on single process for simplicity
            )
            logger.info(f"Epoch {epoch} eval: {eval_metrics}")

            if wandb_logger:
                wandb_logger.log(eval_metrics, step=global_step)

        # Save checkpoint at end of epoch
        if rank == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                epoch=epoch,
                output_dir=os.path.join(output_dir, "phase2"),
                is_ddp=is_ddp,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    if rank == 0:
        final_path = os.path.join(output_dir, "phase2", "final_model.pt")
        if is_ddp:
            torch.save(model.module.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)
        logger.info(f"Saved final model: {final_path}")

        if wandb_logger:
            wandb_logger.log_model(os.path.join(output_dir, "phase2"), "chroma-moe-embedding")

    # Cleanup
    if wandb_logger:
        wandb_logger.finish()

    if is_ddp:
        cleanup_ddp()

    if rank == 0:
        logger.info("Contrastive training completed!")


if __name__ == "__main__":
    main()
