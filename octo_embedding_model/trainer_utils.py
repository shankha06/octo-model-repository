"""
Training utilities for Chroma-MoE embedding model.

Includes:
- DDP setup/validation
- InfoNCE loss
- GradCache for memory-efficient large batch training
- W&B integration
- Learning rate schedulers
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DDPConfig:
    """Configuration for Distributed Data Parallel training."""

    backend: str = "nccl"
    find_unused_parameters: bool = True  # Required for MoE models where not all experts are used
    gradient_as_bucket_view: bool = True


def setup_ddp(rank: int, world_size: int, config: DDPConfig | None = None) -> None:
    """
    Initialize DDP process group.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        config: DDP configuration
    """
    if config is None:
        config = DDPConfig()

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    dist.init_process_group(
        backend=config.backend,
        rank=rank,
        world_size=world_size,
    )

    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def validate_ddp() -> dict[str, Any]:
    """
    Validate that DDP is working correctly.
    
    Returns:
        dict with validation results
    """
    results = {
        "is_initialized": dist.is_initialized(),
        "rank": -1,
        "world_size": -1,
        "backend": None,
        "all_reduce_test": False,
    }

    if not dist.is_initialized():
        return results

    results["rank"] = dist.get_rank()
    results["world_size"] = dist.get_world_size()
    results["backend"] = dist.get_backend()

    # Test all-reduce
    try:
        tensor = torch.ones(1).cuda() * dist.get_rank()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(dist.get_world_size()))
        results["all_reduce_test"] = abs(tensor.item() - expected) < 0.001
    except Exception as e:
        results["all_reduce_error"] = str(e)

    return results


def wrap_model_ddp(
    model: nn.Module,
    device_id: int,
    config: DDPConfig | None = None,
) -> DDP:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: Model to wrap
        device_id: CUDA device ID
        config: DDP configuration
        
    Returns:
        DDP-wrapped model
    """
    if config is None:
        config = DDPConfig()

    return DDP(
        model.to(device_id),
        device_ids=[device_id],
        find_unused_parameters=config.find_unused_parameters,
        gradient_as_bucket_view=config.gradient_as_bucket_view,
    )


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
    
    Supports:
    - In-batch negatives
    - Hard negatives
    - Learnable temperature
    """

    def __init__(
        self,
        temperature: float = 0.05,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.learnable_temperature = learnable_temperature

        if learnable_temperature:
            # Initialize log temperature for numerical stability
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

    @property
    def temp(self) -> torch.Tensor:
        """Get current temperature value."""
        if self.learnable_temperature:
            return self.log_temperature.exp()
        return self.temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None = None,
        hard_negative_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute InfoNCE loss.
        
        Args:
            query_embeddings: [batch_size, embed_dim]
            positive_embeddings: [batch_size, embed_dim]
            negative_embeddings: Optional [batch_size * num_negs, embed_dim]
            hard_negative_weight: Weight for hard negatives vs in-batch negatives
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        batch_size = query_embeddings.size(0)

        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Compute similarity with positives
        positive_sim = torch.sum(query_embeddings * positive_embeddings, dim=1)  # [batch_size]

        # In-batch negatives: all other positives in the batch
        # [batch_size, batch_size]
        in_batch_sim = torch.matmul(query_embeddings, positive_embeddings.t())

        # Create mask to exclude positive pairs on diagonal
        mask = torch.eye(batch_size, dtype=torch.bool, device=query_embeddings.device)

        # Compute logits
        logits = in_batch_sim / self.temp

        # Add hard negatives if provided
        if negative_embeddings is not None and negative_embeddings.size(0) > 0:
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
            # Reshape if needed: [batch_size, num_negs, embed_dim]
            if negative_embeddings.dim() == 2:
                num_negs = negative_embeddings.size(0) // batch_size
                if num_negs > 0:
                    negative_embeddings = negative_embeddings.view(batch_size, num_negs, -1)
                    # [batch_size, num_negs]
                    hard_neg_sim = torch.bmm(
                        query_embeddings.unsqueeze(1),
                        negative_embeddings.transpose(1, 2),
                    ).squeeze(1)
                    hard_neg_logits = hard_neg_sim / self.temp * hard_negative_weight
                    logits = torch.cat([logits, hard_neg_logits], dim=1)

        # Labels: positive is always at position i for query i (diagonal)
        labels = torch.arange(batch_size, device=query_embeddings.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy for logging
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "temperature": self.temp.item(),
            "mean_positive_sim": positive_sim.mean().item(),
        }

        return loss, metrics


class MoEAuxLoss(nn.Module):
    """
    Auxiliary load balancing loss for Mixture of Experts.
    
    Ensures experts are utilized evenly to prevent expert collapse.
    """

    def __init__(self, num_experts: int, alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha

    def forward(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Args:
            router_logits: Router output logits [batch*seq, num_experts]
            expert_indices: Selected expert indices [batch*seq, top_k]
            
        Returns:
            Auxiliary loss value
        """
        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Compute fraction of tokens assigned to each expert
        # One-hot encode selected experts and average
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # Sum over top_k
        tokens_per_expert = expert_mask.mean(dim=0)

        # Compute mean probability for each expert
        mean_prob_per_expert = router_probs.mean(dim=0)

        # Load balancing loss: minimize product of fraction and probability
        # This encourages uniform distribution
        aux_loss = self.alpha * self.num_experts * (tokens_per_expert * mean_prob_per_expert).sum()

        return aux_loss


class GradCache:
    """
    Gradient caching for memory-efficient large batch contrastive learning.
    
    Based on: "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup"
    
    Splits batch into chunks, computes forward pass for all chunks,
    then accumulates gradients to simulate large batch training.
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 16,
    ):
        self.model = model
        self.chunk_size = chunk_size

    def _get_embeddings_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get embeddings in chunks without gradients."""
        embeddings = []
        batch_size = input_ids.size(0)

        for i in range(0, batch_size, self.chunk_size):
            chunk_ids = input_ids[i : i + self.chunk_size]
            chunk_mask = attention_mask[i : i + self.chunk_size]

            with torch.no_grad():
                chunk_emb = self.model(chunk_ids, chunk_mask)
                embeddings.append(chunk_emb)

        return embeddings

    def forward_backward(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_mask: torch.Tensor,
        loss_fn: InfoNCELoss,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute forward and backward pass with gradient caching.
        
        Args:
            query_ids: Query input IDs
            query_mask: Query attention mask
            positive_ids: Positive document input IDs
            positive_mask: Positive document attention mask
            loss_fn: Loss function to use
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with metrics
        """
        # Step 1: Get all embeddings without gradients
        query_embeddings = self._get_embeddings_chunked(query_ids, query_mask)
        positive_embeddings = self._get_embeddings_chunked(positive_ids, positive_mask)

        # Stack embeddings
        all_query_emb = torch.cat(query_embeddings, dim=0)
        all_positive_emb = torch.cat(positive_embeddings, dim=0)

        # Enable gradients for loss computation
        all_query_emb.requires_grad_(True)
        all_positive_emb.requires_grad_(True)

        # Step 2: Compute loss on full batch
        loss, metrics = loss_fn(all_query_emb, all_positive_emb)

        # Step 3: Get gradients w.r.t embeddings
        loss.backward()
        query_grads = all_query_emb.grad.split(self.chunk_size)
        positive_grads = all_positive_emb.grad.split(self.chunk_size)

        # Step 4: Backward through model chunks with cached gradients
        batch_size = query_ids.size(0)

        for i, (q_grad, p_grad) in enumerate(zip(query_grads, positive_grads)):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, batch_size)

            # Query backward
            query_chunk_ids = query_ids[start:end]
            query_chunk_mask = query_mask[start:end]
            query_chunk_emb = self.model(query_chunk_ids, query_chunk_mask)
            query_chunk_emb.backward(q_grad)

            # Positive backward
            pos_chunk_ids = positive_ids[start:end]
            pos_chunk_mask = positive_mask[start:end]
            pos_chunk_emb = self.model(pos_chunk_ids, pos_chunk_mask)
            pos_chunk_emb.backward(p_grad)

        return loss.detach(), metrics


class WandbLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(
        self,
        project: str,
        config: dict[str, Any],
        run_name: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and self._wandb_available()
        self._run = None

        if self.enabled:
            import wandb

            self._run = wandb.init(
                project=project,
                config=config,
                name=run_name,
                entity=entity,
                tags=tags or [],
                save_code=True,
            )

    @staticmethod
    def _wandb_available() -> bool:
        """Check if wandb is available."""
        try:
            import wandb

            return True
        except ImportError:
            return False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to W&B."""
        if self.enabled and self._run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def log_model(self, model_path: str, name: str = "model") -> None:
        """Log model artifact to W&B."""
        if self.enabled and self._run is not None:
            import wandb

            artifact = wandb.Artifact(name, type="model")
            artifact.add_dir(model_path)
            self._run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled and self._run is not None:
            import wandb

            wandb.finish()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of max LR
        
    Returns:
        Learning rate scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Get model configuration based on active profile.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Model configuration dictionary
    """
    profile = config.get("active_model_profile", "debug")
    return config.get("model", {}).get(profile, config.get("model", {}).get("debug", {}))


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    output_dir: str,
    is_ddp: bool = False,
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        step: Current training step
        epoch: Current epoch
        output_dir: Directory to save checkpoint
        is_ddp: Whether model is wrapped in DDP
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get model state dict (unwrap DDP if needed)
    if is_ddp:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
    }

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}.pt")
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        
    Returns:
        Checkpoint metadata (step, epoch)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
    }
