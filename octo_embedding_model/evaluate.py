"""
Evaluation script for Chroma-MoE embedding model.

Evaluates the trained model on:
- FinMTEB (finance domain)
- MTEB (general benchmark)
- Amazon ESCI (e-commerce retrieval)

Usage:
    # Evaluate on all benchmarks
    python evaluate.py --model-path ./checkpoints/phase2/final_model.pt --tokenizer-path ./models/tokenizer
    
    # Evaluate on specific benchmark
    python evaluate.py --model-path ./checkpoints/phase2/final_model.pt --benchmark finmteb
    
    # Use debug model for quick testing
    python evaluate.py --model-path ./checkpoints/phase2/final_model.pt --debug
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from octo_embedding_model.model_architecture import ChromaConfig, ChromeMoEModel
from octo_embedding_model.trainer_utils import get_model_config, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChromaEmbeddingModel:
    """Wrapper for Chroma-MoE model for evaluation."""
    
    def __init__(
        self,
        model: ChromeMoEModel,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
    
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode sentences to embeddings."""
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for i in iterator:
                batch = sentences[i:i + batch_size]
                
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                embeddings = self.model(input_ids, attention_mask)
                
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    tokenizer_path: str,
    device: str = "cuda",
) -> ChromaEmbeddingModel:
    """Load trained model from checkpoint."""
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    model_config = get_model_config(config)
    
    chroma_config = ChromaConfig(
        vocab_size=len(tokenizer),
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
        max_position_embeddings=model_config.get("max_position_embeddings", 512),
        latent_pooler_dim=model_config.get("latent_pooler_dim", 512),
    )
    
    model = ChromeMoEModel(chroma_config)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return ChromaEmbeddingModel(model, tokenizer, device=device)


def evaluate_finmteb(model: ChromaEmbeddingModel) -> dict[str, float]:
    """Evaluate on FinMTEB benchmark."""
    logger.info("Evaluating on FinMTEB...")
    
    try:
        from mteb import MTEB
        from datasets import load_dataset
        
        # FinMTEB tasks
        finmteb_tasks = [
            "FinancialPhrasebankClassification",
            "FiQA2018",
        ]
        
        results = {}
        
        for task_name in finmteb_tasks:
            try:
                evaluation = MTEB(tasks=[task_name])
                task_results = evaluation.run(model, output_folder=None)
                
                for result in task_results:
                    score = result.get("main_score", 0)
                    results[task_name] = score
                    logger.info(f"  {task_name}: {score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Could not evaluate {task_name}: {e}")
                results[task_name] = 0.0
        
        avg_score = np.mean(list(results.values())) if results else 0.0
        results["average"] = avg_score
        logger.info(f"FinMTEB Average: {avg_score:.4f}")
        
        return results
        
    except ImportError:
        logger.warning("MTEB not installed. Install with: pip install mteb")
        return {}


def evaluate_mteb(model: ChromaEmbeddingModel, tasks: list[str] | None = None) -> dict[str, float]:
    """Evaluate on MTEB benchmark."""
    logger.info("Evaluating on MTEB...")
    
    try:
        from mteb import MTEB
        
        # Default MTEB tasks for retrieval
        if tasks is None:
            tasks = [
                "ArguAna",
                "SCIDOCS",
                "NFCorpus",
            ]
        
        results = {}
        
        for task_name in tasks:
            try:
                evaluation = MTEB(tasks=[task_name])
                task_results = evaluation.run(model, output_folder=None)
                
                for result in task_results:
                    score = result.get("main_score", 0)
                    results[task_name] = score
                    logger.info(f"  {task_name}: {score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Could not evaluate {task_name}: {e}")
                results[task_name] = 0.0
        
        avg_score = np.mean(list(results.values())) if results else 0.0
        results["average"] = avg_score
        logger.info(f"MTEB Average: {avg_score:.4f}")
        
        return results
        
    except ImportError:
        logger.warning("MTEB not installed. Install with: pip install mteb")
        return {}


def evaluate_esci(model: ChromaEmbeddingModel, max_samples: int = 10000) -> dict[str, float]:
    """Evaluate on Amazon ESCI dataset."""
    logger.info("Evaluating on Amazon ESCI...")
    
    try:
        from datasets import load_dataset
        from sklearn.metrics import ndcg_score
        
        dataset = load_dataset("tasksource/esci", split="test")
        
        # Group by query
        query_to_products: dict[str, list[tuple[str, int]]] = {}
        
        for item in dataset:
            query = item.get("query", "")
            product = item.get("product_title", "") or item.get("product", "")
            label = item.get("esci_label", "") or item.get("label", "")
            
            if not query or not product:
                continue
            
            # Map labels to relevance scores
            relevance = {"exact": 3, "substitute": 2, "complement": 1, "irrelevant": 0}
            score = relevance.get(label.lower(), 0)
            
            if query not in query_to_products:
                query_to_products[query] = []
            query_to_products[query].append((product, score))
            
            if len(query_to_products) >= max_samples:
                break
        
        # Evaluate retrieval
        ndcg_scores = []
        mrr_scores = []
        
        for query, products in tqdm(query_to_products.items(), desc="ESCI Eval"):
            if len(products) < 2:
                continue
            
            product_texts = [p[0] for p in products]
            true_scores = [p[1] for p in products]
            
            # Encode query and products
            query_emb = model.encode([query], show_progress=False)
            product_embs = model.encode(product_texts, show_progress=False)
            
            # Compute similarities
            similarities = np.dot(product_embs, query_emb.T).flatten()
            
            # Compute NDCG
            if any(s > 0 for s in true_scores):
                ndcg = ndcg_score([true_scores], [similarities])
                ndcg_scores.append(ndcg)
            
            # Compute MRR
            ranked_indices = np.argsort(similarities)[::-1]
            for rank, idx in enumerate(ranked_indices, 1):
                if true_scores[idx] == 3:  # Exact match
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)
        
        results = {
            "NDCG@10": np.mean(ndcg_scores) if ndcg_scores else 0.0,
            "MRR": np.mean(mrr_scores) if mrr_scores else 0.0,
        }
        
        logger.info(f"  NDCG@10: {results['NDCG@10']:.4f}")
        logger.info(f"  MRR: {results['MRR']:.4f}")
        
        return results
        
    except Exception as e:
        logger.warning(f"Could not evaluate ESCI: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chroma-MoE Embedding Model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="./models/tokenizer",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["finmteb", "mteb", "esci", "all"],
        default="all",
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug mode with smaller sample sizes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    
    # Handle tokenizer path
    if not os.path.exists(args.tokenizer_path):
        logger.warning(f"Tokenizer not found at {args.tokenizer_path}, using bert-base-uncased")
        args.tokenizer_path = "bert-base-uncased"
    
    model = load_model_from_checkpoint(
        checkpoint_path=args.model_path,
        config_path=args.config,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )
    
    # Run evaluations
    all_results = {}
    
    if args.benchmark in ["finmteb", "all"]:
        all_results["finmteb"] = evaluate_finmteb(model)
    
    if args.benchmark in ["mteb", "all"]:
        all_results["mteb"] = evaluate_mteb(model)
    
    if args.benchmark in ["esci", "all"]:
        max_samples = 1000 if args.debug else 10000
        all_results["esci"] = evaluate_esci(model, max_samples=max_samples)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    
    for benchmark, results in all_results.items():
        logger.info(f"\n{benchmark.upper()}:")
        for metric, score in results.items():
            logger.info(f"  {metric}: {score:.4f}")
    
    return all_results


if __name__ == "__main__":
    main()
