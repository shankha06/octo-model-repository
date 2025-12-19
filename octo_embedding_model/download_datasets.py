"""
Pre-download script for Phase 1 training datasets.

Downloads all required HuggingFace datasets to local Parquet files,
enabling offline training without network dependency.

Usage:
    # Download with default settings
    python download_datasets.py --config ../config.yaml

    # Download with custom settings
    python download_datasets.py --max-samples 100000 --output-dir ./data/custom_cache

    # Download specific dataset types only
    python download_datasets.py --datasets finance ecommerce
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from octo_embedding_model.trainer_utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_finance_datasets(
    output_dir: Path,
    max_samples: int = 3000000,
) -> int:
    """
    Download finance datasets to local Parquet file.
    
    Sources:
    - ashraq/financial-news
    - zeroshot/twitter-financial-news-topic
    """
    output_file = output_dir / "finance.parquet"
    
    # Skip if already exists
    if output_file.exists():
        existing = pq.read_metadata(output_file)
        logger.info(f"Finance data already exists ({existing.num_rows:,} rows), skipping...")
        return existing.num_rows
    
    finance_datasets = [
        ("ashraq/financial-news", ["headline", "title", "text"]),
        ("zeroshot/twitter-financial-news-topic", ["text", "tweet", "sentence"]),
    ]
    
    texts = []
    total_count = 0
    
    for dataset_id, text_fields in finance_datasets:
        if total_count >= max_samples:
            break
            
        try:
            logger.info(f"Downloading {dataset_id}...")
            dataset = load_dataset(dataset_id, split="train", streaming=True)
            
            # Calculate remaining samples needed
            remaining = max_samples - total_count
            
            for item in tqdm(dataset, desc=f"Processing {dataset_id}", total=remaining):
                text = None
                for field in text_fields:
                    if field in item and item[field]:
                        text = str(item[field])
                        break
                
                if text and len(text) > 20:
                    texts.append(text)
                    total_count += 1
                    
                if total_count >= max_samples:
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to load {dataset_id}: {e}")
            continue
    
    if texts:
        # Write to Parquet
        table = pa.table({"text": texts})
        pq.write_table(table, output_file, compression="snappy")
        logger.info(f"Saved {len(texts):,} finance samples to {output_file}")
    
    return len(texts)


def download_ecommerce_datasets(
    output_dir: Path,
    max_samples: int = 3000000,
) -> int:
    """
    Download e-commerce datasets to local Parquet file.
    
    Sources:
    - thebajajra/Ecom-niverse
    """
    output_file = output_dir / "ecommerce.parquet"
    
    # Skip if already exists
    if output_file.exists():
        existing = pq.read_metadata(output_file)
        logger.info(f"E-commerce data already exists ({existing.num_rows:,} rows), skipping...")
        return existing.num_rows
    
    texts = []
    
    try:
        logger.info("Downloading thebajajra/Ecom-niverse...")
        dataset = load_dataset("thebajajra/Ecom-niverse", split="train", streaming=True)
        
        for item in tqdm(dataset, desc="Processing Ecom-niverse", total=max_samples):
            for field in ["title", "description", "text", "product_name"]:
                if field in item and item[field]:
                    text = str(item[field])
                    if len(text) > 50:
                        texts.append(text)
                        break
            
            if len(texts) >= max_samples:
                break
                
    except Exception as e:
        logger.warning(f"Failed to load Ecom-niverse: {e}")
    
    if texts:
        table = pa.table({"text": texts})
        pq.write_table(table, output_file, compression="snappy")
        logger.info(f"Saved {len(texts):,} e-commerce samples to {output_file}")
    
    return len(texts)


def download_generic_datasets(
    output_dir: Path,
    max_samples: int = 3000000,
    subset: str = "sample-10BT",
) -> int:
    """
    Download generic/web datasets to local Parquet file.
    
    Sources:
    - HuggingFaceFW/fineweb-edu
    """
    output_file = output_dir / "generic.parquet"
    
    # Skip if already exists
    if output_file.exists():
        existing = pq.read_metadata(output_file)
        logger.info(f"Generic data already exists ({existing.num_rows:,} rows), skipping...")
        return existing.num_rows
    
    texts = []
    
    try:
        logger.info(f"Downloading HuggingFaceFW/fineweb-edu ({subset})...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            subset,
            split="train",
            streaming=True,
        )
        
        for item in tqdm(dataset, desc="Processing fineweb-edu", total=max_samples):
            text = item.get("text", "")
            if text and len(text) > 100:
                # Chunk long texts
                if len(text) > 5000:
                    for i in range(0, len(text), 2000):
                        chunk = text[i:i+2500]
                        if len(chunk) > 200:
                            texts.append(chunk)
                            if len(texts) >= max_samples:
                                break
                else:
                    texts.append(text)
            
            if len(texts) >= max_samples:
                break
                
    except Exception as e:
        logger.warning(f"Failed to load fineweb-edu: {e}")
    
    if texts:
        table = pa.table({"text": texts})
        pq.write_table(table, output_file, compression="snappy")
        logger.info(f"Saved {len(texts):,} generic samples to {output_file}")
    
    return len(texts)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download datasets for Phase 1 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (optional, for reading dataset settings)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/pretraining_cache",
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (overrides config)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["finance", "ecommerce", "generic", "all"],
        default=["all"],
        help="Which datasets to download",
    )
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    
    # Determine max samples
    if args.max_samples:
        max_samples = args.max_samples
    else:
        max_samples = config.get("data", {}).get("max_samples_per_dataset", 3000000)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading datasets to: {output_dir}")
    logger.info(f"Max samples per dataset: {max_samples:,}")
    
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["finance", "ecommerce", "generic"]
    
    total_samples = 0
    
    # Download each dataset type
    if "finance" in datasets_to_download:
        count = download_finance_datasets(output_dir, max_samples)
        total_samples += count
    
    if "ecommerce" in datasets_to_download:
        count = download_ecommerce_datasets(output_dir, max_samples)
        total_samples += count
    
    if "generic" in datasets_to_download:
        subset = config.get("data", {}).get("general", {}).get("fineweb_edu", {}).get("subset", "sample-10BT")
        count = download_generic_datasets(output_dir, max_samples, subset)
        total_samples += count
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Download complete!")
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Cache directory: {output_dir}")
    logger.info(f"\nTo use with training:")
    logger.info(f"  python train_phase1.py --config config.yaml --local-data {output_dir}")


if __name__ == "__main__":
    main()
