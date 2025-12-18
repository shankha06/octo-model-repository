"""
Custom BPE Tokenizer Training for Chroma-MoE Embedding Model.

Trains a domain-specific BPE tokenizer on:
- Ecom-niverse (e-commerce data)
- EDGAR (SEC financial filings)

This ensures domain-specific terms like "EBITDA", "RoI", "HDMI", "polyester"
are single tokens rather than fragmented sub-words.

Usage:
    # Train tokenizer with default settings
    python train_tokenizer.py --output-dir ./models/tokenizer
    
    # Train with specific vocab size
    python train_tokenizer.py --vocab-size 65536 --output-dir ./models/tokenizer
    
    # Debug mode with small sample
    python train_tokenizer.py --debug --output-dir ./models/tokenizer
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Domain-specific terms that should be single tokens
SPECIAL_DOMAIN_TOKENS = [
    # Finance terms
    "EBITDA", "EBIT", "ROI", "ROE", "ROA", "ROIC", "WACC", "DCF", "NPV", "IRR",
    "P/E", "EPS", "P/B", "P/S", "CAGR", "YoY", "QoQ", "MoM", "TTM", "LTM",
    "10-K", "10-Q", "8-K", "S-1", "DEF14A", "13F", "Form4",
    "GDP", "CPI", "PPI", "FOMC", "Fed", "SEC", "GAAP", "IFRS",
    "amortization", "depreciation", "accrual", "leverage", "liquidity",
    "covenant", "derivative", "hedging", "arbitrage", "volatility",
    "bullish", "bearish", "dividend", "equity", "liability",
    # E-commerce terms
    "SKU", "UPC", "ASIN", "EAN", "GTIN", "ISBN", "MPN",
    "HDMI", "USB-C", "USB-A", "Bluetooth", "WiFi", "NFC", "RFID",
    "polyester", "cotton", "nylon", "spandex", "viscose", "rayon",
    "waterproof", "breathable", "antimicrobial", "hypoallergenic",
    "eco-friendly", "sustainable", "organic", "vegan", "cruelty-free",
    "XS", "XXL", "XXXL", "petite", "plus-size", "maternity",
    "refurbished", "pre-owned", "open-box", "clearance",
    "bestseller", "new-arrival", "limited-edition", "exclusive",
    
    # --- Product Identifiers & Codes ---
    "SKU", "UPC", "ASIN", "EAN", "GTIN", "ISBN", "MPN", "ISSN", "JAN", "PLU",
    "QR-Code", "Barcode", "Model-No", "Serial-No", "Batch-No",

    # --- Electronics: Connectivity & Ports ---
    "HDMI", "USB-C", "USB-A", "Micro-USB", "Mini-USB", "Lightning", "Thunderbolt",
    "DisplayPort", "VGA", "DVI", "Ethernet", "RJ45", "Aux", "Jack",
    "Bluetooth", "WiFi", "NFC", "RFID", "GPS", "LTE", "5G", "4G", "3G",
    "Zigbee", "Z-Wave", "LoRaWAN", "Infrared", "Qi-Charging", "MagSafe",

    # --- Electronics: Display & Audio ---
    "4K", "8K", "HD", "FHD", "QHD", "UHD", "HDR", "HDR10", "Dolby-Vision",
    "OLED", "QLED", "AMOLED", "IPS", "LCD", "LED", "Micro-LED", "Mini-LED",
    "Retina", "Gorilla-Glass", "Anti-glare", "Touchscreen",
    "60Hz", "90Hz", "120Hz", "144Hz", "240Hz", "360Hz",
    "Dolby-Atmos", "DTS", "Hi-Res", "Noise-Cancelling", "ANC", "Transparency-Mode",

    # --- Electronics: Specs & Hardware ---
    "CPU", "GPU", "RAM", "ROM", "SSD", "HDD", "NVMe", "SATA", "PCIe", "M.2",
    "DDR3", "DDR4", "DDR5", "LPDDR", "GDDR",
    "mAh", "Wh", "Li-Ion", "Li-Po", "Fast-Charge", "Power-Delivery", "GaN",
    "IP67", "IP68", "Water-Resistant", "Dust-Proof", "Shock-Proof",
    "Dual-SIM", "eSIM", "Unlocked", "Jailbroken", "Rooted",
    "Windows", "macOS", "iOS", "Android", "Linux", "ChromeOS",

    # --- Fashion: Fabrics & Materials ---
    "polyester", "cotton", "nylon", "spandex", "viscose", "rayon", "elastane",
    "denim", "leather", "suede", "silk", "linen", "cashmere", "wool", "merino",
    "fleece", "velvet", "corduroy", "flannel", "chiffon", "satin", "lace",
    "canvas", "mesh", "neoprene", "microfiber", "gore-tex", "thermal",
    "gold-plated", "sterling-silver", "stainless-steel", "titanium", "carbide",

    # --- Fashion: Attributes & Cuts ---
    "waterproof", "breathable", "moisture-wicking", "quick-dry", "wrinkle-free",
    "slim-fit", "regular-fit", "loose-fit", "oversized", "skinny", "tapered",
    "bootcut", "straight-leg", "high-rise", "mid-rise", "low-rise",
    "v-neck", "crew-neck", "turtleneck", "mock-neck", "halter", "strapless",
    "long-sleeve", "short-sleeve", "sleeveless", "cap-sleeve", "raglan",

    # --- Sizes & Demographics ---
    "XS", "S", "M", "L", "XL", "XXL", "XXXL", "XXXXL", "One-Size",
    "petite", "plus-size", "maternity", "tall", "big-and-tall",
    "infant", "toddler", "kids", "junior", "unisex", "mens", "womens",

    # --- Beauty & Cosmetics ---
    "antimicrobial", "hypoallergenic", "non-comedogenic", "dermatologist-tested",
    "sulfate-free", "paraben-free", "phthalate-free", "silicone-free",
    "oil-free", "fragrance-free", "alcohol-free", "gluten-free",
    "vegan", "cruelty-free", "organic", "natural", "clean-beauty",
    "SPF", "PA+++", "UVA", "UVB", "broad-spectrum",
    "retinol", "hyaluronic-acid", "vitamin-c", "niacinamide", "salicylic-acid",
    "matte", "glossy", "dewy", "shimmer", "satin", "metallic",

    # --- Home & Appliances ---
    "HEPA", "BPA-free", "dishwasher-safe", "microwave-safe", "oven-safe",
    "induction", "convection", "inverter", "energy-star",
    "BTU", "CFM", "RPM", "decibel", "voltage", "wattage", "lumen", "kelvin",
    "smart-home", "IoT", "Alexa", "Google-Assistant", "Siri", "HomeKit",

    # --- Sales, Condition & Logistics ---
    "refurbished", "pre-owned", "open-box", "used-like-new", "renewed",
    "clearance", "bestseller", "new-arrival", "trending", "limited-edition",
    "exclusive", "bundle", "gift-set", "multipack",
    "prime", "express-shipping", "same-day", "next-day", "curbside-pickup",
    "backorder", "pre-order", "out-of-stock", "in-stock", "dropshipping",

    # --- Promotions & Fintech (Offers/Deals) ---
    "discount", "promo-code", "coupon", "voucher", "cashback", "rebate",
    "BOGO", "flash-sale", "clearance-sale", "final-sale",
    "Black-Friday", "Cyber-Monday", "Prime-Day", "holiday-deal",
    "BNPL", "EMI", "no-cost-EMI", "APR", "credit-card", "debit-card",
    "wallet", "crypto", "loyalty-points", "rewards", "subscription"
]

def _load_dataset_with_fallback(
    dataset_options: list[tuple[str, str | None]],
    source_name: str,
    default_split: str = "train",
) -> any:
    """Helper to load dataset with fallback options."""
    for dataset_id, subset in dataset_options:
        try:
            logger.info(f"Trying to load {source_name} dataset: {dataset_id}")
            if subset:
                dataset = load_dataset(dataset_id, subset, split=default_split, streaming=True)
            else:
                dataset = load_dataset(dataset_id, split=default_split, streaming=True)
            logger.info(f"Successfully loaded {source_name} dataset: {dataset_id}")
            return dataset
        except Exception as e:
            logger.warning(f"Could not load {dataset_id}: {e}")
            continue
    logger.warning(f"Could not load any {source_name} dataset")
    return None


def _chunk_long_text(text: str, chunk_size: int = 2000, overlap: int = 500, min_len: int = 200) -> list[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= chunk_size + overlap:
        return [text]
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size + overlap]
        if len(chunk) >= min_len:
            chunks.append(chunk)
    return chunks


def collect_ecommerce_corpus(max_samples: int | None = None, batch_size: int = 10_000) -> list[str]:
    """
    Collect e-commerce dataset texts using optimized batched mapping.
    """
    logger.info("Loading e-commerce dataset...")
    
    dataset_options = [
        ("thebajajra/Ecom-niverse", None),
        ("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty"),
        ("spacemanidol/product-search-corpus", None),
    ]
    
    dataset = _load_dataset_with_fallback(dataset_options, "e-commerce")
    if dataset is None:
        return []
    
    # OPTIMIZATION 1: Dynamic Field Detection
    # Inspect the first item to find which relevant columns actually exist.
    # This avoids checking 6 different dictionary keys for every single row.
    try:
        # Create a temporary iterator to peek without consuming the main dataset
        first_item = next(iter(dataset))
        available_columns = list(first_item.keys())
    except StopIteration:
        logger.warning("Dataset is empty.")
        return []

    target_fields = ["title", "description", "text", "product_name", "product_description", "name"]
    active_fields = [f for f in target_fields if f in available_columns]
    
    if not active_fields:
        logger.warning(f"No valid text fields found. Available: {available_columns}")
        return []
        
    logger.info(f"Active fields detected: {active_fields}")

    # OPTIMIZATION 2: Batched Processing
    # This runs the extraction logic on chunks of 1000 rows at a time
    def batch_extract(examples):
        # examples is a dict of lists: {'title': ['t1', 't2'], ...}
        batch_outputs = []
        
        # Get the number of rows in this batch safely
        num_rows = len(examples[active_fields[0]])
        
        for i in range(num_rows):
            row_extracted = []
            for field in active_fields:
                val = examples[field][i]
                # Fast validation: check existence and length > 50
                if val:
                    s_val = str(val)
                    if len(s_val) > 50:
                        row_extracted.append(s_val)
            batch_outputs.append(row_extracted)
            
        # Return a single column containing lists of strings
        return {"extracted_texts": batch_outputs}

    # Apply Map: This creates a lazy generator (doesn't run immediately)
    # remove_columns saves memory by dropping the unused raw data
    processed_dataset = dataset.map(
        batch_extract,
        batched=True,
        batch_size=100_000,
        remove_columns=available_columns
    )

    corpus = []
    log_threshold = 1_000_000
    
    # Iterate the stream until we reach max_samples
    # The map function is triggered here as we consume rows
    for row in processed_dataset:
        extracted = row["extracted_texts"]
        if extracted:
            corpus.extend(extracted)
            
        # Check exit condition
        if max_samples and len(corpus) >= max_samples:
            corpus = corpus[:max_samples]
            break
            
        # Efficient Logging
        if len(corpus) >= log_threshold:
             logger.info(f"  E-commerce: collected {len(corpus):,} samples...")
             log_threshold += 1_000_000

    logger.info(f"Collected {len(corpus):,} samples from e-commerce")
    return corpus


def collect_financial_corpus(max_samples: int | None = None, batch_size: int = 10000) -> list[str]:
    """
    Collect financial dataset texts in batches.
    Returns a list instead of yielding for better performance.
    """
    logger.info("Loading financial dataset...")
    
    dataset_options = [
        ("ashraq/financial-news", None),
        ("zeroshot/twitter-financial-news-topic", None),
        ("nickmuchi/financial-classification", None),
        ("eloukas/edgar-corpus", None),
        ("JanosAudworx/sec-10-k-filings", None),
    ]
    
    dataset = _load_dataset_with_fallback(dataset_options, "financial")
    if dataset is None:
        return []
    
    fields = ("text", "headline", "title", "content", "body", "filing_text", "document")
    corpus = []
    batch = []
    
    for item in dataset:
        # Get first available text field
        text = next((str(item[f]) for f in fields if item.get(f)), None)
        
        if not text or len(text) < 20:
            continue
        
        # Chunk long documents
        if len(text) > 5000:
            batch.extend(_chunk_long_text(text))
        else:
            batch.append(text)
        
        # Process in batches
        if len(batch) >= batch_size:
            corpus.extend(batch)
            batch = []
            if max_samples and len(corpus) >= max_samples:
                corpus = corpus[:max_samples]
                break
            if len(corpus) % 1_000_000 == 0:
                logger.info(f"  Financial: collected {len(corpus):,} samples...")
    
    # Add remaining batch
    if batch:
        remaining = max_samples - len(corpus) if max_samples else len(batch)
        corpus.extend(batch[:remaining])
    
    logger.info(f"Collected {len(corpus):,} samples from financial")
    return corpus


def collect_generic_corpus(max_samples: int | None = None, batch_size: int = 10_000) -> list[str]:
    """
    Collect generic web dataset texts using optimized batched mapping.
    Includes fast filtering and chunking.
    """
    logger.info("Loading generic web dataset...")
    
    dataset_options = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        ("HuggingFaceFW/fineweb-edu", "sample-100BT"),
    ]
    
    dataset = _load_dataset_with_fallback(dataset_options, "generic")
    if dataset is None:
        return []

    # OPTIMIZATION 1: Safe Column Detection
    # Although FineWeb usually has 'text', this prevents crashes if the schema changes
    try:
        first_item = next(iter(dataset))
        available_columns = list(first_item.keys())
    except StopIteration:
        logger.warning("Generic dataset is empty.")
        return []

    # Identify the text column (usually "text")
    text_col = "text" if "text" in available_columns else next(iter(available_columns))
    logger.info(f"Using column '{text_col}' for generic corpus")

    # OPTIMIZATION 2: Batched Logic
    # Moves the heavy lifting (len checks & chunking) into the efficient map loop
    def batch_process(examples):
        batch_outputs = []
        raw_texts = examples[text_col]
        
        for text in raw_texts:
            if not text:
                batch_outputs.append([]) 
                continue
                
            s_text = str(text)
            
            # Filter short texts (< 100 chars)
            if len(s_text) < 100:
                batch_outputs.append([])
                continue
                
            # Chunk long documents
            if len(s_text) > 5000:
                # Assuming _chunk_long_text is available in scope
                chunks = _chunk_long_text(s_text)
                batch_outputs.append(chunks)
            else:
                batch_outputs.append([s_text])
                
        # Returns list of lists to maintain 1:1 mapping with input rows
        return {"processed_chunks": batch_outputs}

    # Apply Map
    processed_dataset = dataset.map(
        batch_process,
        batched=True,
        batch_size=1000,
        remove_columns=available_columns
    )

    corpus = []
    log_threshold = 1_000_000

    # OPTIMIZATION 3: Fast Consumption
    for row in processed_dataset:
        chunks = row["processed_chunks"]
        if chunks:
            corpus.extend(chunks)

        # Check limits
        if max_samples and len(corpus) >= max_samples:
            corpus = corpus[:max_samples]
            break
            
        if len(corpus) >= log_threshold:
            logger.info(f"  Generic: collected {len(corpus):,} samples...")
            log_threshold += 1_000_000
    
    logger.info(f"Collected {len(corpus):,} samples from generic")
    return corpus


def collect_corpus_fast(
    max_samples_per_source: int | None = None,
    data_dir: str | None = None,
) -> list[str]:
    """
    Collect corpus data from all sources using batch-based collection.
    Much faster than iterator-based approach for large datasets.
    
    Args:
        max_samples_per_source: Maximum samples from each data source
        data_dir: Directory to save the collected corpus data (optional)
    
    Returns:
        List of text samples
    """
    logger.info("Collecting corpus data (batch mode)...")
    
    # Collect from each source in parallel-ready batches
    corpus = []
    
    # E-commerce corpus
    ecom_corpus = collect_ecommerce_corpus(max_samples=max_samples_per_source)
    corpus.extend(ecom_corpus)
    
    # Financial corpus
    financial_corpus = collect_financial_corpus(max_samples=max_samples_per_source)
    corpus.extend(financial_corpus)
    
    # Generic corpus  
    generic_corpus = collect_generic_corpus(max_samples=max_samples_per_source)
    corpus.extend(generic_corpus)
    
    # Shuffle the combined corpus
    logger.info(f"Shuffling {len(corpus):,} samples...")
    random.shuffle(corpus)
    logger.info(f"Total corpus size: {len(corpus):,} samples")
    
    # Save corpus data if data_dir is specified
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        corpus_path = os.path.join(data_dir, "corpus.jsonl")
        logger.info(f"Saving corpus data to {corpus_path}...")
        
        # Write in larger chunks for better I/O performance
        write_buffer = []
        buffer_size = 10000
        
        with open(corpus_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(corpus):
                write_buffer.append(json.dumps({"text": text}, ensure_ascii=False))
                if len(write_buffer) >= buffer_size:
                    f.write("\n".join(write_buffer) + "\n")
                    write_buffer = []
            # Write remaining
            if write_buffer:
                f.write("\n".join(write_buffer) + "\n")
        
        logger.info(f"Saved {len(corpus):,} samples to {corpus_path}")
    
    return corpus


def train_bpe_tokenizer(
    vocab_size: int = 65536,
    min_frequency: int = 2,
    output_dir: str = "./models/tokenizer",
    data_dir: str = "./data",
    max_samples_per_source: int | None = None,
    special_tokens: list[str] | None = None,
) -> PreTrainedTokenizerFast:
    """
    Train a BPE tokenizer on domain-specific corpus.
    
    Args:
        vocab_size: Target vocabulary size (64K or 96K recommended)
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save the tokenizer
        data_dir: Directory to save the collected corpus data
        max_samples_per_source: Maximum samples from each data source
        special_tokens: Additional special tokens to add
        
    Returns:
        Trained tokenizer as PreTrainedTokenizerFast
    """
    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}")
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Normalizer: lowercase and unicode normalization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
    ])
    
    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(),
    ])
    
    # Decoder
    tokenizer.decoder = decoders.BPEDecoder()
    
    # Define special tokens
    default_special_tokens = [
        "[PAD]",
        "[UNK]", 
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[BOS]",
        "[EOS]",
    ]
    
    all_special_tokens = default_special_tokens.copy()
    
    # Add domain-specific terms as special tokens
    for term in SPECIAL_DOMAIN_TOKENS:
        all_special_tokens.append(f"[{term}]")
        
    if special_tokens:
        all_special_tokens.extend(special_tokens)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_special_tokens = []
    for token in all_special_tokens:
        if token not in seen:
            seen.add(token)
            unique_special_tokens.append(token)
    
    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=unique_special_tokens,
        show_progress=True,
    )
    
    # Train on combined corpus using batch-based collection
    logger.info("Starting tokenizer training...")
    
    corpus = collect_corpus_fast(max_samples_per_source, data_dir=data_dir)
    logger.info(f"Training on {len(corpus):,} samples...")
    tokenizer.train_from_iterator(iter(corpus), trainer=trainer, length=len(corpus))
    
    # Add post-processor for [CLS] and [SEP]
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    # Wrap as PreTrainedTokenizerFast for HuggingFace compatibility
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    
    # Save in HuggingFace format
    fast_tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved HuggingFace tokenizer to {output_dir}")
    
    # Log vocabulary statistics
    vocab = tokenizer.get_vocab()
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Check domain-specific terms
    domain_terms_found = 0
    for term in ["ebitda", "roi", "hdmi", "polyester", "sku", "wacc"]:
        if term in vocab:
            domain_terms_found += 1
            logger.info(f"  '{term}' is a single token (ID: {vocab[term]})")
    
    logger.info(f"Domain terms as single tokens: {domain_terms_found}/{6}")
    
    return fast_tokenizer


def verify_tokenizer(tokenizer: PreTrainedTokenizerFast) -> None:
    """Verify tokenizer works correctly on domain-specific text."""
    
    test_texts = [
        "The company reported strong EBITDA growth with ROI exceeding expectations.",
        "This HDMI cable supports 4K resolution and is made of durable polyester.",
        "According to the 10-K filing, the company's WACC decreased by 50 basis points.",
        "SKU: ABC123 - Bluetooth wireless earbuds with USB-C charging.",
    ]
    
    logger.info("\n=== Tokenizer Verification ===")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        logger.info(f"\nText: {text}")
        logger.info(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        
        # Check specific terms
        for term in ["ebitda", "roi", "hdmi", "polyester", "sku"]:
            if term.lower() in text.lower():
                # Check if term is tokenized as single token
                term_tokens = tokenizer.tokenize(term)
                is_single = len(term_tokens) == 1
                logger.info(f"  '{term}' -> {term_tokens} ({'✓ single' if is_single else '✗ fragmented'})")


def main():
    parser = argparse.ArgumentParser(
        description="Train custom BPE tokenizer for Chroma-MoE"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        choices=[32768, 65536, 98304],
        help="Vocabulary size (32K, 64K, or 96K)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/tokenizer",
        help="Directory to save tokenizer",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save collected corpus data",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per data source (None = all)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with small sample size",
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        default=None,
        help="Only verify an existing tokenizer at this path",
    )
    args = parser.parse_args()
    
    if args.verify_only:
        logger.info(f"Loading tokenizer from {args.verify_only}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.verify_only)
        verify_tokenizer(tokenizer)
        return
    
    # Set sample size
    max_samples = args.max_samples
    if args.debug:
        max_samples = 5000
        logger.info("Debug mode: using 5000 samples per source")
    
    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_samples_per_source=max_samples,
    )
    
    # Verify tokenizer
    verify_tokenizer(tokenizer)
    
    logger.info(f"\n✓ Tokenizer saved to {args.output_dir}")
    logger.info(f"✓ Corpus data saved to {args.data_dir}")
    logger.info("To use in training, set tokenizer path in config.yaml")


if __name__ == "__main__":
    main()
