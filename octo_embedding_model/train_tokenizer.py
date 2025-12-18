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
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterator
import glob

import pyarrow as pa
import pyarrow.parquet as pq

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


def iter_ecommerce_corpus(max_samples: int | None = None) -> Iterator[str]:
    """
    Generator that yields e-commerce texts one at a time.
    Memory-efficient: only processes one batch at a time.
    """
    logger.info("Loading e-commerce dataset...")
    
    dataset_options = [
        ("thebajajra/Ecom-niverse", None),
        ("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty"),
        ("spacemanidol/product-search-corpus", None),
    ]
    
    dataset = _load_dataset_with_fallback(dataset_options, "e-commerce")
    if dataset is None:
        return
    
    # Dynamic field detection
    try:
        first_item = next(iter(dataset))
        available_columns = list(first_item.keys())
    except StopIteration:
        logger.warning("Dataset is empty.")
        return

    target_fields = ["title", "description", "text", "product_name", "product_description", "name"]
    active_fields = [f for f in target_fields if f in available_columns]
    
    if not active_fields:
        logger.warning(f"No valid text fields found. Available: {available_columns}")
        return
        
    logger.info(f"Active fields detected: {active_fields}")

    count = 0
    log_threshold = 1_000_000
    
    # Stream through dataset
    for item in dataset:
        for field in active_fields:
            val = item.get(field)
            if val:
                s_val = str(val)
                if len(s_val) > 50:
                    yield s_val
                    count += 1
                    
                    if max_samples and count >= max_samples:
                        logger.info(f"Collected {count:,} samples from e-commerce")
                        return
                    
                    if count >= log_threshold:
                        logger.info(f"  E-commerce: yielded {count:,} samples...")
                        log_threshold += 1_000_000

    logger.info(f"Collected {count:,} samples from e-commerce")


def iter_financial_corpus(max_samples: int | None = None) -> Iterator[str]:
    """
    Generator that yields financial texts one at a time.
    Memory-efficient: streams data directly.
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
        return
    
    fields = ("text", "headline", "title", "content", "body", "filing_text", "document")
    count = 0
    log_threshold = 1_000_000
    
    for item in dataset:
        # Get first available text field
        text = next((str(item[f]) for f in fields if item.get(f)), None)
        
        if not text or len(text) < 20:
            continue
        
        # Chunk long documents and yield each chunk
        if len(text) > 5000:
            for chunk in _chunk_long_text(text):
                yield chunk
                count += 1
                if max_samples and count >= max_samples:
                    logger.info(f"Collected {count:,} samples from financial")
                    return
        else:
            yield text
            count += 1
            if max_samples and count >= max_samples:
                logger.info(f"Collected {count:,} samples from financial")
                return
        
        if count >= log_threshold:
            logger.info(f"  Financial: yielded {count:,} samples...")
            log_threshold += 1_000_000
    
    logger.info(f"Collected {count:,} samples from financial")


def iter_generic_corpus(max_samples: int | None = None) -> Iterator[str]:
    """
    Generator that yields generic web texts one at a time.
    Optimized with batched processing for high throughput.
    """
    logger.info("Loading generic web dataset...")
    
    max_samples = max_samples * 2
    dataset_options = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        ("HuggingFaceFW/fineweb-edu", "sample-100BT"),
    ]
    
    # 1. Load Dataset
    dataset = _load_dataset_with_fallback(dataset_options, "generic")
    if dataset is None:
        return

    # 2. Dynamic Column Detection
    try:
        first_item = next(iter(dataset))
        available_columns = list(first_item.keys())
        text_col = "text" if "text" in available_columns else next(iter(available_columns))
    except StopIteration:
        return

    # 3. Define Batched Processor
    # This runs on chunks (e.g., 1000 items) at a time, reducing Python overhead.
    def batch_process(examples):
        outputs = []
        # 'examples' is a dict of lists. We iterate the list of texts.
        for text in examples[text_col]:
            if not text: 
                continue
                
            # Filter short
            if len(text) < 100:
                continue
                
            # Chunk long
            if len(text) > 5000:
                # Use extend for efficiency (assuming _chunk_long_text returns a list)
                outputs.extend(_chunk_long_text(text))
            else:
                outputs.append(text)
        
        # Return dict of lists. The 'datasets' library will automatically
        # flatten this and yield items one by one in the main loop.
        return {text_col: outputs}

    # 4. Apply Lazy Mapping
    # 'remove_columns' drops raw data immediately to save RAM.
    processed_dataset = dataset.map(
        batch_process,
        batched=True,
        batch_size=100000,
        remove_columns=available_columns
    )

    # 5. Fast Yield Loop
    count = 0
    log_threshold = 1_000_000
    
    # Iterate over the pre-processed stream
    for item in processed_dataset:
        yield item[text_col]
        
        count += 1
        
        # Check limits on the *output* count
        if max_samples and count >= max_samples:
            logger.info(f"Collected {count:,} samples from generic")
            return

        if count >= log_threshold:
            logger.info(f"  Generic: yielded {count:,} samples...")
            log_threshold += 1_000_000

    logger.info(f"Collected {count:,} samples from generic")


def iter_wiki_companies_corpus(max_samples: int | None = None) -> Iterator[str]:
    """
    Generator that yields Wikipedia company texts from local parquet file.
    Uses the filtered Wikipedia companies data.
    """
    wiki_path = Path("data/filtered_wiki_companies.parquet")
    
    if not wiki_path.exists():
        logger.warning(f"Wikipedia companies file not found: {wiki_path}")
        return
    
    logger.info(f"Loading Wikipedia companies from {wiki_path}...")
    
    try:
        parquet_file = pq.ParquetFile(wiki_path)
        count = 0
        log_threshold = 100_000
        
        for batch in parquet_file.iter_batches(batch_size=10_000, columns=["text"]):
            for text in batch.to_pydict()["text"]:
                if not text or len(text) < 100:
                    continue
                
                # Chunk long documents
                if len(text) > 5000:
                    for chunk in _chunk_long_text(text):
                        yield chunk
                        count += 1
                        if max_samples and count >= max_samples:
                            logger.info(f"Collected {count:,} samples from Wikipedia companies")
                            return
                else:
                    yield text
                    count += 1
                    if max_samples and count >= max_samples:
                        logger.info(f"Collected {count:,} samples from Wikipedia companies")
                        return
                
                if count >= log_threshold:
                    logger.info(f"  Wikipedia companies: yielded {count:,} samples...")
                    log_threshold += 100_000
        
        logger.info(f"Collected {count:,} samples from Wikipedia companies")
    
    except Exception as e:
        logger.warning(f"Error reading Wikipedia companies parquet: {e}")
        return


def _stream_corpus_to_disk(
    corpus_path: str,
    max_samples_per_source: int | None = None,
    write_buffer_size: int = 50_000,
) -> int:
    """
    Stream corpus data from all sources directly to disk as Parquet.
    Memory-efficient: only holds one write buffer at a time.
    
    Args:
        corpus_path: Path to save the Parquet file
        max_samples_per_source: Maximum samples from each data source
        write_buffer_size: Number of samples to buffer before writing as a row group
    
    Returns:
        Total number of samples written
    """
    logger.info(f"Streaming corpus data to {corpus_path}...")
    
    total_written = 0
    write_buffer = []
    writer = None
    schema = pa.schema([("text", pa.string())])
    
    def flush_buffer(writer, buffer, schema, corpus_path):
        """Write buffer to parquet file as a row group."""
        nonlocal total_written
        if buffer:
            table = pa.Table.from_pydict({"text": buffer}, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(corpus_path, schema, compression="snappy")
            writer.write_table(table)
        return writer, []
    
    try:
        # Process each source using generators (stream directly to disk)
        sources = [
            ("e-commerce", iter_ecommerce_corpus),
            ("financial", iter_financial_corpus),
            ("generic", iter_generic_corpus),
            ("wiki-companies", iter_wiki_companies_corpus),
        ]
        
        for source_name, collect_func in sources:
            logger.info(f"Processing {source_name} source...")
            source_data = collect_func(max_samples=max_samples_per_source)
            
            for text in source_data:
                write_buffer.append(text)
                
                if len(write_buffer) >= write_buffer_size:
                    writer, write_buffer = flush_buffer(writer, write_buffer, schema, corpus_path)
                    total_written += write_buffer_size
                    
                    if total_written % 1_000_000 == 0:
                        logger.info(f"  Written {total_written:,} samples to disk...")
            
            # Flush remaining after each source and update count
            remaining = len(write_buffer)
            writer, write_buffer = flush_buffer(writer, write_buffer, schema, corpus_path)
            total_written += remaining
            
            # Clear source data from memory immediately
            del source_data
            
            logger.info(f"  Completed {source_name}: total written = {total_written:,}")
    finally:
        if writer is not None:
            writer.close()
    
    logger.info(f"Streaming complete: {total_written:,} samples written to {corpus_path}")
    return total_written


def _shuffle_parquet_file(file_path: str, chunk_size: int = 500_000) -> None:
    """
    Shuffle rows in a Parquet file efficiently.
    Reads the file, shuffles row indices, then writes back.
    """
    import shutil
    
    logger.info(f"Shuffling corpus parquet file (chunk size: {chunk_size:,})...")
    
    # Read the parquet file
    table = pq.read_table(file_path)
    total_rows = table.num_rows
    
    if total_rows <= chunk_size:
        # Small file - shuffle in memory
        indices = list(range(total_rows))
        random.shuffle(indices)
        shuffled_table = table.take(indices)
        pq.write_table(shuffled_table, file_path, compression="snappy")
        logger.info(f"Shuffled {total_rows:,} rows in memory")
        return
    
    # Large file - use chunked shuffle with multiple passes
    temp_path = file_path + ".shuffle_tmp"
    
    for pass_num in range(2):  # 2 passes for better randomization
        logger.info(f"  Shuffle pass {pass_num + 1}/2...")
        
        table = pq.read_table(file_path)
        schema = table.schema
        writer = pq.ParquetWriter(temp_path, schema, compression="snappy")
        
        try:
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk_indices = list(range(start_idx, end_idx))
                random.shuffle(chunk_indices)
                chunk_table = table.take(chunk_indices)
                writer.write_table(chunk_table)
        finally:
            writer.close()
        
        shutil.move(temp_path, file_path)
    
    logger.info(f"Shuffled {total_rows:,} rows using chunked approach")


def _file_line_iterator(file_path: str, batch_size: int = 10_000) -> Iterator[str]:
    """
    Memory-efficient iterator over text from a Parquet file.
    Reads in batches for efficiency.
    """
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
        for text in batch.to_pydict()["text"]:
            if text:
                yield text


def collect_corpus_streaming(
    data_dir: str,
    max_samples_per_source: int | None = None,
) -> tuple[str, int]:
    """
    Collect corpus data by streaming directly to disk as Parquet.
    Memory-efficient: suitable for 10M+ samples.
    
    Args:
        data_dir: Directory to save the corpus file
        max_samples_per_source: Maximum samples from each data source
    
    Returns:
        Tuple of (corpus_path, total_samples)
    """
    os.makedirs(data_dir, exist_ok=True)
    corpus_path = os.path.join(data_dir, "corpus.parquet")
    
    # Stream to disk
    total_samples = _stream_corpus_to_disk(
        corpus_path=corpus_path,
        max_samples_per_source=max_samples_per_source,
    )
    
    # Shuffle the file
    _shuffle_parquet_file(corpus_path)
    
    return corpus_path, total_samples

def collect_corpus_to_shards(data_dir, max_samples):
    """
    Collect corpus data from all sources and save as sharded text files.
    Includes e-commerce, financial, and generic web data.
    """
    import itertools
    
    os.makedirs(data_dir, exist_ok=True)
    shard_size = 100_000
    current_shard = []
    shard_idx = 0
    
    # Combine all three corpus iterators
    combined_corpus = itertools.chain(
        iter_ecommerce_corpus(max_samples),
        iter_financial_corpus(max_samples),
        iter_generic_corpus(max_samples),
        iter_wiki_companies_corpus(max_samples),
    )
    
    # Iterate through combined corpus
    for i, text in enumerate(combined_corpus):
        current_shard.append(text)
        
        if len(current_shard) >= shard_size:
            with open(f"{data_dir}/shard_{shard_idx}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(current_shard))
            current_shard = []
            shard_idx += 1
            
    # Save remainder
    if current_shard:
        with open(f"{data_dir}/shard_{shard_idx}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(current_shard))

def train_bpe_tokenizer(
    vocab_size: int = 65536,
    min_frequency: int = 2,
    output_dir: str = "./models/tokenizer",
    data_dir: str = "./data",
    max_samples_per_source: int | None = None,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    
    logger.info(f"Initializing BPE tokenizer (Vocab: {vocab_size})...")
    
    # 1. Setup Tokenizer (Same as before)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(),
    ])
    
    tokenizer.decoder = decoders.BPEDecoder()

    # 2. Setup Special Tokens
    default_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    # Assuming SPECIAL_DOMAIN_TOKENS is defined globally
    domain_tokens = [f"[{t}]" for t in SPECIAL_DOMAIN_TOKENS] if 'SPECIAL_DOMAIN_TOKENS' in globals() else []
    
    all_special_tokens = list(dict.fromkeys(default_special_tokens + domain_tokens + (special_tokens or [])))

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=all_special_tokens,
        show_progress=True,
    )

    # 3. FAST DATA COLLECTION
    # Instead of streaming to one file, ensure your collection step saves 
    # multiple .txt files (shards) into data_dir.
    logger.info("Ensuring corpus data is ready...")
    
    # If the directory is empty, run collection (assumed to save .txt files)
    if not os.path.exists(data_dir) or not glob.glob(f"{data_dir}/*.txt"):
        # Modify your collect function to save files directly to data_dir
        # e.g., corpus_1.txt, corpus_2.txt
        collect_corpus_to_shards(data_dir=data_dir, max_samples=max_samples_per_source)

    # 4. FAST TRAINING (The Fix)
    # Get list of all text files
    files = glob.glob(f"{data_dir}/*.txt")
    logger.info(f"Training on {len(files)} files using native Rust parallelism...")
    
    if not files:
        raise ValueError("No training data found!")

    # .train() handles file I/O in Rust, releasing the GIL and using all CPU cores.
    tokenizer.train(files, trainer=trainer)
    
    # 5. Post-Processing
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
        choices=[32768, 65536, 98304, 131072],
        help="Vocabulary size (32K, 64K, 96K, or 128K)",
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
