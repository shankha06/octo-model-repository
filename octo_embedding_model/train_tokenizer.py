"""
Custom BPE Tokenizer Training for Chroma-MoE Embedding Model.

Trains a domain-specific BPE tokenizer on:
- Ecom-niverse (e-commerce data)
- EDGAR (SEC financial filings)

This ensures domain-specific terms like "EBITDA", "RoI", "HDMI", "polyester"
are single tokens rather than fragmented sub-words.

Usage:
    # Train tokenizer with default settings
    python train_tokenizer.py --output-dir ./tokenizer
    
    # Train with specific vocab size
    python train_tokenizer.py --vocab-size 65536 --output-dir ./tokenizer
    
    # Debug mode with small sample
    python train_tokenizer.py --debug --output-dir ./tokenizer
"""

import argparse
import logging
import os
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


def get_ecomniverse_iterator(
    max_samples: int | None = None,
    english_only: bool = True,
) -> Iterator[str]:
    """
    Iterator over Ecom-niverse dataset texts.
    
    Dataset: thebajajra/Ecom-niverse
    """
    logger.info("Loading Ecom-niverse dataset...")
    
    try:
        dataset = load_dataset(
            "thebajajra/Ecom-niverse",
            split="train",
            streaming=True,
        )
        
        count = 0
        for item in dataset:
            # Extract text content from various fields
            texts = []
            
            if "title" in item and item["title"]:
                texts.append(str(item["title"]))
            if "description" in item and item["description"]:
                texts.append(str(item["description"]))
            if "text" in item and item["text"]:
                texts.append(str(item["text"]))
            if "product_name" in item and item["product_name"]:
                texts.append(str(item["product_name"]))
            if "product_description" in item and item["product_description"]:
                texts.append(str(item["product_description"]))
                
            for text in texts:
                if len(text) > 50:  # Filter very short texts
                    # Simple English check (ASCII ratio)
                    if english_only:
                        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
                        if ascii_ratio < 0.85:
                            continue
                    
                    yield text
                    count += 1
                    
                    if max_samples and count >= max_samples:
                        return
                        
    except Exception as e:
        logger.warning(f"Could not load Ecom-niverse: {e}")
        return


def get_edgar_iterator(
    max_samples: int | None = None,
    english_only: bool = True,
) -> Iterator[str]:
    """
    Iterator over EDGAR SEC filings dataset texts.
    
    Uses financial-reports-sec or similar EDGAR datasets.
    """
    logger.info("Loading EDGAR dataset...")
    
    # Try different EDGAR dataset sources
    dataset_options = [
        ("eloukas/edgar-corpus", None),
        ("JanosAudworx/sec-10-k-filings", None),
        ("financial-reports-sec", None),
    ]
    
    dataset = None
    for dataset_id, subset in dataset_options:
        try:
            if subset:
                dataset = load_dataset(dataset_id, subset, split="train", streaming=True)
            else:
                dataset = load_dataset(dataset_id, split="train", streaming=True)
            logger.info(f"Loaded EDGAR dataset: {dataset_id}")
            break
        except Exception as e:
            logger.debug(f"Could not load {dataset_id}: {e}")
            continue
    
    if dataset is None:
        logger.warning("Could not load any EDGAR dataset, using fallback financial news")
        # Fallback to financial news
        try:
            dataset = load_dataset(
                "Brianferrell787/financial-news-multisource",
                split="train",
                streaming=True,
            )
            logger.info("Loaded financial news as fallback")
        except Exception as e:
            logger.warning(f"Could not load fallback dataset: {e}")
            return
    
    count = 0
    for item in dataset:
        # Extract text content
        text = None
        for field in ["text", "content", "body", "filing_text", "document"]:
            if field in item and item[field]:
                text = str(item[field])
                break
        
        if not text or len(text) < 100:
            continue
            
        # For long documents, yield chunks
        if len(text) > 5000:
            # Split into ~2000 char chunks
            for i in range(0, len(text), 2000):
                chunk = text[i:i+2500]
                if len(chunk) > 200:
                    yield chunk
                    count += 1
                    if max_samples and count >= max_samples:
                        return
        else:
            yield text
            count += 1
            
        if max_samples and count >= max_samples:
            return


def combined_corpus_iterator(
    ecom_samples: int | None = None,
    edgar_samples: int | None = None,
    mix_ratio: float = 0.5,
) -> Iterator[str]:
    """
    Combined iterator over both datasets with specified mix ratio.
    
    Args:
        ecom_samples: Max samples from Ecom-niverse
        edgar_samples: Max samples from EDGAR
        mix_ratio: Ratio of Ecom-niverse to EDGAR (0.5 = equal mix)
    """
    # Create iterators
    ecom_iter = get_ecomniverse_iterator(max_samples=ecom_samples)
    edgar_iter = get_edgar_iterator(max_samples=edgar_samples)
    
    ecom_exhausted = False
    edgar_exhausted = False
    
    import random
    
    while not (ecom_exhausted and edgar_exhausted):
        # Decide which source to sample from
        if ecom_exhausted:
            source = "edgar"
        elif edgar_exhausted:
            source = "ecom"
        else:
            source = "ecom" if random.random() < mix_ratio else "edgar"
        
        try:
            if source == "ecom":
                text = next(ecom_iter)
                yield text
            else:
                text = next(edgar_iter)
                yield text
        except StopIteration:
            if source == "ecom":
                ecom_exhausted = True
            else:
                edgar_exhausted = True


def train_bpe_tokenizer(
    vocab_size: int = 65536,
    min_frequency: int = 2,
    output_dir: str = "./tokenizer",
    max_samples_per_source: int | None = None,
    special_tokens: list[str] | None = None,
) -> PreTrainedTokenizerFast:
    """
    Train a BPE tokenizer on domain-specific corpus.
    
    Args:
        vocab_size: Target vocabulary size (64K or 96K recommended)
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save the tokenizer
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
    
    # Train on combined corpus
    logger.info("Starting tokenizer training...")
    
    def corpus_iterator():
        yield from combined_corpus_iterator(
            ecom_samples=max_samples_per_source,
            edgar_samples=max_samples_per_source,
            mix_ratio=0.5,
        )
    
    tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
    
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
        default="./tokenizer",
        help="Directory to save tokenizer",
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
        max_samples = 1000
        logger.info("Debug mode: using 1000 samples per source")
    
    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir,
        max_samples_per_source=max_samples,
    )
    
    # Verify tokenizer
    verify_tokenizer(tokenizer)
    
    logger.info(f"\n✓ Tokenizer saved to {args.output_dir}")
    logger.info("To use in training, set tokenizer path in config.yaml")


if __name__ == "__main__":
    main()
