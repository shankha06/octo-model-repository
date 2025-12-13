"""
Data loading utilities for Chroma-MoE embedding model training.

Supports loading from HuggingFace datasets:
- FinMTEB (finance)
- Amazon ESCI (e-commerce)
- ConvFinQA (finance Q&A)
- MS-MARCO (general)
- FineWeb-Edu (general grammatical competence - 15% mixture for Phase 1)

Note: All datasets are filtered to English-only content.
"""

import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class DataConfig:
    """Configuration for data loading."""

    max_length: int = 512
    mask_ratio: float = 0.15
    mean_span_length: int = 3
    debug_max_samples: int = 1000
    english_only: bool = True


def is_likely_english(text: str, min_ascii_ratio: float = 0.85) -> bool:
    """
    Fast heuristic to check if text is likely English.
    
    Uses ASCII ratio as a proxy (English text is mostly ASCII).
    This is faster than langdetect and sufficient for filtering.
    
    Args:
        text: Text to check
        min_ascii_ratio: Minimum ratio of ASCII characters (default 0.85)
        
    Returns:
        True if text is likely English
    """
    if not text or len(text) < 20:
        return True  # Allow short texts through
    
    # Count ASCII letters and common punctuation
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / len(text)
    
    # Additional check: presence of common English words
    english_patterns = re.compile(
        r'\b(the|and|is|are|was|were|have|has|been|will|would|could|should|'
        r'this|that|with|from|for|not|but|what|all|when|can|there|their|'
        r'which|how|about|into|more|other|than|then|some|these|only|new|'
        r'after|also|who|get|our|out|just|your|over|such|make|may)\b',
        re.IGNORECASE
    )
    has_english_words = bool(english_patterns.search(text[:500]))
    
    return ratio >= min_ascii_ratio and has_english_words


def filter_english(texts: list[str]) -> list[str]:
    """Filter list of texts to English-only content."""
    return [t for t in texts if is_likely_english(t)]


class SpanMaskingCollator:
    """
    Collator for Phase 1 pre-training with T5-style span masking.
    
    Masks contiguous spans of tokens rather than individual tokens,
    which is more challenging and leads to better representations.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mask_ratio: float = 0.15,
        mean_span_length: int = 3,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.mean_span_length = mean_span_length
        self.max_length = max_length
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id

    def _generate_span_mask(self, seq_length: int) -> list[tuple[int, int]]:
        """Generate random spans to mask."""
        num_tokens_to_mask = int(seq_length * self.mask_ratio)
        spans = []
        masked_count = 0

        while masked_count < num_tokens_to_mask:
            # Sample span length from geometric distribution
            span_length = min(
                max(1, int(random.expovariate(1.0 / self.mean_span_length))),
                num_tokens_to_mask - masked_count,
            )
            # Sample span start position
            start = random.randint(1, max(1, seq_length - span_length - 1))
            end = start + span_length

            # Check for overlap with existing spans
            overlap = False
            for s, e in spans:
                if not (end <= s or start >= e):
                    overlap = True
                    break

            if not overlap:
                spans.append((start, end))
                masked_count += span_length

        return sorted(spans)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate batch with span masking."""
        texts = [item["text"] for item in batch]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].clone()
        labels = encoded["input_ids"].clone()
        attention_mask = encoded["attention_mask"]

        # Apply span masking to each sequence
        for i in range(input_ids.size(0)):
            seq_length = attention_mask[i].sum().item()
            spans = self._generate_span_mask(int(seq_length))

            for start, end in spans:
                # Replace span with mask tokens
                input_ids[i, start:end] = self.mask_token_id

            # Set non-masked tokens to -100 in labels (ignore in loss)
            mask = torch.zeros_like(labels[i], dtype=torch.bool)
            for start, end in spans:
                mask[start:end] = True
            labels[i][~mask] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ContrastiveCollator:
    """
    Collator for Phase 2 contrastive training.
    
    Handles query-passage pairs with optional hard negatives.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        query_template: str = "Instruct: {instruction}\nQuery: {query}",
        passage_template: str = "{passage}",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_template = query_template
        self.passage_template = passage_template

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate batch for contrastive learning."""
        queries = []
        positives = []
        hard_negatives = []

        for item in batch:
            # Format query with instruction if available
            if "instruction" in item:
                query = self.query_template.format(
                    instruction=item.get("instruction", "Retrieve relevant passages."),
                    query=item["query"],
                )
            else:
                query = item["query"]

            queries.append(query)
            positives.append(item["positive"])

            # Collect hard negatives if available
            if "hard_negatives" in item:
                hard_negatives.extend(item["hard_negatives"])

        # Tokenize queries
        query_encoded = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize positives
        positive_encoded = self.tokenizer(
            positives,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "query_input_ids": query_encoded["input_ids"],
            "query_attention_mask": query_encoded["attention_mask"],
            "positive_input_ids": positive_encoded["input_ids"],
            "positive_attention_mask": positive_encoded["attention_mask"],
        }

        # Add hard negatives if available
        if hard_negatives:
            negative_encoded = self.tokenizer(
                hard_negatives,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["negative_input_ids"] = negative_encoded["input_ids"]
        return result


class PreTrainingDataset(Dataset):
    """Dataset for Phase 1 pre-training (in-memory)."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"text": self.texts[idx]}


class StreamingPreTrainingDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset for Phase 1 pre-training.
    
    Uses HuggingFace streaming datasets directly - training starts immediately
    without loading all data into memory first.
    """

    def __init__(
        self,
        config: dict[str, Any],
        max_samples_per_source: int = 3000000,
        debug: bool = False,
    ):
        self.config = config
        self.debug = debug
        self.max_samples = 1000 if debug else max_samples_per_source
        self._estimated_size = self.max_samples * 3  # 3 sources
        
    def __len__(self) -> int:
        """Estimated size for progress bars."""
        return self._estimated_size
        
    def _stream_finance(self):
        """Stream finance data."""
        finance_datasets = [
            ("ashraq/financial-news", ["headline", "title", "text"]),
            ("zeroshot/twitter-financial-news-topic", ["text", "tweet", "sentence"]),
        ]
        
        count = 0
        for dataset_id, text_fields in finance_datasets:
            if count >= self.max_samples:
                break
            try:
                dataset = load_dataset(dataset_id, split="train", streaming=True)
                for item in dataset:
                    text = None
                    for field in text_fields:
                        if field in item and item[field]:
                            text = str(item[field])
                            break
                    if text and len(text) > 20:
                        yield {"text": text}
                        count += 1
                        if count >= self.max_samples:
                            break
            except Exception:
                continue
    
    def _stream_ecommerce(self):
        """Stream e-commerce data."""
        count = 0
        try:
            dataset = load_dataset("thebajajra/Ecom-niverse", split="train", streaming=True)
            for item in dataset:
                for field in ["title", "description", "text", "product_name"]:
                    if field in item and item[field]:
                        text = str(item[field])
                        if len(text) > 50:
                            yield {"text": text}
                            count += 1
                            if count >= self.max_samples:
                                return
                            break
        except Exception:
            pass
    
    def _stream_generic(self):
        """Stream generic web data."""
        count = 0
        try:
            subset = self.config.get("general", {}).get("fineweb_edu", {}).get("subset", "sample-10BT")
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", subset, split="train", streaming=True)
            for item in dataset:
                text = item.get("text", "")
                if text and len(text) > 100:
                    # Chunk long texts
                    if len(text) > 5000:
                        for i in range(0, len(text), 2000):
                            chunk = text[i:i+2500]
                            if len(chunk) > 200:
                                yield {"text": chunk}
                                count += 1
                                if count >= self.max_samples:
                                    return
                    else:
                        yield {"text": text}
                        count += 1
                        if count >= self.max_samples:
                            return
        except Exception:
            pass
    
    def __iter__(self):
        """Interleave all sources for balanced training."""
        sources = [
            self._stream_finance(),
            self._stream_ecommerce(),
            self._stream_generic(),
        ]
        
        # Round-robin interleaving
        exhausted = [False] * len(sources)
        while not all(exhausted):
            for i, source in enumerate(sources):
                if exhausted[i]:
                    continue
                try:
                    yield next(source)
                except StopIteration:
                    exhausted[i] = True


class ContrastiveDataset(Dataset):
    """Dataset for Phase 2 contrastive training."""

    def __init__(
        self,
        queries: list[str],
        positives: list[str],
        hard_negatives: list[list[str]] | None = None,
        instructions: list[str] | None = None,
    ):
        self.queries = queries
        self.positives = positives
        self.hard_negatives = hard_negatives
        self.instructions = instructions

        assert len(queries) == len(positives), "Queries and positives must have same length"

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "query": self.queries[idx],
            "positive": self.positives[idx],
        }

        if self.hard_negatives is not None:
            item["hard_negatives"] = self.hard_negatives[idx]

        if self.instructions is not None:
            item["instruction"] = self.instructions[idx]

        return item


def load_finmteb(
    split: str = "train",
    max_samples: int | None = None,
    english_only: bool = True,
) -> PreTrainingDataset:
    """
    Load financial datasets for finance domain pre-training.
    
    Uses multiple public finance datasets as alternatives to FinMTEB.
    Note: Filtered to English-only content.
    """
    texts = []
    
    # List of alternative finance datasets to try
    finance_datasets = [
        ("ashraq/financial-news", ["headline", "title", "text"]),
        ("zeroshot/twitter-financial-news-topic", ["text", "tweet", "sentence"]),
        ("nickmuchi/financial-classification", ["text", "sentence", "content"]),
    ]
    
    for dataset_id, text_fields in finance_datasets:
        try:
            print(f"Loading finance dataset: {dataset_id}")
            dataset = load_dataset(dataset_id, split=split, streaming=True)
            
            sample_count = 0
            target = (max_samples * 2) if max_samples else None  # Over-sample for filtering
            
            for item in dataset:
                # Try multiple possible text fields
                text = None
                for field in text_fields:
                    if field in item and item[field]:
                        text = str(item[field])
                        break
                
                if text and len(text) > 20:
                    texts.append(text)
                    sample_count += 1
                
                if target and sample_count >= target:
                    break
            
            print(f"Loaded {sample_count} samples from {dataset_id}")
            
            # If we got enough, stop trying more datasets
            if max_samples and len(texts) >= max_samples:
                break
                
        except Exception as e:
            print(f"Warning: Could not load {dataset_id}: {e}")
            continue
    
    # Filter to English only
    if english_only and texts:
        texts = filter_english(texts)
        print(f"Filtered to {len(texts)} English samples from finance datasets")
    
    return PreTrainingDataset(texts[:max_samples] if max_samples else texts)


def load_esci(
    split: str = "train",
    max_samples: int | None = None,
    include_hard_negatives: bool = True,
) -> ContrastiveDataset:
    """
    Load Amazon ESCI dataset for e-commerce contrastive training.
    
    Dataset: tasksource/esci
    Uses "Exact" matches as positives, "Substitute" as hard negatives.
    """
    try:
        dataset = load_dataset("tasksource/esci", split=split)

        queries = []
        positives = []
        hard_negatives = []

        # Group by query for hard negative mining
        query_to_products: dict[str, dict[str, list]] = {}

        for item in dataset:
            query = item.get("query", "")
            product = item.get("product_title", "") or item.get("product", "")
            label = item.get("esci_label", "") or item.get("label", "")

            if not query or not product:
                continue

            if query not in query_to_products:
                query_to_products[query] = {"exact": [], "substitute": [], "complement": []}

            if label.lower() in ["exact", "e"]:
                query_to_products[query]["exact"].append(product)
            elif label.lower() in ["substitute", "s"]:
                query_to_products[query]["substitute"].append(product)
            elif label.lower() in ["complement", "c"]:
                query_to_products[query]["complement"].append(product)

        # Create training pairs
        for query, products in query_to_products.items():
            if not products["exact"]:
                continue

            for positive in products["exact"]:
                queries.append(query)
                positives.append(positive)

                # Use substitutes as hard negatives
                if include_hard_negatives:
                    negs = products["substitute"][:7]  # Max 7 hard negatives
                    hard_negatives.append(negs)

            if max_samples and len(queries) >= max_samples:
                break

        if max_samples:
            queries = queries[:max_samples]
            positives = positives[:max_samples]
            hard_negatives = hard_negatives[:max_samples] if include_hard_negatives else None

        return ContrastiveDataset(
            queries=queries,
            positives=positives,
            hard_negatives=hard_negatives if include_hard_negatives else None,
            instructions=["Find the exact matching product"] * len(queries),
        )
    except Exception as e:
        print(f"Warning: Could not load ESCI: {e}")
        return ContrastiveDataset([], [])


def load_convfinqa(
    split: str = "train",
    max_samples: int | None = None,
) -> ContrastiveDataset:
    """
    Load ConvFinQA dataset for financial Q&A contrastive training.
    
    Dataset: MehdiHosseiniMoghadam/ConvFinQA
    """
    try:
        dataset = load_dataset("MehdiHosseiniMoghadam/ConvFinQA", split=split)

        queries = []
        positives = []
        instructions = []

        for item in dataset:
            question = item.get("question", "") or item.get("query", "")
            context = item.get("context", "") or item.get("passage", "") or item.get("text", "")

            if question and context:
                queries.append(question)
                positives.append(context)
                instructions.append("Find the financial document that answers this question")

            if max_samples and len(queries) >= max_samples:
                break

        return ContrastiveDataset(
            queries=queries[:max_samples] if max_samples else queries,
            positives=positives[:max_samples] if max_samples else positives,
            instructions=instructions[:max_samples] if max_samples else instructions,
        )
    except Exception as e:
        print(f"Warning: Could not load ConvFinQA: {e}")
        return ContrastiveDataset([], [])


def load_msmarco(
    split: str = "train",
    max_samples: int | None = None,
) -> ContrastiveDataset:
    """
    Load MS-MARCO dataset for general domain regularization.
    
    Dataset: ms_marco (v1.1)
    """
    try:
        dataset = load_dataset("ms_marco", "v1.1", split=split)

        queries = []
        positives = []
        instructions = []

        for item in dataset:
            query = item.get("query", "")
            passages = item.get("passages", {})

            # Get the positive (is_selected) passage
            if isinstance(passages, dict) and "passage_text" in passages:
                passage_texts = passages["passage_text"]
                is_selected = passages.get("is_selected", [0] * len(passage_texts))

                for i, text in enumerate(passage_texts):
                    if is_selected[i] == 1:
                        queries.append(query)
                        positives.append(text)
                        instructions.append("Retrieve the passage that answers this question")
                        break

            if max_samples and len(queries) >= max_samples:
                break

        return ContrastiveDataset(
            queries=queries[:max_samples] if max_samples else queries,
            positives=positives[:max_samples] if max_samples else positives,
            instructions=instructions[:max_samples] if max_samples else instructions,
        )
    except Exception as e:
        print(f"Warning: Could not load MS-MARCO: {e}")
        return ContrastiveDataset([], [])


def load_fineweb_edu(
    subset: str = "sample-10BT",
    split: str = "train",
    max_samples: int | None = None,
    english_only: bool = True,
) -> PreTrainingDataset:
    """
    Load FineWeb-Edu dataset for general grammatical competence.
    
    Dataset: HuggingFaceFW/fineweb-edu
    Uses the sample-10BT subset for faster loading.
    Note: Filtered to English-only content.
    """
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            subset,
            split=split,
            streaming=True,  # Use streaming for large dataset
        )

        texts = []
        # Over-sample for filtering
        target = max_samples * 2 if max_samples else None
        
        for item in dataset:
            text = item.get("text", "")
            # Filter short texts and apply English check inline for efficiency
            if text and len(text) > 200:
                if not english_only or is_likely_english(text):
                    texts.append(text)

            if target and len(texts) >= target:
                break

        texts = texts[:max_samples] if max_samples else texts
        print(f"Loaded {len(texts)} English samples from FineWeb-Edu")
        return PreTrainingDataset(texts)
    except Exception as e:
        print(f"Warning: Could not load FineWeb-Edu: {e}")
        return PreTrainingDataset([])


def load_ecomniverse(
    split: str = "train",
    max_samples: int | None = None,
    english_only: bool = True,
) -> PreTrainingDataset:
    """
    Load Ecom-niverse dataset for e-commerce domain pre-training.
    
    Dataset: thebajajra/Ecom-niverse
    Note: Filtered to English-only content.
    """
    print(f"Loading Ecom-niverse dataset (max {max_samples:,} samples)..." if max_samples else "Loading Ecom-niverse dataset...")
    
    try:
        dataset = load_dataset(
            "thebajajra/Ecom-niverse",
            split=split,
            streaming=True,
        )
        
        texts = []
        target = max_samples * 2 if max_samples else None  # Over-sample for filtering
        
        for item in dataset:
            # Extract text from various fields
            for field in ["title", "description", "text", "product_name", "product_description", "name"]:
                if field in item and item[field]:
                    text = str(item[field])
                    if len(text) > 50:
                        if not english_only or is_likely_english(text):
                            texts.append(text)
                            if target and len(texts) >= target:
                                break
            if target and len(texts) >= target:
                break
        
        texts = texts[:max_samples] if max_samples else texts
        print(f"Loaded {len(texts):,} English samples from Ecom-niverse")
        return PreTrainingDataset(texts)
    except Exception as e:
        print(f"Warning: Could not load Ecom-niverse: {e}")
        return PreTrainingDataset([])


def create_combined_pretraining_dataset(
    config: dict[str, Any],
    debug: bool = False,
) -> PreTrainingDataset:
    """
    Create combined pre-training dataset from multiple sources.
    
    Sources:
    - Finance: ashraq/financial-news, etc.
    - E-commerce: thebajajra/Ecom-niverse
    - General: HuggingFaceFW/fineweb-edu
    """
    # Default to 3M samples per dataset, 1K for debug
    default_max_samples = 3000000
    debug_max = config.get("debug", {}).get("max_samples_per_dataset", 1000)
    max_samples = debug_max if debug else config.get("max_samples_per_dataset", default_max_samples)
    
    print(f"\n=== Loading Pre-training Data (max {max_samples:,} per source) ===")

    all_texts = []

    # Load Finance datasets
    if config.get("finance", {}).get("finmteb", {}).get("enabled", True):
        finance_max = config.get("finance", {}).get("finmteb", {}).get("max_samples", max_samples)
        finmteb = load_finmteb(max_samples=finance_max)
        all_texts.extend(finmteb.texts)
        print(f"Finance samples: {len(finmteb):,}")

    # Load E-commerce datasets (Ecom-niverse)
    if config.get("ecommerce", {}).get("enabled", True):
        ecom_max = config.get("ecommerce", {}).get("max_samples", max_samples)
        ecom = load_ecomniverse(max_samples=ecom_max)
        all_texts.extend(ecom.texts)
        print(f"E-commerce samples: {len(ecom):,}")

    # Load FineWeb-Edu (general domain)
    fineweb_config = config.get("general", {}).get("fineweb_edu", {})
    if fineweb_config.get("enabled", True):
        fineweb_max = fineweb_config.get("max_samples", max_samples)
        fineweb = load_fineweb_edu(
            subset=fineweb_config.get("subset", "sample-10BT"),
            max_samples=fineweb_max,
        )
        all_texts.extend(fineweb.texts)
        print(f"Generic samples: {len(fineweb):,}")

    # Shuffle combined dataset
    random.shuffle(all_texts)

    print(f"\n=== Total pre-training samples: {len(all_texts):,} ===")

    return PreTrainingDataset(all_texts)


def create_streaming_pretraining_dataset(
    config: dict[str, Any],
    debug: bool = False,
) -> StreamingPreTrainingDataset:
    """
    Create streaming pre-training dataset for immediate training start.
    
    Uses IterableDataset that streams data directly from HuggingFace,
    avoiding the delay of loading all data into memory first.
    """
    max_samples = config.get("max_samples_per_dataset", 3000000)
    
    print(f"\n=== Creating Streaming Dataset (max {max_samples:,} per source) ===")
    print("Training will start immediately while data streams in...")
    
    return StreamingPreTrainingDataset(
        config=config,
        max_samples_per_source=max_samples,
        debug=debug,
    )


def create_combined_contrastive_dataset(
    config: dict[str, Any],
    debug: bool = False,
) -> ContrastiveDataset:
    """
    Create combined contrastive dataset from multiple sources.
    """
    max_samples = config.get("debug", {}).get("max_samples_per_dataset", 1000) if debug else None

    all_queries = []
    all_positives = []
    all_hard_negatives = []
    all_instructions = []

    # Load ESCI
    if config.get("retail", {}).get("esci", {}).get("enabled", True):
        esci = load_esci(max_samples=max_samples)
        all_queries.extend(esci.queries)
        all_positives.extend(esci.positives)
        if esci.hard_negatives:
            all_hard_negatives.extend(esci.hard_negatives)
        if esci.instructions:
            all_instructions.extend(esci.instructions)
        print(f"Loaded {len(esci)} samples from ESCI")

    # Load ConvFinQA
    if config.get("finance", {}).get("convfinqa", {}).get("enabled", True):
        convfinqa = load_convfinqa(max_samples=max_samples)
        all_queries.extend(convfinqa.queries)
        all_positives.extend(convfinqa.positives)
        # ConvFinQA doesn't have hard negatives, pad with empty lists
        all_hard_negatives.extend([[] for _ in range(len(convfinqa))])
        if convfinqa.instructions:
            all_instructions.extend(convfinqa.instructions)
        print(f"Loaded {len(convfinqa)} samples from ConvFinQA")

    # Load MS-MARCO
    if config.get("general", {}).get("msmarco", {}).get("enabled", True):
        msmarco = load_msmarco(max_samples=max_samples)
        all_queries.extend(msmarco.queries)
        all_positives.extend(msmarco.positives)
        all_hard_negatives.extend([[] for _ in range(len(msmarco))])
        if msmarco.instructions:
            all_instructions.extend(msmarco.instructions)
        print(f"Loaded {len(msmarco)} samples from MS-MARCO")

    # Shuffle together
    combined = list(zip(all_queries, all_positives, all_hard_negatives, all_instructions))
    random.shuffle(combined)
    all_queries, all_positives, all_hard_negatives, all_instructions = zip(*combined) if combined else ([], [], [], [])

    return ContrastiveDataset(
        queries=list(all_queries),
        positives=list(all_positives),
        hard_negatives=list(all_hard_negatives) if all_hard_negatives else None,
        instructions=list(all_instructions) if all_instructions else None,
    )
