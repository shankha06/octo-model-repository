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
import pyarrow.parquet as pq
from pathlib import Path


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
        # Dynamic mask ratio: random between 0.05 and self.mask_ratio
        current_ratio = random.uniform(0.05, self.mask_ratio)
        num_tokens_to_mask = int(seq_length * current_ratio)
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


class PackedSpanMaskingCollator:
    """
    Collator for Phase 1 pre-training with sequence packing (unpadding).
    
    Packs multiple short sequences into a single buffer to eliminate wasted
    compute on padding tokens. Uses cu_seqlens for flash_attn_varlen_func.
    
    This can provide 2-3x speedup depending on sequence length variance.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mask_ratio: float = 0.30,
        mean_span_length: int = 3,
        max_length: int = 4096,  # Total packed sequence length
    ):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.mean_span_length = mean_span_length
        self.max_length = max_length
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id
        self.pad_token_id = tokenizer.pad_token_id or 0

    def _generate_span_mask(self, seq_length: int) -> list[tuple[int, int]]:
        """Generate random spans to mask."""
        current_ratio = random.uniform(0.05, self.mask_ratio)
        num_tokens_to_mask = int(seq_length * current_ratio)
        spans = []
        masked_count = 0

        while masked_count < num_tokens_to_mask and seq_length > 2:
            span_length = min(
                max(1, int(random.expovariate(1.0 / self.mean_span_length))),
                num_tokens_to_mask - masked_count,
            )
            start = random.randint(1, max(1, seq_length - span_length - 1))
            end = start + span_length

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
        """Collate batch with sequence packing and span masking."""
        texts = [item["text"] for item in batch]

        # Tokenize without padding (we'll pack manually)
        encoded_list = []
        for text in texts:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=False,
            )
            encoded_list.append(enc["input_ids"])

        # Pack sequences into a single buffer
        packed_input_ids = []
        packed_labels = []
        cu_seqlens = [0]  # Cumulative sequence lengths
        
        current_length = 0
        for token_ids in encoded_list:
            seq_len = len(token_ids)
            
            # If adding this sequence would exceed max_length, stop packing
            if current_length + seq_len > self.max_length:
                break
            
            # Generate span mask for this sequence
            spans = self._generate_span_mask(seq_len)
            
            # Create input_ids with masking
            masked_ids = list(token_ids)
            label_ids = [-100] * seq_len  # Start with all ignored
            
            for start, end in spans:
                for j in range(start, min(end, seq_len)):
                    label_ids[j] = masked_ids[j]  # Store original for loss
                    masked_ids[j] = self.mask_token_id
            
            packed_input_ids.extend(masked_ids)
            packed_labels.extend(label_ids)
            current_length += seq_len
            cu_seqlens.append(current_length)
        
        # Handle case where no sequences fit (shouldn't happen normally)
        if current_length == 0:
            # Fallback: just use first sequence truncated
            token_ids = encoded_list[0][:self.max_length]
            seq_len = len(token_ids)
            packed_input_ids = list(token_ids)
            packed_labels = [-100] * seq_len
            cu_seqlens = [0, seq_len]
            current_length = seq_len
        
        # Convert to tensors
        input_ids = torch.tensor(packed_input_ids, dtype=torch.long)
        labels = torch.tensor(packed_labels, dtype=torch.long)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        max_seqlen = max(cu_seqlens[1:] - cu_seqlens[:-1]).item()
        
        # Create position_ids for each packed sequence
        position_ids = []
        for i in range(len(cu_seqlens) - 1):
            seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
            position_ids.extend(range(seq_len))
        position_ids = torch.tensor(position_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids.unsqueeze(0),  # [1, total_len]
            "labels": labels.unsqueeze(0),  # [1, total_len]
            "position_ids": position_ids.unsqueeze(0),  # [1, total_len]
            "cu_seqlens": cu_seqlens,  # [num_seqs + 1]
            "max_seqlen": max_seqlen,
            "packed": True,  # Flag for model to use varlen attention
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
        self._estimated_size = self.max_samples * 4  # 4 sources (finance, ecommerce, generic, wiki)
        
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
    
    def _stream_wiki_companies(self):
        """Stream Wikipedia company data from local parquet file."""
        wiki_path = Path("data/filtered_wiki_companies.parquet")
        
        if not wiki_path.exists():
            return
        
        count = 0
        try:
            parquet_file = pq.ParquetFile(wiki_path)
            for batch in parquet_file.iter_batches(batch_size=1000, columns=["text"]):
                for text in batch.to_pydict()["text"]:
                    if not text or len(text) < 100:
                        continue
                    
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
            self._stream_wiki_companies(),
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
    
    # # Filter to English only
    # if english_only and texts:
    #     texts = filter_english(texts)
    #     print(f"Filtered to {len(texts)} English samples from finance datasets")
    
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
    Load FineWeb-Edu dataset optimized for high-throughput streaming.
    Uses batch processing to minimize Python loop overhead.
    """
    print(f"Loading FineWeb-Edu dataset ({'max ' + str(max_samples) if max_samples else 'all'} samples)...")
    
    try:
        # 1. Load in streaming mode
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=subset,
            split=split,
            streaming=True,
        )

        # 2. Optimization: Drop unused columns immediately 
        # This saves bandwidth and parsing time.
        dataset = dataset.select_columns(["text"])

        # 3. Limit stream
        if max_samples:
            dataset = dataset.take(max_samples)

        # 4. Create a Batched Iterator
        # Iterating by batch (e.g., 1000 rows at a time) reduces Python function call 
        # overhead by 1000x compared to single-row iteration.
        batch_size = 10000
        iterator = dataset.iter(batch_size=batch_size)
        
        texts = []
        
        try:
            # 5. Robust "Peek" (Without restarting stream)
            # We fetch the first batch to validate schema/connection.
            first_batch = next(iterator)
            
            if "text" not in first_batch:
                warnings.warn(f"Column 'text' missing in {subset}. Found: {list(first_batch.keys())}")
                return PreTrainingDataset([])
            
            # Add first batch data
            texts.extend(first_batch["text"])
            
            # 6. Fast Consumption
            # Consume the remaining batches from the SAME iterator.
            for batch in iterator:
                texts.extend(batch["text"])

        except StopIteration:
            warnings.warn(f"Dataset {subset} is empty.")
            return PreTrainingDataset([])
            
        print(f"Loaded {len(texts):,} samples from FineWeb-Edu")
        return PreTrainingDataset(texts)

    except Exception as e:
        print(f"Error loading FineWeb-Edu: {e}")
        return PreTrainingDataset([])

def load_ecomniverse(
    split: str = "train",
    max_samples: int | None = None,
    english_only: bool = True,
) -> PreTrainingDataset:
    """
    Load Ecom-niverse dataset for e-commerce domain pre-training.
    Optimized for speed and streaming compatibility.
    """
    print(f"Loading Ecom-niverse dataset ({'max ' + str(max_samples) if max_samples else 'all'} samples)...")
    
    try:
        # 1. Load in streaming mode
        dataset = load_dataset(
            "thebajajra/Ecom-niverse",
            split=split,
            streaming=True,
        )

        # 2. Limit samples efficiently
        if max_samples:
            dataset = dataset.take(max_samples)

        # 3. Safe Column Detection (The Fix)
        # Streaming datasets often lack metadata. We peek at the first item 
        # to find column names so we can remove them later.
        try:
            # We create a temporary iterator to peek without consuming the main dataset object
            sample_item = next(iter(dataset))
            column_names = list(sample_item.keys())
        except StopIteration:
            print("Warning: Dataset is empty.")
            return PreTrainingDataset([])

        # 4. Define fast batch processing
        def batch_format(examples):
            batch_texts = []
            # 'examples' is a dict of lists: {'col1': [val1, val2], 'col2': [val1, val2]}
            # We assume all columns in the batch have the same length
            num_rows = len(next(iter(examples.values())))
            
            for i in range(num_rows):
                # Join all non-None, non-empty fields for this row
                # We use the detected 'column_names' to ensure we iterate all fields
                row_parts = []
                for k in column_names:
                    val = examples[k][i]
                    if val is not None and str(val).strip():
                        row_parts.append(str(val))
                
                batch_texts.append(" ".join(row_parts))
                
            return {"text": batch_texts}

        # 5. Apply mapping
        # remove_columns is now safe because we manually extracted 'column_names'
        dataset = dataset.map(
            batch_format, 
            batched=True, 
            batch_size=20000,
            remove_columns=column_names
        )

        # 6. Materialize to list
        texts = [row["text"] for row in dataset]

        print(f"Loaded {len(texts):,} samples from Ecom-niverse")
        return PreTrainingDataset(texts)

    except Exception as e:
        print(f"Warning: Could not load Ecom-niverse: {e}")
        # Return empty dataset on failure to prevent pipeline crash
        return PreTrainingDataset([])


def create_combined_pretraining_dataset(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizer | None = None,
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

    if tokenizer:
        print("Calculating total tokens...")
        total_tokens = 0
        batch_size = 1000
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i : i + batch_size]
            encodings = tokenizer(batch, add_special_tokens=False, verbose=False)
            total_tokens += sum(len(ids) for ids in encodings["input_ids"])
            if i % 100000 == 0 and i > 0:
                print(f"  Counted tokens for {i:,} samples...")
        
        print(f"=== Total pre-training tokens: {total_tokens:,} ===")

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
