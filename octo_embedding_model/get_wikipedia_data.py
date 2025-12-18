"""
Wikipedia Company Data Extraction Script.

Extracts Wikipedia pages for companies, organizations, and products
from the English Wikipedia dataset for use in training embeddings.

Features:
- Streaming mode for memory efficiency
- Comprehensive filtering for business entities
- Checkpointing with incremental saves
- Progress tracking with detailed logging
- Graceful interrupt handling

Usage:
    python get_wikipedia_data.py
    python get_wikipedia_data.py --max-pages 50000
    python get_wikipedia_data.py --force  # Re-run even if output exists
"""

import argparse
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_PATH = Path("data/filtered_wiki_companies.parquet")
DEFAULT_MAX_PAGES = 100_000
CHECKPOINT_INTERVAL = 10000  # Save checkpoint every N pages
MIN_CONTENT_LENGTH = 1000


# =============================================================================
# Filter Keywords (Compile once for performance)
# =============================================================================

# Wikipedia namespace prefixes to ignore
IGNORE_PREFIXES = (
    "User:", "Talk:", "File:", "Template:", "Category:",
    "Wikipedia:", "Help:", "Portal:", "Draft:", "MediaWiki:",
    "Book:", "TimedText:", "Module:"
)

# Title patterns to reject (compiled regex for speed)
TITLE_REJECT_PATTERNS = [
    re.compile(r"^(1[0-9]{3}|20[0-9]{2})$"),  # Years: 1900-2099
    re.compile(r"^(1[0-9]|20)[0-9]0s$"),       # Decades: 1990s
    re.compile(r"^\d+(st|nd|rd|th) century", re.IGNORECASE),
]

MONTHS = frozenset([
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
])

# Positive category keywords (indicates business/organization)
BIZ_CAT_KEYWORDS = frozenset([
    "companies", "corporations", "organizations", "businesses",
    "enterprises", "conglomerates", "holdings", "subsidiaries",
    "manufacturers", "retailers", "brands", "products", "software",
    "video games", "computing", "services", "financial", "banks", "airlines",
    "technology companies", "internet companies", "telecommunications",
    "pharmaceutical", "automotive", "energy companies", "oil companies",
    "food companies", "beverage companies", "restaurant chains",
    "hotel chains", "retail chains", "supermarket chains",
    "software companies", "video game companies", "social media",
    "e-commerce", "fintech", "startups", "unicorn companies",
    "mobile apps", "websites", "online services", "cloud computing",
    "investment banks", "hedge funds", "private equity", "venture capital",
    "insurance companies", "credit card", "payment systems",
    "consumer electronics", "home appliances", "clothing brands",
    "cosmetics brands", "luxury brands", "sporting goods",
])

# Positive intro text patterns
BIZ_INTRO_KEYWORDS = (
    "is a company", "is a corporation", "is a multinational",
    "is a brand", "is a software", "is an application", "is a platform",
    "is a service", "is a product", "is a manufacturer", "is a provider",
    "is a retailer", "is a chain", "is a conglomerate", "is an enterprise",
    "is a startup", "is a subsidiary", "is a holding company",
    "is a technology company", "is a tech company", "is an internet company",
    "is a financial services", "is a bank", "is an insurance company",
    "is a telecommunications", "is a telecom", "is a media company",
    "is a pharmaceutical", "is a biotech", "is an energy company",
    "is a consumer goods", "is a food company", "is a beverage company",
    "is an e-commerce", "is a fintech", "is a payment",
    "headquartered in", "founded in", "publicly traded",
)

# Negative category keywords (definitely not companies)
NEGATIVE_CAT_KEYWORDS = frozenset([
    # People
    "births", "deaths", "people", "alumni", "players", "politicians",
    "actresses", "actors", "singers", "musicians", "artists", "writers",
    "scientists", "mathematicians", "philosophers", "historians",
    "athletes", "sportspeople", "footballers", "basketball players",
    "coaches", "managers", "executives who", "ceos", "founders who",
    "nobel laureates", "award winners", "recipients of",
    # Places
    "cities", "towns", "villages", "municipalities", "districts",
    "counties", "states", "provinces", "regions", "countries",
    "rivers", "mountains", "lakes", "islands", "valleys", "deserts",
    "national parks", "protected areas", "landmarks", "monuments",
    "buildings in", "structures in", "architecture of",
    # Entertainment/Media (not companies)
    "films", "movies", "albums", "songs", "singles", "discographies",
    "episodes", "seasons", "television series", "tv series", "web series",
    "novels", "books", "short stories", "poems", "plays",
    "paintings", "sculptures", "artworks",
    # Sports events
    "matches", "tournaments", "championships", "olympics", "world cups",
    "seasons in", "in sports", "sports events",
    # Historical/Academic
    "wars", "battles", "conflicts", "revolutions", "uprisings",
    "treaties", "agreements", "conferences",
    "languages", "dialects", "scripts", "alphabets",
    "religions", "religious", "churches", "temples", "mosques",
    "universities", "colleges", "schools", "educational institutions",
    "hospitals", "medical centers", "government agencies",
    # Nature/Science
    "species", "genera", "animals", "plants", "fungi", "bacteria",
    "diseases", "disorders", "syndromes", "medical conditions",
    "elements", "compounds", "minerals", "chemicals",
    "astronomical", "stars", "planets", "galaxies", "nebulae",
    # Meta content
    "list of", "lists of", "index of", "glossary of", "outline of",
    "bibliography", "discography", "filmography", "awards and nominations",
])

# Biography rejection patterns
BIO_PATTERNS = (
    "was born", "born in", "born on", " died ", "died in", "died on",
    "is a city", "is a town", "is a village", "is a municipality",
    "is a river", "is a mountain", "is a lake", "is an island",
    "is a film", "is a movie", "is a song", "is an album",
    "is a novel", "is a book", "is a television", "is a tv series",
    "is a species", "is a genus", "is a disease", "is a disorder",
    "is a language", "is a religion", "is a university", "is a college",
    "was a ", "were a ",
)

# Stock exchange signals (strong positive)
STOCK_SIGNALS = (
    "nyse:", "nasdaq:", "traded on", "listed on", "stock symbol",
    "ticker symbol", "publicly traded", "stock exchange",
    "fortune 500", "fortune 1000", "forbes global",
)


# =============================================================================
# Filter Function
# =============================================================================

def is_company_or_product(example: dict) -> bool:
    """
    Filter function to identify company/organization/product pages.
    
    Uses a multi-stage filtering approach:
    1. Namespace filtering
    2. Title-based quick rejections
    3. Content length check
    4. Category-based classification
    5. Intro text analysis
    
    Args:
        example: Wikipedia page dict with 'title', 'text', and optionally 'categories'
        
    Returns:
        True if the page should be included, False otherwise
    """
    title = example.get("title", "")
    text = example.get("text", "")
    
    # Stage 1: Namespace filter
    if title.startswith(IGNORE_PREFIXES):
        return False
    
    # Stage 2: Title-based rejections
    title_lower = title.lower()
    
    # Disambiguation and list pages
    if "(disambiguation)" in title_lower:
        return False
    
    reject_prefixes = ("list of ", "lists of ", "index of ", "outline of ", 
                       "history of ", "timeline of ")
    if title_lower.startswith(reject_prefixes):
        return False
    
    # Year/date pages
    for pattern in TITLE_REJECT_PATTERNS:
        if pattern.match(title):
            return False
    
    # Month pages (e.g., "January 1")
    first_word = title_lower.split()[0] if title_lower else ""
    if first_word in MONTHS:
        return False
    
    # Stage 3: Content length
    if len(text) < MIN_CONTENT_LENGTH:
        return False
    
    # Stage 4: Category analysis
    categories_list = example.get("categories", [])
    
    if not categories_list:
        # Extract from text if not provided
        text_tail = text[-2000:]
        found_cats = re.findall(r"Category:([^\]\n]+)", text_tail, re.IGNORECASE)
        categories_list = [c.strip().lower() for c in found_cats]
    else:
        categories_list = [str(c).lower() for c in categories_list]
    
    # Check negative categories first (fast reject)
    for cat in categories_list:
        if any(neg in cat for neg in NEGATIVE_CAT_KEYWORDS):
            return False
    
    # Check positive categories
    for cat in categories_list:
        if any(biz in cat for biz in BIZ_CAT_KEYWORDS):
            return True
    
    # Stage 5: Intro text analysis
    intro_text = text[:800].lower()
    
    # Reject biography patterns
    if any(pattern in intro_text for pattern in BIO_PATTERNS):
        return False
    
    # Check positive intro patterns
    if any(phrase in intro_text for phrase in BIZ_INTRO_KEYWORDS):
        return True
    
    # Check stock exchange signals
    if any(signal in intro_text for signal in STOCK_SIGNALS):
        return True
    
    return False


# =============================================================================
# Data Extraction
# =============================================================================

class WikipediaExtractor:
    """Handles Wikipedia data extraction with checkpointing and progress tracking."""
    
    def __init__(
        self,
        output_path: Path,
        max_pages: int = DEFAULT_MAX_PAGES,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
    ):
        self.output_path = Path(output_path)
        self.max_pages = max_pages
        self.checkpoint_interval = checkpoint_interval
        self.extracted_data: list[dict] = []
        self.start_time: float = 0
        self._interrupted = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.warning("\nInterrupt received, saving progress...")
        self._interrupted = True
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save current data to parquet file."""
        if not self.extracted_data:
            return
        
        df = pd.DataFrame(self.extracted_data)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, engine="pyarrow", compression="snappy")
        
        status = "Final" if is_final else "Checkpoint"
        logger.info(f"{status} saved: {len(self.extracted_data):,} pages to {self.output_path}")
    
    def _log_progress(self, count: int, title: str):
        """Log extraction progress."""
        elapsed = time.time() - self.start_time
        rate = count / elapsed if elapsed > 0 else 0
        eta = (self.max_pages - count) / rate if rate > 0 else 0
        
        logger.info(
            f"[{count:,}/{self.max_pages:,}] {title[:50]:<50} | "
            f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m"
        )
    
    def extract(self) -> int:
        """
        Run the extraction process.
        
        Returns:
            Number of pages extracted
        """
        logger.info("Initializing Wikipedia stream (English)...")
        
        try:
            dataset = load_dataset(
                "omarkamali/wikipedia-monthly",
                "latest.en",
                split="train",
                streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return 0
        
        logger.info("Applying filters...")
        filtered_stream = dataset.filter(is_company_or_product)
        
        logger.info(f"Starting extraction (max: {self.max_pages:,} pages)...")
        self.start_time = time.time()
        
        try:
            for page in filtered_stream:
                if self._interrupted:
                    break
                
                self.extracted_data.append({
                    "title": page["title"],
                    "url": page.get("url", f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"),
                    "text": page["text"]
                })
                
                count = len(self.extracted_data)
                
                # Log progress
                if count % 1000 == 0:
                    self._log_progress(count, page["title"])
                
                # Checkpoint save
                if count % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                # Check limit
                if count >= self.max_pages:
                    logger.info(f"Reached maximum limit of {self.max_pages:,} pages")
                    break
                    
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
        
        # Final save
        self._save_checkpoint(is_final=True)
        
        elapsed = time.time() - self.start_time
        logger.info(
            f"Extraction complete: {len(self.extracted_data):,} pages "
            f"in {elapsed/60:.1f} minutes"
        )
        
        return len(self.extracted_data)


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Wikipedia company/organization pages for embedding training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--max-pages", "-n",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum number of pages to extract",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help="Save checkpoint every N pages",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-extraction even if output file exists",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if output already exists
    if args.output.exists() and not args.force:
        logger.info(f"Output file already exists: {args.output}")
        logger.info("Use --force to re-extract. Skipping.")
        return 0
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Run extraction
    extractor = WikipediaExtractor(
        output_path=args.output,
        max_pages=args.max_pages,
        checkpoint_interval=args.checkpoint_interval,
    )
    
    count = extractor.extract()
    
    if count == 0:
        logger.warning("No pages were extracted!")
        return 1
    
    logger.info(f"Successfully extracted {count:,} pages to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())