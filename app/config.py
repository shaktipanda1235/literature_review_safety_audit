"""Configuration constants for the pharma LangGraph application."""

# API Base URLs
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2"
OPENFDA_BASE = "https://api.fda.gov/drug"

# HTTP settings
API_TIMEOUT_SECONDS = 30
MAX_RESULTS_PER_SOURCE = 10
DEFAULT_DRUG_QUERY = "Compound-X"

# Cache settings
CACHE_TTL_SECONDS = 300  # 5 minutes

# Rate limiting
RATE_LIMIT_DELAY_SECONDS = 0.22  # Per-host rate limit delay
