"""Configuration constants for the pharma LangGraph application."""

# API Base URLs
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2"
OPENFDA_BASE = "https://api.fda.gov/drug"
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
WEB_FALLBACK_ALLOWED_DOMAINS = (
	"fda.gov",
	"nih.gov",
	"ema.europa.eu",
	"who.int",
	"dailymed.nlm.nih.gov",
)
WEB_FALLBACK_MAX_CONTENT_CHARS = 4000

# HTTP settings
API_TIMEOUT_SECONDS = 30
MAX_RESULTS_PER_SOURCE = 10
DEFAULT_DRUG_QUERY = "metformin"
CLINICALTRIALS_PAGE_SIZE = 100
CLINICALTRIALS_MAX_PAGES = 3

# Cache settings
CACHE_TTL_SECONDS = 300  # 5 minutes

# Rate limiting
RATE_LIMIT_DELAY_SECONDS = 0.22  # Per-host rate limit delay
