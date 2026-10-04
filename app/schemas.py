"""Schemas for the pharma LangGraph application."""

from dataclasses import dataclass, asdict
from datetime import datetime
from hashlib import sha256
from typing import Optional, List


@dataclass
class EvidenceItem:
    """Normalized evidence item from any data source."""
    
    source: str  # e.g., "PubMed", "ClinicalTrials.gov", "openFDA"
    title: str
    summary: str
    url: str
    publication_date: Optional[str] = None
    study_type: Optional[str] = None
    safety_flags: List[str] = None
    raw_payload: Optional[dict] = None
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize defaults for mutable fields."""
        if self.safety_flags is None:
            self.safety_flags = []
        if self.raw_payload is None:
            self.raw_payload = {}
        if not self.doc_id:
            identity = "\0".join((self.source, self.url, self.title))
            self.doc_id = sha256(identity.encode("utf-8")).hexdigest()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
