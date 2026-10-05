"""Schemas for the pharma LangGraph application."""

from dataclasses import dataclass, asdict
from datetime import datetime
from hashlib import sha256
from typing import Dict, Literal, Optional, List

from pydantic import BaseModel, Field


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


class DocumentRelevance(BaseModel):
    """LLM assessment of one retrieved evidence item."""

    doc_id: str
    relevance: Literal["relevant", "partial", "irrelevant"]
    rationale: str = ""


class GradeResult(BaseModel):
    """Structured assessment of the relevance and sufficiency of retrieved evidence."""

    document_relevance: List[DocumentRelevance] = Field(default_factory=list)
    evidence_level: Literal["none", "weak", "moderate", "strong"]
    sufficient: bool
    missing_topics: List[str] = Field(default_factory=list)


class QueryRewriteResult(BaseModel):
    """An LLM-proposed search query based on evidence gaps."""

    search_query: str = Field(min_length=1)
    rationale: str = ""


class SafetyFinding(BaseModel):
    """A deterministic safety finding linked to one or more evidence documents."""

    category: str
    severity: Literal["critical", "high", "moderate", "info", "no_evidence"]
    summary: str
    supporting_doc_ids: List[str] = Field(default_factory=list)
    quote: Optional[str] = None
    caveat: Optional[str] = None


class SafetyExtractionCandidate(BaseModel):
    """A model-proposed finding whose evidence is validated before use."""

    category: str
    severity: Literal["critical", "high", "moderate", "info"]
    summary: str
    supporting_doc_ids: List[str] = Field(min_length=1)
    quote: str = Field(min_length=1)


class SafetyExtractionResult(BaseModel):
    """Structured output from one category-specific safety extraction call."""

    findings: List[SafetyExtractionCandidate] = Field(default_factory=list)


NON_MEDICAL_ADVICE_DISCLAIMER = (
    "For developer practice only. This literature and label summary is not medical advice "
    "and must not be used for clinical or dosing decisions."
)


class CitedClaim(BaseModel):
    """A brief statement that must cite one or more retrieved documents."""

    text: str
    doc_ids: List[str] = Field(default_factory=list)


class BriefSource(BaseModel):
    """Source information for one retrieved evidence document."""

    doc_id: str
    source: str
    title: str
    url: str


class Brief(BaseModel):
    """Structured, citation-validated regulatory evidence summary."""

    summary: Optional[CitedClaim] = None
    evidence_level: Literal["none", "weak", "moderate", "strong"]
    key_findings: List[CitedClaim] = Field(default_factory=list)
    safety_findings_by_severity: Dict[str, List[CitedClaim]] = Field(default_factory=dict)
    evidence_gaps: List[str] = Field(default_factory=list)
    sources: List[BriefSource] = Field(default_factory=list)
    disclaimer: str = NON_MEDICAL_ADVICE_DISCLAIMER


class HumanReviewResponse(BaseModel):
    """Validated response to a human review interrupt."""

    decision: Literal["approve", "edit", "reject"]
    feedback: str = ""
    edited_brief: Optional[str] = None
