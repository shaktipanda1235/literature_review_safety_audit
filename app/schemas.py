from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    source: str
    title: str
    summary: str
    url: str = ""
    publication_date: Optional[str] = None
    study_type: Optional[str] = None
    safety_flags: List[str] = field(default_factory=list)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "publication_date": self.publication_date,
            "study_type": self.study_type,
            "safety_flags": self.safety_flags,
            "raw_payload": self.raw_payload,
        }
