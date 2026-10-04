import yaml

from app.safety.rules import (
    findings_from_evidence,
    findings_from_faers,
    findings_from_label,
    load_safety_watchlist,
)
from app.schemas import EvidenceItem


def test_boxed_warning_creates_critical_finding_with_document_id():
    document = EvidenceItem(
        source="openFDA",
        title="Drug label - Boxed warning",
        summary="Serious hepatotoxicity has occurred.",
        url="https://api.fda.gov/drug/label.json?id=label-1",
        raw_payload={"section": "boxed_warning"},
        doc_id="label-1-boxed-warning",
    )

    findings = findings_from_label(document)

    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].supporting_doc_ids == ["label-1-boxed-warning"]
    assert findings[0].quote == document.summary


def test_contraindications_create_high_finding():
    document = EvidenceItem(
        source="openFDA",
        title="Drug label - Contraindications",
        summary="Use is contraindicated with medicine X.",
        url="https://api.fda.gov/drug/label.json?id=label-2",
        raw_payload={"section": "contraindications"},
        doc_id="label-2-contraindications",
    )

    findings = findings_from_label(document)

    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].supporting_doc_ids == ["label-2-contraindications"]


def test_faers_watchlist_creates_moderate_finding_with_caveat():
    watchlist = {"cardiac": ["cardiac arrest", "QT"]}
    document = EvidenceItem(
        source="openFDA FAERS",
        title="FAERS reaction counts for example drug",
        summary="QT prolongation: 8",
        url="https://api.fda.gov/drug/event.json",
        safety_flags=["FAERS"],
        raw_payload={
            "drug_query": "example drug",
            "reactions": [{"rank": 1, "term": "QT PROLONGATION", "count": 8}],
            "caveat": "Spontaneous reports do not establish causation or incidence.",
        },
        doc_id="faers-example-drug",
    )

    findings = findings_from_faers(document, watchlist)

    assert len(findings) == 1
    assert findings[0].category == "cardiac"
    assert findings[0].severity == "moderate"
    assert findings[0].supporting_doc_ids == ["faers-example-drug"]
    assert "do not establish causation or incidence" in findings[0].caveat
    assert "8 times" in findings[0].summary


def test_watchlist_is_loaded_from_yaml_and_nonmatching_terms_are_ignored(tmp_path):
    watchlist_path = tmp_path / "watchlist.yaml"
    watchlist_path.write_text(yaml.safe_dump({"renal": ["renal failure"]}), encoding="utf-8")
    watchlist = load_safety_watchlist(watchlist_path)
    document = EvidenceItem(
        source="openFDA FAERS",
        title="FAERS reactions",
        summary="",
        url="https://api.fda.gov/drug/event.json",
        safety_flags=["FAERS"],
        raw_payload={
            "reactions": [
                {"term": "RENAL FAILURE", "count": 4},
                {"term": "HEADACHE", "count": 20},
            ]
        },
        doc_id="faers-renal",
    )

    findings = findings_from_evidence([document], watchlist)

    assert watchlist == {"renal": ["renal failure"]}
    assert len(findings) == 1
    assert findings[0].category == "renal"