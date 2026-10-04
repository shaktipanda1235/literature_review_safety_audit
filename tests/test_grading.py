import pytest

from app import graph
from app.config import GRADE_PARSE_RETRIES
from app.schemas import EvidenceItem, GradeResult


class ScriptedStructuredGrader:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_documents():
    return [
        EvidenceItem(
            source="PubMed",
            title="Safety study",
            summary="Direct safety evidence.",
            url="https://pubmed.ncbi.nlm.nih.gov/1/",
            doc_id="doc-1",
        ),
        EvidenceItem(
            source="openFDA",
            title="Drug label",
            summary="Label safety section.",
            url="https://api.fda.gov/drug/label/1",
            doc_id="doc-2",
        ),
    ]


def make_state(documents):
    return {
        "drug_query": "metformin",
        "canonical_name": "metformin",
        "documents": documents,
        "literature_raw_data": [f"{item.source}: {item.title}" for item in documents],
        "audit_log": [],
    }


def grade_response(documents, relevant_count, sufficient=True):
    assessments = [
        {
            "doc_id": item.doc_id,
            "relevance": "relevant" if index < relevant_count else "partial",
            "rationale": "Fixture assessment",
        }
        for index, item in enumerate(documents)
    ]
    return {
        "document_relevance": assessments,
        "evidence_level": "moderate",
        "sufficient": sufficient,
        "missing_topics": [],
    }


@pytest.mark.asyncio
async def test_grade_node_deterministically_enforces_relevant_document_floor(monkeypatch):
    documents = make_documents()
    grader = ScriptedStructuredGrader([grade_response(documents, relevant_count=1)])
    monkeypatch.setattr(graph, "get_llm", lambda role: grader)

    updates = await graph.grade_node(make_state(documents))

    assert updates["grade_result"].sufficient is False
    assert updates["grade_decision"] == "fallback"
    assert updates["grade_result"].evidence_level == "moderate"
    assert len(updates["grade_result"].document_relevance) == 2
    assert any("At least 2 relevant documents" in topic for topic in updates["grade_result"].missing_topics)
    assert grader.calls == 1


@pytest.mark.asyncio
async def test_grade_node_accepts_sufficient_structured_result(monkeypatch):
    documents = make_documents()
    grader = ScriptedStructuredGrader([grade_response(documents, relevant_count=2)])
    monkeypatch.setattr(graph, "get_llm", lambda role: grader)

    updates = await graph.grade_node(make_state(documents))

    assert isinstance(updates["grade_result"], GradeResult)
    assert updates["grade_result"].sufficient is True
    assert updates["grade_decision"] == "clear"
    assert updates["evidence_level"] == "moderate"


@pytest.mark.asyncio
async def test_grade_node_retries_invalid_output_then_fails_closed(monkeypatch):
    documents = make_documents()
    grader = ScriptedStructuredGrader([{"invalid": True}, {"invalid": True}])
    monkeypatch.setattr(graph, "get_llm", lambda role: grader)

    updates = await graph.grade_node(make_state(documents))

    assert grader.calls == GRADE_PARSE_RETRIES + 1
    assert updates["grade_result"].sufficient is False
    assert updates["grade_result"].evidence_level == "none"
    assert updates["grade_decision"] == "fallback"
    assert any("LLM grading failed after retry" in entry for entry in updates["audit_log"])