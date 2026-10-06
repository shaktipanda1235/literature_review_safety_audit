import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app import graph
from app.api import create_app
from app.schemas import Brief, CitedClaim, EvidenceItem


async def wait_for_run_status(client, thread_id, expected_status):
    for _ in range(200):
        response = await client.get(f"/runs/{thread_id}")
        if response.status_code == 200 and response.json()["status"] == expected_status:
            return response
        await asyncio.sleep(0.01)
    pytest.fail(f"Run {thread_id} did not reach status {expected_status}")


@pytest.mark.asyncio
async def test_api_start_pending_review_stream_resume_and_complete():
    async def normalize(state):
        return {"canonical_name": state["drug_query"], "synonyms": []}

    async def retrieve(state):
        return {
            "documents": [
                EvidenceItem(
                    source="mock",
                    title="Mock evidence",
                    summary="A mocked evidence record.",
                    url="https://example.test/evidence",
                    doc_id="mock-1",
                )
            ]
        }

    async def grade(state):
        return {"grade_decision": "clear", "evidence_level": "moderate"}

    async def safety_audit(state):
        return {"safety_findings": [], "risk_level": "unknown"}

    async def generate(state):
        return {
            "brief": Brief(
                summary=CitedClaim(
                    text="Mock evidence was retrieved.", doc_ids=["mock-1"]
                ),
                evidence_level="moderate",
            ),
            "regulatory_brief": "Mock regulatory brief.",
        }

    def graph_builder(checkpointer):
        return graph.build_compiled_graph(
            checkpointer,
            node_overrides={
                "normalize_node": normalize,
                "retrieve_node": retrieve,
                "grade_node": grade,
                "safety_audit_node": safety_audit,
                "generator_node": generate,
            },
        )

    @asynccontextmanager
    async def checkpointer_context_factory():
        async with InMemorySaver() as checkpointer:
            yield checkpointer

    application = create_app(
        checkpointer_context_factory=checkpointer_context_factory,
        graph_builder=graph_builder,
    )

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            started = await client.post(
                "/runs", json={"drug_query": "test compound", "thread_id": "api-run-1"}
            )
            assert started.status_code == 202
            assert started.json() == {"thread_id": "api-run-1", "status": "running"}

            stream = await client.get("/runs/api-run-1/stream")
            assert stream.status_code == 200
            assert "event: progress" in stream.text
            assert "generator_node" in stream.text

            status = await wait_for_run_status(client, "api-run-1", "pending_review")
            assert status.json()["state"]["regulatory_brief"] == "Mock regulatory brief."

            resumed = await client.post(
                "/runs/api-run-1/resume", json={"decision": "approve"}
            )
            assert resumed.status_code == 200
            assert resumed.json()["status"] == "running"
            completed = await wait_for_run_status(client, "api-run-1", "completed")
            assert completed.json()["state"]["human_approved"] is True


@pytest.mark.asyncio
async def test_api_returns_not_found_for_unknown_run():
    @asynccontextmanager
    async def checkpointer_context_factory():
        async with InMemorySaver() as checkpointer:
            yield checkpointer

    application = create_app(checkpointer_context_factory=checkpointer_context_factory)
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/runs/missing")

    assert response.status_code == 404