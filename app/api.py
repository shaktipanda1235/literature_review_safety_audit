from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.checkpointing import create_checkpointer
from app.config import DEFAULT_DRUG_QUERY
from app.graph import PharmaGraphState, build_compiled_graph

RunStatus = Literal[
    "running",
    "pending_review",
    "completed",
    "needs_manual_review",
    "rejected",
    "failed",
]


class RunCreateRequest(BaseModel):
    """Input for creating a graph run."""

    drug_query: str = Field(default=DEFAULT_DRUG_QUERY, min_length=1, max_length=200)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)


class RunResumeRequest(BaseModel):
    """Human decision submitted to an interrupted review node."""

    decision: Literal["approve", "edit", "reject"]
    feedback: str = Field(default="", max_length=10_000)
    edited_brief: str | None = Field(default=None, max_length=100_000)


class RunStatusResponse(BaseModel):
    """Status and a small, JSON-safe view of the current graph state."""

    thread_id: str
    status: RunStatus
    state: dict[str, Any]


class RunStartResponse(BaseModel):
    """Identifier and initial status returned when starting a graph run."""

    thread_id: str
    status: RunStatus


GraphBuilder = Callable[[Any], Any]
CheckpointerContextFactory = Callable[[], Any]


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _snapshot_status(
    snapshot: Any,
    thread_id: str,
    *,
    active: bool = False,
    error: str | None = None,
) -> RunStatusResponse:
    values = snapshot.values or {}
    if error:
        status = "failed"
    elif any(getattr(task, "interrupts", ()) for task in snapshot.tasks):
        status = "pending_review"
    elif active or snapshot.next:
        status = "running"
    elif values.get("review_status") == "needs_manual_review":
        status = "needs_manual_review"
    elif values.get("review_status") == "rejected":
        status = "rejected"
    else:
        status = "completed"

    summary_fields = (
        "drug_query",
        "canonical_name",
        "regulatory_brief",
        "risk_level",
        "safety_findings",
        "human_decision",
        "human_rounds",
        "human_approved",
        "review_status",
        "audit_log",
    )
    return RunStatusResponse(
        thread_id=thread_id,
        status=status,
        state=jsonable_encoder({
            **{field: values[field] for field in summary_fields if field in values},
            **({"error": error} if error else {}),
        }),
    )


def _has_run(application: FastAPI, snapshot: Any, thread_id: str) -> bool:
    return bool(
        thread_id in application.state.started_runs
        or snapshot.values
        or snapshot.next
        or snapshot.tasks
    )


def _publish_event(app: FastAPI, thread_id: str, update: Any) -> None:
    history = app.state.run_events.setdefault(thread_id, [])
    event_id = len(history)
    data = jsonable_encoder(update)
    history.append(data)
    for queue in app.state.run_subscribers.get(thread_id, set()):
        queue.put_nowait((event_id, data))


async def _run_graph(application: FastAPI, thread_id: str, graph_input: Any) -> None:
    try:
        async for update in application.state.graph.astream(
            graph_input,
            config=_thread_config(thread_id),
            stream_mode="updates",
        ):
            _publish_event(application, thread_id, update)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        application.state.run_errors[thread_id] = f"{type(error).__name__}: {error}"
    finally:
        application.state.active_runs.discard(thread_id)
        application.state.run_tasks.pop(thread_id, None)
        for queue in application.state.run_subscribers.get(thread_id, set()):
            queue.put_nowait(None)


def _schedule_graph(application: FastAPI, thread_id: str, graph_input: Any) -> None:
    application.state.started_runs.add(thread_id)
    application.state.active_runs.add(thread_id)
    application.state.run_events.setdefault(thread_id, [])
    application.state.run_errors.pop(thread_id, None)
    application.state.run_tasks[thread_id] = asyncio.create_task(
        _run_graph(application, thread_id, graph_input)
    )


def create_app(
    *,
    checkpointer_context_factory: CheckpointerContextFactory = create_checkpointer,
    graph_builder: GraphBuilder = build_compiled_graph,
) -> FastAPI:
    """Create the API and keep its async checkpointer open for the app lifetime."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with checkpointer_context_factory() as checkpointer:
            application.state.graph = graph_builder(checkpointer)
            application.state.run_events = {}
            application.state.run_subscribers = {}
            application.state.started_runs = set()
            application.state.active_runs = set()
            application.state.run_errors = {}
            application.state.run_tasks = {}
            yield
            tasks = list(application.state.run_tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    application = FastAPI(title="Pharma Evidence Review API", lifespan=lifespan)

    @application.post("/runs", response_model=RunStartResponse, status_code=202)
    async def start_run(payload: RunCreateRequest, request: Request) -> RunStartResponse:
        thread_id = payload.thread_id or str(uuid4())
        config = _thread_config(thread_id)
        application = request.app
        existing = await application.state.graph.aget_state(config)
        if _has_run(application, existing, thread_id):
            raise HTTPException(status_code=409, detail="thread_id already exists")

        initial_state: PharmaGraphState = {
            "drug_query": payload.drug_query.strip(),
            "literature_raw_data": [],
            "web_fallback_data": [],
            "safety_violations": [],
            "regulatory_brief": "",
            "grade_decision": "clear",
            "human_approved": False,
            "chat_history": [],
        }
        _schedule_graph(application, thread_id, initial_state)
        return RunStartResponse(thread_id=thread_id, status="running")

    @application.get("/runs/{thread_id}", response_model=RunStatusResponse)
    async def get_run(thread_id: str, request: Request) -> RunStatusResponse:
        snapshot = await request.app.state.graph.aget_state(_thread_config(thread_id))
        application = request.app
        if not _has_run(application, snapshot, thread_id):
            raise HTTPException(status_code=404, detail="run not found")
        return _snapshot_status(
            snapshot,
            thread_id,
            active=thread_id in application.state.active_runs,
            error=application.state.run_errors.get(thread_id),
        )

    @application.get("/runs/{thread_id}/stream")
    async def stream_run(thread_id: str, request: Request) -> StreamingResponse:
        snapshot = await request.app.state.graph.aget_state(_thread_config(thread_id))
        application = request.app
        if not _has_run(application, snapshot, thread_id):
            raise HTTPException(status_code=404, detail="run not found")

        async def event_stream() -> AsyncIterator[str]:
            queue: asyncio.Queue = asyncio.Queue()
            subscribers = application.state.run_subscribers.setdefault(thread_id, set())
            subscribers.add(queue)
            watch_active = thread_id in application.state.active_runs
            history = list(application.state.run_events.get(thread_id, []))
            next_event_id = len(history)
            try:
                for event_id, update in enumerate(history):
                    data = json.dumps(update, ensure_ascii=False)
                    yield f"id: {event_id}\nevent: progress\ndata: {data}\n\n"

                if watch_active:
                    while True:
                        event = await queue.get()
                        if event is None:
                            break
                        event_id, update = event
                        if event_id < next_event_id:
                            continue
                        data = json.dumps(update, ensure_ascii=False)
                        yield f"id: {event_id}\nevent: progress\ndata: {data}\n\n"
                        next_event_id = event_id + 1

                final_snapshot = await application.state.graph.aget_state(
                    _thread_config(thread_id)
                )
                status = _snapshot_status(
                    final_snapshot,
                    thread_id,
                    active=thread_id in application.state.active_runs,
                    error=application.state.run_errors.get(thread_id),
                )
                yield f"event: status\ndata: {status.model_dump_json()}\n\n"
            finally:
                subscribers.discard(queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @application.post("/runs/{thread_id}/resume", response_model=RunStartResponse)
    async def resume_run(
        thread_id: str, payload: RunResumeRequest, request: Request
    ) -> RunStatusResponse:
        graph = request.app.state.graph
        config = _thread_config(thread_id)
        snapshot = await graph.aget_state(config)
        application = request.app
        if not _has_run(application, snapshot, thread_id):
            raise HTTPException(status_code=404, detail="run not found")
        status = _snapshot_status(
            snapshot,
            thread_id,
            active=thread_id in application.state.active_runs,
            error=application.state.run_errors.get(thread_id),
        )
        if status.status != "pending_review":
            raise HTTPException(status_code=409, detail="run is not waiting for review")

        _schedule_graph(
            application,
            thread_id,
            Command(resume=payload.model_dump(exclude_none=True)),
        )
        return RunStartResponse(thread_id=thread_id, status="running")

    return application


app = create_app()