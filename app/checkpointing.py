from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from app.config import (
    CHECKPOINT_POSTGRES_URL,
    CHECKPOINT_SQLITE_PATH,
    CHECKPOINTER_BACKEND,
)


@asynccontextmanager
async def create_checkpointer(
    backend: str | None = None,
    *,
    sqlite_path: str | Path | None = None,
    postgres_url: str | None = None,
) -> AsyncIterator[object]:
    """Open, set up, and close an official async LangGraph checkpointer."""
    selected_backend = (backend or CHECKPOINTER_BACKEND).strip().casefold()

    if selected_backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        database_path = str(
            sqlite_path if sqlite_path is not None else CHECKPOINT_SQLITE_PATH
        )
        if database_path != ":memory:":
            Path(database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(database_path) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
        return

    if selected_backend in {"postgres", "postgresql"}:
        connection_string = (
            postgres_url if postgres_url is not None else CHECKPOINT_POSTGRES_URL
        )
        if not connection_string:
            raise ValueError(
                "CHECKPOINT_POSTGRES_URL or DATABASE_URL is required when "
                "CHECKPOINTER_BACKEND=postgres"
            )

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(connection_string) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
        return

    raise ValueError(
        f"Unsupported CHECKPOINTER_BACKEND '{selected_backend}'; use 'sqlite' or 'postgres'"
    )