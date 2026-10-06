# Pharma LangGraph Demo

This project is a small, runnable demo of the architecture described in `pharma_langgraph_architecture.md`.

This project uses the official LangGraph runtime with configurable SQLite or PostgreSQL checkpointing. It includes a CLI demo, a FastAPI service, and offline tests.

Run the demo:

Windows PowerShell / cmd:

```bash
python -m app.main
```

Run the API:

```bash
python -m uvicorn app.api:app --reload
```

The interactive API schema is available at `http://127.0.0.1:8000/docs`.

Run tests:

```bash
pytest -q
```
