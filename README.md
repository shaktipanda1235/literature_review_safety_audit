# Pharma LangGraph Demo

This project is a small, runnable demo of the architecture described in `pharma_langgraph_architecture.md`.

What I added:
- A tiny local shim for `langgraph` and `langchain_core` so the demo runs without external services.
- A runnable demo at `app/main.py` that compiles and invokes the state graph.
- A `TUTORIAL.md` that walks through the components and why they exist.
- A pytest test to validate the workflow runs.

Run the demo:

Windows PowerShell / cmd:

```bash
python -m app.main
```

Run tests:

```bash
pytest -q
```
