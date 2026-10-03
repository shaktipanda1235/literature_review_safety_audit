# Tutorial: Learning LangGraph via the Pharma Demo

This guide walks through the demo implementation and why each part exists.

1) Purpose
- Teach the LangGraph-style state machine and routing using a concrete pharma safety review workflow.

2) Key files
- `app/graph.py`: The workflow nodes, routing functions, and graph construction.
- `app/main.py`: Minimal runner that invokes the compiled graph with an initial state.
- `langgraph/`: A local shim implementing a tiny `StateGraph`, `CompiledGraph`, and `MemorySaver` so the demo runs offline.
- `langchain_core/messages.py`: Minimal `BaseMessage`/`HumanMessage`/`AIMessage` used by state typing.

3) How the demo works (high level)
- Nodes are plain Python functions that accept a state dict and return updates.
- `StateGraph` stores nodes and edges. Conditional edges use a routing function that inspects the current state and returns a key which is mapped to the next node.
- `CompiledGraph.invoke()` runs nodes sequentially, applying updates and following conditional decisions.

4) Learning checkpoints / exercises
- Change `mock_db` entries in `app/graph.py` and observe routing decisions.
- Add a new node that performs an additional check (e.g., drug-drug interaction) and wire it into the graph.
- Replace the shimbed `langgraph` with the real package once installed and compare behavior.

5) Next steps for production hardening
- Add schema validation (Pydantic) for state shapes.
- Implement asynchronous subgraphs for parallel safety checks.
- Add a human-in-the-loop web UI to review interrupted states and resume execution.
