import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional

from langgraph.types import (
    Command,
    GraphInterrupt,
    _NO_RESUME,
    _reset_resume_value,
    _set_resume_value,
)

# Lightweight shim of a StateGraph and compile/runtime for local demo/testing.
START = "__START__"
END = "__END__"


class StateGraph:
    def __init__(self, state_type: Any = None):
        self.state_type = state_type
        self.nodes: Dict[str, Callable] = {}
        self.adj: Dict[str, List[str]] = {}
        self.conditional: Dict[str, Dict] = {}

    def add_node(self, name: str, func: Callable):
        self.nodes[name] = func
        self.adj.setdefault(name, [])

    def add_edge(self, src: str | List[str], dst: str):
        sources = src if isinstance(src, list) else [src]
        for source in sources:
            self.adj.setdefault(source, []).append(dst)

    def add_conditional_edges(self, name: str, route_fn: Callable, mapping: Dict[str, str]):
        self.conditional[name] = {"route_fn": route_fn, "mapping": mapping}

    def compile(self, checkpointer: Optional[Any] = None, interrupt_before: Optional[List[str]] = None):
        return CompiledGraph(self, checkpointer=checkpointer, interrupt_before=interrupt_before or [])


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: Optional[Any] = None, interrupt_before: List[str] = None):
        self.graph = graph
        self.checkpointer = checkpointer
        self.interrupt_before = set(interrupt_before or [])

    def invoke(self, initial_state: Dict | Command, config: Optional[Dict] = None) -> Dict:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(initial_state, config=config))
        raise RuntimeError("invoke() cannot run inside an event loop; use ainvoke() instead")

    async def ainvoke(self, initial_state: Dict | Command, config: Optional[Dict] = None) -> Dict:
        configurable = (config or {}).get("configurable", {})
        thread_id = configurable.get("thread_id")
        resuming = isinstance(initial_state, Command)
        resume_value = initial_state.resume if resuming else _NO_RESUME
        resume_node = None

        if resuming:
            if self.checkpointer is None or not thread_id:
                raise RuntimeError("Resuming an interrupt requires a checkpointer and thread_id")
            checkpoint = self.checkpointer.load_checkpoint(thread_id)
            if checkpoint is None:
                raise RuntimeError(f"No interrupted checkpoint found for thread_id '{thread_id}'")
            state = dict(checkpoint["state"])
            current = checkpoint["node"]
            resume_node = current
            state.pop("__interrupt__", None)
            state.pop("__interrupted_at", None)
        else:
            state = dict(initial_state)
            starts = self.graph.adj.get(START, [])
            if not starts:
                return state

            if len(starts) > 1:
                async def run_start_node(node_name: str):
                    node_fn = self.graph.nodes.get(node_name)
                    if node_fn is None:
                        raise KeyError(f"Node '{node_name}' is not registered")
                    updates = node_fn(state)
                    if inspect.isawaitable(updates):
                        updates = await updates
                    return updates

                branch_updates = await asyncio.gather(
                    *(run_start_node(node_name) for node_name in starts)
                )
                for updates in branch_updates:
                    if isinstance(updates, dict):
                        state.update(updates)

                common_successors = [
                    candidate
                    for candidate in self.graph.adj.get(starts[0], [])
                    if all(
                        candidate in self.graph.adj.get(node_name, [])
                        for node_name in starts[1:]
                    )
                ]
                current = common_successors[0] if common_successors else None
            else:
                current = starts[0]
        max_steps = 200
        steps = 0

        while current and current != END and steps < max_steps:
            steps += 1
            if current in self.interrupt_before:
                state["__interrupted_at"] = current
                return state

            node_fn = self.graph.nodes.get(current)
            if not node_fn:
                break

            resume_for_node = resume_value if current == resume_node else _NO_RESUME
            resume_value = _NO_RESUME
            resume_token = _set_resume_value(resume_for_node)
            try:
                updates = node_fn(state)
                if inspect.isawaitable(updates):
                    updates = await updates
            except GraphInterrupt as interruption:
                if self.checkpointer is None or not thread_id:
                    raise RuntimeError("interrupt() requires a checkpointer and thread_id")
                state["__interrupted_at"] = current
                state["__interrupt__"] = [{"value": interruption.value}]
                self.checkpointer.save_checkpoint(thread_id, current, state)
                return state
            finally:
                _reset_resume_value(resume_token)
            if isinstance(updates, dict):
                state.update(updates)

            # Conditional routing
            if current in self.graph.conditional:
                route_fn = self.graph.conditional[current]["route_fn"]
                mapping = self.graph.conditional[current]["mapping"]
                try:
                    chosen = route_fn(state)
                except Exception:
                    chosen = None
                # map chosen key to node name if present
                next_node = mapping.get(chosen, chosen)
                current = next_node
                continue

            # Normal adjacency
            next_nodes = self.graph.adj.get(current, [])
            current = next_nodes[0] if next_nodes else None

        if resuming and self.checkpointer is not None and thread_id:
            self.checkpointer.clear_checkpoint(thread_id)

        # mark end
        state["__finished"] = True
        return state
