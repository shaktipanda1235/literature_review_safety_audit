import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional

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

    def add_edge(self, src: str, dst: str):
        self.adj.setdefault(src, []).append(dst)

    def add_conditional_edges(self, name: str, route_fn: Callable, mapping: Dict[str, str]):
        self.conditional[name] = {"route_fn": route_fn, "mapping": mapping}

    def compile(self, checkpointer: Optional[Any] = None, interrupt_before: Optional[List[str]] = None):
        return CompiledGraph(self, checkpointer=checkpointer, interrupt_before=interrupt_before or [])


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: Optional[Any] = None, interrupt_before: List[str] = None):
        self.graph = graph
        self.checkpointer = checkpointer
        self.interrupt_before = set(interrupt_before or [])

    def invoke(self, initial_state: Dict, config: Optional[Dict] = None) -> Dict:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(initial_state, config=config))
        raise RuntimeError("invoke() cannot run inside an event loop; use ainvoke() instead")

    async def ainvoke(self, initial_state: Dict, config: Optional[Dict] = None) -> Dict:
        state = dict(initial_state)
        current = None
        # Start from START
        starts = self.graph.adj.get(START, [])
        if not starts:
            return state

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

            updates = node_fn(state)
            if inspect.isawaitable(updates):
                updates = await updates
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

        # mark end
        state["__finished"] = True
        return state
