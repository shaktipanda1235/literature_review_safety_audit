import json
from typing import Any, Dict


class MemorySaver:
    """Minimal in-memory checkpointer used for the demo.

    This stores state snapshots keyed by a thread id in a dict and
    serializes to a JSON string when asked. It's intentionally tiny.
    """

    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def save(self, thread_id: str, state: Dict[str, Any]):
        self.store[thread_id] = dict(state)

    def load(self, thread_id: str) -> Dict[str, Any]:
        return dict(self.store.get(thread_id, {}))

    def dumps(self) -> str:
        return json.dumps(self.store)
