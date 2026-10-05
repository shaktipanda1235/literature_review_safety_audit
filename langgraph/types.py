from contextvars import ContextVar, Token
from typing import Any


_NO_RESUME = object()
_resume_value: ContextVar[Any] = ContextVar("langgraph_resume_value", default=_NO_RESUME)


class GraphInterrupt(Exception):
    def __init__(self, value: Any):
        super().__init__("Graph execution interrupted")
        self.value = value


class Command:
    def __init__(self, *, resume: Any):
        self.resume = resume


def interrupt(value: Any, response_schema: Any = None) -> Any:
    """Pause local shim execution or return the value supplied by Command(resume=...)."""
    response = _resume_value.get()
    if response is _NO_RESUME:
        raise GraphInterrupt(value)
    _resume_value.set(_NO_RESUME)
    return response


def _set_resume_value(value: Any) -> Token:
    return _resume_value.set(value)


def _reset_resume_value(token: Token) -> None:
    _resume_value.reset(token)