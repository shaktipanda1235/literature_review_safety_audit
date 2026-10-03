from dataclasses import dataclass


@dataclass
class BaseMessage:
    content: str


class HumanMessage(BaseMessage):
    pass


class AIMessage(BaseMessage):
    pass
