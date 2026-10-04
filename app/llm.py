from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, DEFAULT_LLM_TEMPERATURE


PROMPT_DIR = Path(__file__).parent / "prompts"
DETERMINISTIC_ROLES = {"grader", "auditor"}


def get_llm(role: str = "default"):
    """Build a LangChain chat model using provider/model environment settings.

    Provider-specific credentials are read by the corresponding LangChain
    integration from its conventional environment variable.
    """
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip()
    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip()
    if not provider:
        raise ValueError("LLM_PROVIDER must not be empty")
    if not model:
        raise ValueError("LLM_MODEL must not be empty")

    if role.casefold() in DETERMINISTIC_ROLES:
        temperature = 0.0
    else:
        raw_temperature = os.getenv("LLM_TEMPERATURE")
        try:
            temperature = (
                float(raw_temperature)
                if raw_temperature is not None
                else DEFAULT_LLM_TEMPERATURE
            )
        except ValueError as error:
            raise ValueError("LLM_TEMPERATURE must be a number") from error

    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
    )


def load_prompt(name: str) -> str:
    """Load a Markdown prompt by name from the packaged prompts directory."""
    prompt_name = name if name.endswith(".md") else f"{name}.md"
    if not name or Path(prompt_name).name != prompt_name:
        raise ValueError("Prompt name must be a plain filename, not a path")

    prompt_path = (PROMPT_DIR / prompt_name).resolve()
    if prompt_path.parent != PROMPT_DIR.resolve():
        raise ValueError("Prompt path must stay inside the prompts directory")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt '{prompt_name}' was not found in {PROMPT_DIR}")
    return prompt_path.read_text(encoding="utf-8").strip()


def create_fake_llm(responses: Sequence[str]) -> FakeListChatModel:
    """Create a deterministic LangChain fake chat model for tests and demos."""
    if not responses:
        raise ValueError("At least one fake response is required")
    return FakeListChatModel(responses=list(responses))