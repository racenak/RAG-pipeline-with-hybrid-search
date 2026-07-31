"""Prompt builder — assemble system and user messages for LLM consumption."""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the "
    "provided context. If the answer is not in the context, say you don't know. "
    "Cite your sources using [1], [2] format."
)

_DEFAULT_USER_TEMPLATE = (
    "Context:\n{context}\n\nQuestion: {query}"
)


@dataclass
class PromptConfig:
    """Configuration for prompt assembly."""

    system_prompt: str = ""
    user_template: str = ""
    citation_instruction: str = ""
    max_context_tokens: int = 4096

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = _DEFAULT_SYSTEM_PROMPT
        if not self.user_template:
            self.user_template = _DEFAULT_USER_TEMPLATE


class PromptBuilder:
    """Builds system + user message dicts from query and context."""

    def build_prompt(
        self,
        query: str,
        context: str,
        config: PromptConfig | None = None,
    ) -> dict[str, str]:
        """Return ``{"system": ..., "user": ...}`` for LLM consumption."""
        cfg = config or PromptConfig()

        system = cfg.system_prompt
        if cfg.citation_instruction:
            system = f"{system}\n\n{cfg.citation_instruction}"

        user = cfg.user_template.format(context=context, query=query)

        return {"system": system, "user": user}
