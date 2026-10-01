from __future__ import annotations

from typing import Any


class OpenAIResponsesProvider:
    """Thin adapter around the OpenAI Responses API."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client

    def create_response(
        self,
        system_instructions: str,
        context: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> Any:
        return self.client.responses.create(
            model=self._model(),
            instructions=system_instructions,
            input=self._input(context),
            tools=tools,
            tool_choice="auto",
        )

    def continue_response(
        self,
        previous_response_id: str,
        tool_outputs: list[dict[str, Any]],
        system_instructions: str,
    ) -> Any:
        return self.client.responses.create(
            model=self._model(),
            instructions=system_instructions,
            previous_response_id=previous_response_id,
            input=tool_outputs,
        )

    @staticmethod
    def _model() -> str:
        import os
        return os.getenv("LLM_MODEL", "gpt-5.6-luna")

    @staticmethod
    def _input(context: dict[str, Any]) -> str:
        import json
        return json.dumps(context, default=str)
