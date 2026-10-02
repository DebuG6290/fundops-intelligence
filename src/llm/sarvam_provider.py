from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from typing import Any


class LLMConfigurationError(RuntimeError):
    """Raised when live inference is requested without credentials."""


class SarvamProvider:
    """Sarvam Chat Completion V1 adapter for the repository's tool loop.

    The Sarvam SDK remains isolated here. The rest of the application
    consumes the same small response protocol as the existing
    investigation loop.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_tokens: int = 4000,
        max_retries: int = 2,
    ) -> None:
        api_key = os.getenv("SARVAM_API_KEY")

        if client is None and not api_key:
            raise LLMConfigurationError(
                "Live Sarvam mode needs SARVAM_API_KEY. "
                "Use Reproducible Demo instead."
            )

        self.model = model or os.getenv(
            "SARVAM_MODEL",
            "sarvam-105b",
        )

        self.timeout_seconds = timeout_seconds or float(
            os.getenv(
                "SARVAM_TIMEOUT_SECONDS",
                "45",
            )
        )

        self.max_tokens = max_tokens
        self.max_retries = max_retries

        if client is None:
            from sarvamai import SarvamAI

            client = SarvamAI(
                api_subscription_key=api_key,
                timeout=self.timeout_seconds,
            )

        self.client = client

        self._conversations: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        self._tools_by_response: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        self.last_call: dict[str, Any] = {}

    def create_response(
        self,
        system_instructions: str,
        context: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Create the initial model response."""

        messages = [
            {
                "role": "system",
                "content": system_instructions,
            },
            {
                "role": "user",
                "content": json.dumps(
                    context,
                    default=str,
                ),
            },
        ]

        return self._request(
            messages,
            tools,
        )

    def continue_response(
        self,
        previous_response_id: str,
        tool_outputs: list[dict[str, Any]],
        system_instructions: str,
    ) -> Any:
        """Continue a conversation after tool execution."""

        if previous_response_id not in self._conversations:
            raise RuntimeError(
                "Unknown Sarvam conversation response ID: "
                f"{previous_response_id}"
            )

        messages = self._conversations[
            previous_response_id
        ]

        for output in tool_outputs:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": output["call_id"],
                    "content": output["output"],
                }
            )

        return self._request(
            messages,
            self._tools_by_response.get(
                previous_response_id,
                [],
            ),
        )

    def repair_structured_response(
        self,
        *,
        invalid_output: str,
        validation_error: str,
        system_instructions: str,
    ) -> Any:
        """Make one bounded schema-repair attempt.

        The original answer is treated as untrusted data.
        The returned content is still parsed and validated by
        the caller.
        """

        messages = [
            {
                "role": "system",
                "content": system_instructions,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "invalid_output": invalid_output,
                        "validation_error": validation_error[:1000],
                        "instruction": (
                            "Return ONLY a corrected JSON object "
                            "matching the requested schema. "
                            "Do not add explanation, markdown, "
                            "code fences, or commentary."
                        ),
                    }
                ),
            },
        ]

        return self._request(
            messages,
            [],
        )

    def _request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Send a Chat Completion V1 request to Sarvam."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_object",
            },
            # Disable reasoning for this structured-output tool loop.
            # This leaves more of the completion budget for the actual
            # InvestigationReport JSON.
            "reasoning_effort": None,
            "request_options": {
                "timeout_in_seconds": int(
                    self.timeout_seconds
                ),
                "max_retries": self.max_retries,
            },
        }

        if tools:
            kwargs["tools"] = [
                self._to_chat_tool(item)
                for item in tools
            ]

            kwargs["tool_choice"] = "auto"

        started = time.perf_counter()

        try:
            response = self.client.chat.completions(
                **kwargs
            )

        except Exception as exc:
            key = os.getenv(
                "SARVAM_API_KEY",
                "",
            )

            detail = (
                str(exc).replace(
                    key,
                    "[redacted]",
                )
                if key
                else str(exc)
            )

            raise RuntimeError(
                "Sarvam request failed "
                f"({type(exc).__name__}): "
                f"{detail[:500]}"
            ) from None

        latency = time.perf_counter() - started

        response_id = _get(
            response,
            "id",
            "",
        )

        choices = _get(
            response,
            "choices",
            [],
        ) or []

        choice = (
            choices[0]
            if choices
            else None
        )

        message = _get(
            choice,
            "message",
            None,
        )

        content = _get(
            message,
            "content",
            "",
        ) or ""

        calls = _get(
            message,
            "tool_calls",
            None,
        ) or []

        output = [
            SimpleNamespace(
                type="function_call",
                name=_get(
                    _get(
                        call,
                        "function",
                        None,
                    ),
                    "name",
                    "",
                ),
                arguments=_get(
                    _get(
                        call,
                        "function",
                        None,
                    ),
                    "arguments",
                    "{}",
                ),
                call_id=_get(
                    call,
                    "id",
                    "",
                ),
            )
            for call in calls
        ]

        self._conversations[response_id] = list(
            messages
        )

        self._tools_by_response[response_id] = list(
            tools
        )

        usage = _get(
            response,
            "usage",
            None,
        )

        self.last_call = {
            "provider": "sarvam",
            "model": self.model,
            "request_id": response_id,
            "latency_seconds": latency,
            "input_tokens": _get(
                usage,
                "prompt_tokens",
                None,
            ),
            "output_tokens": _get(
                usage,
                "completion_tokens",
                None,
            ),
        }

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }

        if calls:
            assistant_message["tool_calls"] = [
                {
                    "id": _get(
                        call,
                        "id",
                        "",
                    ),
                    "type": "function",
                    "function": {
                        "name": _get(
                            _get(
                                call,
                                "function",
                                None,
                            ),
                            "name",
                            "",
                        ),
                        "arguments": _get(
                            _get(
                                call,
                                "function",
                                None,
                            ),
                            "arguments",
                            "{}",
                        ),
                    },
                }
                for call in calls
            ]

        messages.append(
            assistant_message
        )

        return SimpleNamespace(
            id=response_id,
            output=output,
            output_text=content,
        )

    @staticmethod
    def _to_chat_tool(
        tool: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert repository tool schema to Chat V1 format."""

        if "function" in tool:
            return tool

        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get(
                    "description",
                    "",
                ),
                "parameters": tool.get(
                    "parameters",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            },
        }


def _get(
    value: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Safely read an attribute from either a dict or SDK object."""

    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(
            key,
            default,
        )

    return getattr(
        value,
        key,
        default,
    )