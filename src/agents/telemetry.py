from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class AgentTelemetry:
    started_at: float = field(default_factory=time.perf_counter)
    completed_at: float | None = None
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def finish(self) -> None:
        self.completed_at = time.perf_counter()

    @property
    def latency_seconds(self) -> float | None:
        if self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "latency_seconds": self.latency_seconds,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
