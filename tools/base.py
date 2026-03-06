from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParam:
    name: str
    type: str  # "string" | "integer" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolResult:
    success: bool
    output: str
    data: Any = None
    error: str | None = None


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    params: list[ToolParam] = field(default_factory=list)

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

    def to_text_description(self) -> str:
        lines = [f"### {self.name}", self.description, "Parameters:"]
        for p in self.params:
            req = "required" if p.required else f"optional, default: {p.default}"
            enum_str = f", enum: {p.enum}" if p.enum else ""
            lines.append(f"  - {p.name} ({p.type}, {req}{enum_str}): {p.description}")
        return "\n".join(lines)
