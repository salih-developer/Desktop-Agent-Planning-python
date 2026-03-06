from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import ollama

from logger import get_logger

if TYPE_CHECKING:
    from core.executor import TaskExecution

_log = get_logger(__name__)


_SYSTEM_PROMPT = """\
You are an expert AI assistant and Senior Software Engineer.
Based on the tool execution results provided, give a clear, concise, and accurate answer to the user's original question.

CRITICAL INSTRUCTIONS:
1. The Tool Execution Results below CONTAIN the actual file contents, code, or command outputs you requested.
2. NEVER say "I don't have the ability to read files" or "Please share the content with me." The content is already provided to you in the results block.
3. Analyze the provided text/code directly and comprehensively.
4. Reference specific data or lines from the tool results when relevant.
5. If some tasks failed or were skipped, acknowledge it and provide the best answer from available results.
"""


class Synthesizer:
    def __init__(self, base_url: str, model: str):
        self._client = ollama.Client(host=base_url)
        self._model = model

    def synthesize(
        self,
        user_query: str,
        task_executions: dict[str, "TaskExecution"],
        plan_reasoning: str = "",
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        context_block = self._build_context(task_executions, plan_reasoning)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"## Tool Execution Results\n{context_block}\n\n## User Question\n{user_query}",
            },
        ]

        _log.debug("SYNTHESIZER CONTEXT:\n%s", context_block)

        response_text = ""
        stream = self._client.chat(model=self._model, messages=messages, stream=True)
        for chunk in stream:
            token = chunk.message.content or ""
            response_text += token
            if on_chunk:
                on_chunk(token)

        _log.info("Synthesizer produced %d chars", len(response_text))
        _log.debug("SYNTHESIZER FULL RESPONSE:\n%s", response_text)
        return response_text

    def _build_context(
        self, executions: dict[str, "TaskExecution"], plan_reasoning: str
    ) -> str:
        parts = []

        if plan_reasoning:
            parts.append(f"### Plan Reasoning\n{plan_reasoning}")

        for task_id, exec_ in executions.items():
            status = exec_.status.value
            tool = exec_.task.tool
            desc = exec_.task.description

            header = f"<result id='{task_id}' tool='{tool}' status='{status}'>"
            header += f"\nDescription: {desc}"

            if exec_.status.value == "skipped":
                body = "(skipped — an upstream dependency failed or was cancelled)"
            elif exec_.result is None:
                body = "(no result)"
            elif not exec_.result.success:
                body = f"ERROR: {exec_.result.error or '(unknown error)'}"
                if exec_.result.output:
                    body += f"\nPartial output:\n{exec_.result.output}"
            else:
                body = exec_.result.output or "(empty output)"

            parts.append(f"{header}\n{body}\n</result>")

        return "\n\n".join(parts)
