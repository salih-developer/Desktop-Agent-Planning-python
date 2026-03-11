from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".desktop_agent" / "config.yaml"


@dataclass
class AppConfig:
    ollama_base_url: str = "http://localhost:11434"
    planner_model: str = "qwen2.5:7b"
    synthesizer_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    db_path: str = str(Path.home() / ".desktop_agent" / "memory.db")
    memory_top_k: int = 5
    shell_timeout_seconds: int = 30
    web_fetch_timeout_seconds: int = 15
    max_task_retries: int = 2
    max_tool_output_chars: int = 8000
    workspace: str = str(Path.home())
    # ReAct loop settings
    use_react_loop: bool = True
    react_model: str = ""          # empty = use planner_model
    react_max_iterations: int = 15
    traces_dir: str = str(Path(__file__).parent / "logs")

    # Conversation & security
    max_conversation_history: int = 8          # how many past turns to include
    prompt_injection_protection: bool = True   # warn LLM when tool output looks like instructions

    blocked_commands: list[str] = field(default_factory=lambda: [
        r"rm\s+-rf\s+/",
        r"format\s+[a-zA-Z]:",
        r"dd\s+if=",
        r"mkfs",
        r"shutdown",
        r"reboot",
    ])


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    path = Path(path)
    if not path.exists():
        cfg = AppConfig()
        save_config(cfg, path)
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = AppConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def save_config(cfg: AppConfig, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in cfg.__dict__.items()}
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
