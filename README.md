# Desktop Agent

A locally-running, Ollama-powered desktop AI assistant. Built on a ReAct loop with native tool calling, supporting file operations, web search, shell execution, remote SSH access, and persistent semantic memory.

---

## Features

### ReAct Loop
- **Observe → Think → Act** loop for step-by-step task solving
- Ollama native tool calling support
- Text-format tool call fallback: models that emit `<function=...>` syntax instead of native calls are handled automatically
- Configurable maximum iteration count (default: 25)
- Workspace injection: agent always knows its working directory — no wasted iterations discovering it
- Detailed `.log` trace file written to `logs/` for every session

### Tools
| Tool | Description |
|------|-------------|
| `file_read` | Reads files; supports line range for large files |
| `file_write` | Creates or overwrites a file |
| `file_edit` | In-place text block replacement within a file |
| `shell_run` | Runs shell commands with timeout and cancel support |
| `web_search` | Web search via DuckDuckGo |
| `web_fetch` | Fetches URL content (HTML → plain text) |
| `dir_tree` | Outputs a directory tree |
| `find_files` | Searches for files by glob pattern |
| `ssh_run` | Executes commands on remote hosts via SSH (password or key auth) |

### Vision & Multimodal
- Attach **images** (PNG, JPG, JPEG, GIF, BMP, WEBP, TIFF) — encoded as base64 and sent directly to the vision model
- Attach **videos** (MP4, AVI, MOV, MKV, WEBM, FLV, WMV, M4V) — evenly-spaced frames are extracted and passed to the vision model
  - Frame extraction uses **OpenCV** (`opencv-python`) with automatic fallback to **ffmpeg**
- **OCR fallback**: if `pytesseract` + `Pillow` are installed, text is extracted from images as supplementary context
- Dedicated `vision_model` config field — switches model automatically when media is attached, then restores the default model after the response
- Tested with **qwen3.5:35b** (multimodal: vision + tools + thinking)

### Semantic Memory
- Every conversation is stored in a `sqlite-vec` vector database
- 768-dimensional embeddings via `nomic-embed-text`
- On each new query, the most relevant past interactions are retrieved via KNN and injected as context
- `memory_max_distance` threshold filters out irrelevant memories (default: 0.8)
- Memory database: `~/.desktop_agent/memory.db`

### Remote SSH Access
- `ssh_run` tool lets the agent execute commands on remote servers (Kubernetes nodes, Docker hosts, CI machines)
- Supports password authentication and private key files
- Useful for: checking pod status, tailing logs, inspecting Docker images, running `kubectl` commands

### Docker & Kubernetes Deployment
The agent's system prompt includes a built-in deployment guide:
- Multi-project .NET Dockerfile templates (correct `sdk` vs `aspnet` image stages)
- Automatic OS dependency detection (`libreoffice`, `imagemagick`, `ffmpeg`, `wkhtmltopdf`)
- Full k8s file checklist: `deployment.yaml`, `service.yaml`, `configmap.yaml`, `secret.yaml`, `pvc.yaml`, `ingress.yaml`, `registry-secret.yaml`
- Azure DevOps pipeline (`azure-pipelines.yml`) generation
- Self-verification step before final answer

### Security
- **Prompt Injection Protection**: detects embedded instruction patterns in tool output (`ignore previous`, `act as`, `you are`, etc.) — flags them with `[⚠ INJECTION WARNING]` and alerts the LLM
- Injection warnings are highlighted in orange in the UI
- **Blocked commands**: destructive commands such as `rm -rf /`, `format`, `dd if=` are blocked from execution
- `prompt_injection_protection` can be toggled in settings

### File & Attachment Handling
- Files, folders, images, and videos can be attached to the chat (📎 button)
- Supported: PNG, JPG, PDF, TXT, MD, CS, SQL, JSON, YAML, MP4, MKV, MOV, and more
- Large files (> 50 KB) are indexed into a session RAG store and retrieved via semantic search
- Small files are injected directly into the prompt

### UI
- Desktop application built with **CustomTkinter**
- Tool calls are shown inline within the chat stream (`⚙ tool calls` block)
- Flow reads top-to-bottom: user message → tool calls → agent answer
- Task tracker panel: status and elapsed time for each step
- **Stop** button to cancel a running agent mid-execution
- Attachment chips: 🖼️ images, 🎬 videos, 📁 folders, 📄 files
- Settings dialog: model, workspace, timeout, iteration limit, memory size, and security options

### Configuration
Settings are stored in `~/.desktop_agent/config.yaml`:

```yaml
ollama_base_url: http://localhost:11434
react_model: qwen3-coder-32k:latest
vision_model: qwen3.5:35b          # used automatically when images/video are attached
planner_model: qwen3-coder-32k:latest
synthesizer_model: qwen3-coder-32k:latest
react_max_iterations: 25
workspace: C:\source
max_conversation_history: 8
max_tool_output_chars: 8000
prompt_injection_protection: true
use_react_loop: true
embedding_model: nomic-embed-text
memory_top_k: 5
memory_max_distance: 0.8
shell_timeout_seconds: 30
```

---

## Getting Started

```bash
pip install -r requirements.txt

# Required models
ollama pull nomic-embed-text
ollama pull qwen3-coder-32k:latest

# Optional: vision model (for image/video support)
ollama pull qwen3.5:35b

python main.py
```

> Ollama must be running at `http://localhost:11434`.

### Optional dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video frame extraction |
| `pytesseract` + `Pillow` | OCR fallback for images |
| `paramiko` | SSH remote access |
| `pdfplumber` | PDF text extraction |

---

## Project Structure

```
DesktopAgentPlanning/
├── main.py                  # Entry point
├── config.py                # AppConfig dataclass + YAML loading
├── logger.py                # Logging setup
├── core/
│   ├── agent.py             # Orchestrator
│   ├── react_loop.py        # ReAct loop engine
│   ├── react_tracer.py      # Session trace writer
│   ├── vision.py            # Image encoding + video frame extraction
│   ├── planner.py           # Task planner (Pydantic)
│   ├── executor.py          # Parallel task runner
│   └── synthesizer.py       # Streaming final answer
├── memory/
│   ├── store.py             # sqlite-vec vector store
│   └── embedder.py          # Ollama embedding client
├── tools/
│   ├── registry.py          # Tool registry and factory
│   ├── file_tools.py        # file_read / file_write / file_edit
│   ├── shell_tool.py        # shell_run
│   ├── web_tools.py         # web_search / web_fetch
│   ├── filesystem_tool.py   # dir_tree / find_files
│   └── ssh_tool.py          # ssh_run (remote SSH execution)
├── ui/
│   ├── app.py               # CTk root, threading bridge, SettingsDialog
│   └── panels/
│       ├── chat_panel.py    # Chat view + attachment management
│       └── task_panel.py    # Task status list
│   └── widgets/
│       ├── message_bubble.py
│       ├── task_item.py
│       └── tool_output_block.py
└── logs/                    # Session trace files (.log)
```
