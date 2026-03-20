from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING

import customtkinter as ctk

from config import AppConfig, load_config, save_config
from core.agent import Agent
from core.executor import TaskExecution
from core.planner import TaskList
from ui.panels.chat_panel import ChatPanel
from ui.panels.task_panel import TaskPanel

if TYPE_CHECKING:
    pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# cp1252 → Türkçe Unicode düzeltme tablosu
_TURKISH_FIX: dict[str, str] = {
    '\xfe': 'ş', '\xde': 'Ş',   # þ → ş, Þ → Ş
    '\xf0': 'ğ', '\xd0': 'Ğ',   # ð → ğ, Ð → Ğ
    '\xfd': 'ı', '\xdd': 'İ',   # ý → ı, Ý → İ
}


def _fix_turkish_key(event) -> None:
    char = event.char
    if char not in _TURKISH_FIX:
        return
    correct = _TURKISH_FIX[char]
    widget = event.widget

    def _do_fix():
        try:
            idx = widget.index("insert")
            prev = widget.index(f"{idx} - 1 chars")
            if widget.get(prev, idx) in _TURKISH_FIX:
                widget.delete(prev, idx)
                widget.insert(prev, correct)
                widget.mark_set("insert", f"{prev} + 1 chars")
        except Exception:
            pass

    widget.after_idle(_do_fix)


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, cfg: AppConfig, on_save):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("440x580")
        self.resizable(False, False)
        self._cfg = cfg
        self._on_save = on_save
        self._build()

    def _build(self):
        self.grid_columnconfigure(1, weight=1)

        text_fields = [
            ("Ollama URL", "ollama_base_url"),
            ("Planner Model", "planner_model"),
            ("Synthesizer Model", "synthesizer_model"),
            ("Embedding Model", "embedding_model"),
            ("Memory DB Path", "db_path"),
            ("Memory Top-K", "memory_top_k"),
            ("Shell Timeout (s)", "shell_timeout_seconds"),
            ("Max Retries", "max_task_retries"),
            ("Max History Turns", "max_conversation_history"),
            ("ReAct Max Iterations", "react_max_iterations"),
        ]
        bool_fields = [
            ("ReAct Mode", "use_react_loop"),
            ("Injection Protection", "prompt_injection_protection"),
        ]

        self._vars: dict[str, ctk.StringVar] = {}
        self._bool_vars: dict[str, ctk.BooleanVar] = {}

        row = 0
        # Section: General
        ctk.CTkLabel(self, text="GENEL", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=("#9CA3AF", "#6B7280")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 2))
        row += 1

        for label, attr in text_fields:
            ctk.CTkLabel(self, text=label, anchor="w").grid(
                row=row, column=0, sticky="w", padx=12, pady=5)
            var = ctk.StringVar(value=str(getattr(self._cfg, attr)))
            self._vars[attr] = var
            ctk.CTkEntry(self, textvariable=var).grid(
                row=row, column=1, sticky="ew", padx=12, pady=5)
            row += 1

        # Section: Toggles
        ctk.CTkLabel(self, text="SEÇENEKLER", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=("#9CA3AF", "#6B7280")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 2))
        row += 1

        for label, attr in bool_fields:
            ctk.CTkLabel(self, text=label, anchor="w").grid(
                row=row, column=0, sticky="w", padx=12, pady=5)
            bvar = ctk.BooleanVar(value=bool(getattr(self._cfg, attr)))
            self._bool_vars[attr] = bvar
            ctk.CTkSwitch(self, text="", variable=bvar, onvalue=True, offvalue=False).grid(
                row=row, column=1, sticky="w", padx=12, pady=5)
            row += 1

        ctk.CTkButton(self, text="Kaydet", command=self._save).grid(
            row=row, column=0, columnspan=2, pady=16)

    def _save(self):
        for attr, var in self._vars.items():
            val = var.get()
            field_type = type(getattr(self._cfg, attr))
            try:
                setattr(self._cfg, attr, field_type(val))
            except (ValueError, TypeError):
                pass
        for attr, bvar in self._bool_vars.items():
            setattr(self._cfg, attr, bvar.get())
        save_config(self._cfg)
        self._on_save(self._cfg)
        self.destroy()


class DesktopAgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Desktop Agent")
        self.geometry("1100x700")
        self.minsize(800, 500)

        self._cfg = load_config()
        self._agent = Agent(self._cfg)
        self._worker_thread: threading.Thread | None = None

        self._build_layout()
        # Tüm widget'larda Türkçe karakter düzeltmesi
        self.bind_all("<Key>", _fix_turkish_key, add=True)

    def _build_layout(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=5)  # chat — main content
        self.grid_columnconfigure(1, weight=2)  # task sidebar

        # ── Top bar ──────────────────────────────────────────────────
        topbar = ctk.CTkFrame(self, height=44, corner_radius=0)
        topbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        topbar.grid_columnconfigure(2, weight=1)  # spacer

        ctk.CTkLabel(
            topbar, text="Desktop Agent",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=8)

        self._model_var = ctk.StringVar(value=self._cfg.planner_model)
        self._model_menu = ctk.CTkOptionMenu(
            topbar, variable=self._model_var,
            values=[self._cfg.planner_model],
            command=self._on_model_change,
            width=180,
        )
        self._model_menu.grid(row=0, column=1, padx=(0, 4), pady=8)

        ctk.CTkButton(
            topbar, text="↻", width=32,
            fg_color="transparent", border_width=1,
            command=lambda: threading.Thread(target=self._refresh_models, daemon=True).start(),
        ).grid(row=0, column=2, padx=(0, 8), pady=8, sticky="w")

        # Workspace label (spacer column stretches)
        ws_frame = ctk.CTkFrame(topbar, fg_color="transparent")
        ws_frame.grid(row=0, column=3, padx=8, pady=4)
        ctk.CTkLabel(ws_frame, text="📁", font=ctk.CTkFont(size=13)).pack(side="left")
        self._ws_label = ctk.CTkLabel(
            ws_frame,
            text=self._short_path(self._cfg.workspace),
            font=ctk.CTkFont(size=11),
            text_color=("#9CA3AF", "#6B7280"),
            cursor="hand2",
        )
        self._ws_label.pack(side="left", padx=4)
        self._ws_label.bind("<Button-1>", lambda _: self._pick_workspace())

        ctk.CTkButton(
            topbar, text="⚙", width=36,
            fg_color="transparent", border_width=1,
            command=self._open_settings,
        ).grid(row=0, column=4, padx=4, pady=8)

        ctk.CTkButton(
            topbar, text="◐", width=36,
            fg_color="transparent", border_width=1,
            command=self._toggle_theme,
        ).grid(row=0, column=5, padx=(0, 12), pady=8)

        # ── Main: chat (left) + task sidebar (right) ──────────────────
        self._chat = ChatPanel(self, on_submit=self._submit_query,
                               on_new_chat=self._on_new_chat)
        self._chat.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self._chat.set_stop_callback(self._stop_agent)

        self._task_panel = TaskPanel(self)
        self._task_panel.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self._task_panel.set_approval_callbacks(
            on_approve=self._approve_plan,
            on_reject=self._reject_plan,
        )

        threading.Thread(target=self._refresh_models, daemon=True).start()

    def _refresh_models(self) -> None:
        models = self._agent.get_available_models()
        if models:
            def _apply(models=models):
                self._model_menu.configure(values=models)
                # Eğer mevcut seçim listede yoksa ilk modeli seç
                if self._model_var.get() not in models:
                    self._model_var.set(models[0])
                    self._on_model_change(models[0])
            self.after(0, _apply)
        else:
            self.after(0, lambda: self._model_menu.configure(
                values=["(Ollama bağlantısı yok)"]
            ))

    def _pick_workspace(self) -> None:
        path = filedialog.askdirectory(
            title="Workspace Seç",
            initialdir=self._cfg.workspace,
        )
        if path:
            self._agent.set_workspace(path)
            self._ws_label.configure(text=self._short_path(path))

    @staticmethod
    def _short_path(path: str, max_len: int = 30) -> str:
        p = Path(path)
        s = str(p)
        if len(s) > max_len:
            return "..." + s[-(max_len - 3):]
        return s

    def _on_model_change(self, model: str) -> None:
        if model.startswith("("):
            return
        self._cfg.planner_model = model
        self._cfg.synthesizer_model = model
        self._agent._planner._model = model
        self._agent._synthesizer._model = model
        self._agent._react_loop._model = model

    def _submit_query(self, query: str, attachments: list[str] | None = None) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._chat.add_user_message(query, attachments=attachments)
        self._chat.start_agent_message()
        self._chat.set_stop_enabled(True)
        self._task_panel.clear()

        def run():
            try:
                answer = self._agent.run(
                    user_query=query,
                    attachments=attachments or [],
                    on_plan_ready=lambda tl: self.after(0, self._on_plan_ready, tl),
                    on_task_update=lambda ex: self.after(0, self._on_task_update, ex),
                    on_output_chunk=lambda ch: self.after(0, self._chat.append_tool_chunk, ch),
                    on_answer_chunk=lambda ch: self.after(0, self._chat.append_agent_chunk, ch),
                    on_image=lambda b64: self.after(0, self._chat.show_screenshot, b64),
                    on_tool_start=lambda: self.after(0, self._chat.tool_start),
                    on_tool_done=lambda: self.after(0, self._chat.tool_done),
                    require_approval=True,
                )
                if not answer:
                    self.after(0, self._chat.append_agent_chunk, "(Plan iptal edildi.)")
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception("Agent run failed: %s", exc)
                self.after(0, self._chat.append_agent_chunk, f"(Hata: {exc})")
            finally:
                self.after(0, self._on_agent_done)

        self._worker_thread = threading.Thread(target=run, daemon=True)
        self._worker_thread.start()

    def _on_plan_ready(self, task_list: TaskList) -> None:
        self._task_panel.load_plan(task_list)
        # Stop button now acts as reject while waiting for approval
        self._chat.set_stop_enabled(True)

    def _approve_plan(self) -> None:
        self._agent.approve_plan()

    def _reject_plan(self) -> None:
        self._agent.reject_plan()
        self._task_panel.hide_approval()

    def _on_task_update(self, execution: TaskExecution) -> None:
        self._task_panel.update_task(execution)

    def _on_agent_done(self) -> None:
        self._chat.finish_agent_message()
        self._chat.set_stop_enabled(False)

    def _on_new_chat(self) -> None:
        self._agent.cancel()
        self._agent.clear_history()
        self._task_panel.clear()
        self._chat.set_stop_enabled(False)

    def _stop_agent(self) -> None:
        self._agent.cancel()
        self._task_panel.hide_approval()
        self._chat.set_stop_enabled(False)

    def _open_settings(self) -> None:
        SettingsDialog(self, self._cfg, on_save=self._on_settings_saved)

    def _on_settings_saved(self, new_cfg: AppConfig) -> None:
        self._cfg = new_cfg
        self._agent = Agent(new_cfg)

    def _toggle_theme(self) -> None:
        current = ctk.get_appearance_mode()
        ctk.set_appearance_mode("light" if current == "Dark" else "dark")
