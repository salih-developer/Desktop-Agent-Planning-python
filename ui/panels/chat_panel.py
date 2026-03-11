from __future__ import annotations

import os
from pathlib import Path
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk

from ui.widgets.message_bubble import MessageBubble
from ui.widgets.tool_output_block import ToolOutputBlock


class ChatPanel(ctk.CTkFrame):
    def __init__(self, parent, on_submit: Callable[[str, list[str]], None],
                 on_new_chat: Callable | None = None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_submit = on_submit
        self._on_new_chat = on_new_chat
        self._bubbles: list[MessageBubble] = []
        self._current_tool_block: ToolOutputBlock | None = None
        self._current_agent_bubble: MessageBubble | None = None
        self._attachments: list[str] = []

        self._build()

    def _build(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self._scroll = ctk.CTkScrollableFrame(self, label_text="")
        self._scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._scroll.grid_columnconfigure(0, weight=1)

        self._auto_scroll = True

        # Attachment chip area (hidden when empty)
        self._chip_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._chip_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        self._chip_frame.grid_columnconfigure(0, weight=1)
        self._chip_frame.grid_remove()  # hidden initially

        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        input_frame.grid_columnconfigure(1, weight=1)

        self._attach_btn = ctk.CTkButton(
            input_frame, text="📎", width=36,
            fg_color="transparent", border_width=1,
            command=self._pick_attachments,
        )
        self._attach_btn.grid(row=0, column=0, rowspan=2, padx=(0, 4), sticky="ns")

        self._input = ctk.CTkTextbox(input_frame, height=60, wrap="word", font=ctk.CTkFont(size=13))
        self._input.grid(row=0, column=1, columnspan=3, sticky="ew", pady=(0, 4))
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)

        self._send_btn = ctk.CTkButton(
            input_frame, text="Send", width=80, command=self._submit
        )
        self._send_btn.grid(row=1, column=1, sticky="w")

        self._stop_btn = ctk.CTkButton(
            input_frame, text="Stop ■", width=80,
            fg_color=("#DC2626", "#B91C1C"),
            hover_color=("#EF4444", "#DC2626"),
            state="disabled",
        )
        self._stop_btn.grid(row=1, column=2, padx=8, sticky="w")

        self._new_btn = ctk.CTkButton(
            input_frame, text="New Chat ＋", width=100, fg_color="transparent",
            border_width=1, command=self._new_chat
        )
        self._new_btn.grid(row=1, column=3, sticky="w")

    # ── Attachment management ──────────────────────────────────────────

    def _pick_attachments(self) -> None:
        """Open a dialog to pick files or a folder."""
        from tkinter import Menu
        menu = Menu(self, tearoff=0)
        menu.add_command(label="Dosya(lar) seç...", command=self._pick_files)
        menu.add_command(label="Klasör seç...", command=self._pick_folder)
        # Show menu near the button
        try:
            x = self._attach_btn.winfo_rootx()
            y = self._attach_btn.winfo_rooty() + self._attach_btn.winfo_height()
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _pick_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Dosya Seç",
            filetypes=[
                ("Tüm desteklenen", "*.pdf *.txt *.md *.py *.cs *.js *.ts *.json *.yaml *.yml *.xml *.html *.csv *.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                ("Görsel", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                ("PDF", "*.pdf"),
                ("Metin", "*.txt *.md"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        for p in paths:
            if p not in self._attachments:
                self._attachments.append(p)
        self._refresh_chips()

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(title="Klasör Seç")
        if path and path not in self._attachments:
            self._attachments.append(path)
        self._refresh_chips()

    def _refresh_chips(self) -> None:
        """Rebuild the chip row from self._attachments."""
        for w in self._chip_frame.winfo_children():
            w.destroy()

        if not self._attachments:
            self._chip_frame.grid_remove()
            return

        self._chip_frame.grid()
        for i, path in enumerate(self._attachments):
            name = Path(path).name or path
            chip = ctk.CTkFrame(self._chip_frame, fg_color=("#E5E7EB", "#374151"), corner_radius=6)
            chip.pack(side="left", padx=(0, 4), pady=2)

            icon = "📁" if os.path.isdir(path) else "📄"
            ctk.CTkLabel(chip, text=f"{icon} {name}", font=ctk.CTkFont(size=11)).pack(side="left", padx=(6, 2))

            idx = i  # capture for closure
            ctk.CTkButton(
                chip, text="✕", width=20, height=20,
                fg_color="transparent",
                hover_color=("#F87171", "#991B1B"),
                font=ctk.CTkFont(size=10),
                command=lambda i=idx: self._remove_attachment(i),
            ).pack(side="left", padx=(0, 4))

    def _remove_attachment(self, idx: int) -> None:
        if 0 <= idx < len(self._attachments):
            self._attachments.pop(idx)
        self._refresh_chips()

    # ── Scroll ────────────────────────────────────────────────────────

    def _scroll_to_bottom(self) -> None:
        if not self._auto_scroll:
            return
        canvas = self._scroll._parent_canvas
        canvas.after_idle(lambda: canvas.after_idle(
            lambda: canvas.yview_moveto(1.0)
        ))

    # ── Public API ────────────────────────────────────────────────────

    def set_stop_callback(self, cb: Callable) -> None:
        self._stop_btn.configure(command=cb)

    def set_stop_enabled(self, enabled: bool) -> None:
        self._stop_btn.configure(state="normal" if enabled else "disabled")
        self._send_btn.configure(state="disabled" if enabled else "normal")

    def add_user_message(self, text: str, attachments: list[str] | None = None) -> None:
        self._auto_scroll = True
        content = text
        if attachments:
            names = ", ".join(Path(p).name or p for p in attachments)
            content = f"[📎 {names}]\n{text}" if text else f"[📎 {names}]"
        bubble = MessageBubble(self._scroll, role="user", content=content)
        bubble.pack(fill="x", pady=2)
        self._scroll_to_bottom()

    def start_agent_message(self) -> None:
        self._auto_scroll = True
        self._current_tool_block = ToolOutputBlock(self._scroll)
        self._current_tool_block.pack(fill="x", padx=8, pady=(2, 0))
        self._current_agent_bubble = None  # created lazily on first answer chunk
        self._scroll_to_bottom()

    def append_tool_chunk(self, chunk: str) -> None:
        if self._current_tool_block:
            self._current_tool_block.append(chunk)
            self._scroll_to_bottom()

    def append_agent_chunk(self, chunk: str) -> None:
        if self._current_agent_bubble is None:
            self._current_agent_bubble = MessageBubble(
                self._scroll, role="assistant", content=""
            )
            self._current_agent_bubble.pack(fill="x", pady=(4, 2))
        self._current_agent_bubble.append(chunk)
        self._scroll_to_bottom()

    def finish_agent_message(self) -> None:
        self._current_tool_block = None
        self._current_agent_bubble = None
        self._scroll_to_bottom()

    def clear(self) -> None:
        for w in self._scroll.winfo_children():
            w.destroy()
        self._bubbles.clear()
        self._current_tool_block = None
        self._current_agent_bubble = None

    # ── Internal ──────────────────────────────────────────────────────

    def _on_enter(self, event) -> str:
        if not (event.state & 0x1):  # Shift not held
            self._submit()
            return "break"
        return ""

    def _submit(self) -> None:
        text = self._input.get("1.0", "end").strip()
        attachments = list(self._attachments)
        if text or attachments:
            self._input.delete("1.0", "end")
            self._attachments.clear()
            self._refresh_chips()
            self._on_submit(text, attachments)

    def _new_chat(self) -> None:
        self.clear()
        self._attachments.clear()
        self._refresh_chips()
        if self._on_new_chat:
            self._on_new_chat()
